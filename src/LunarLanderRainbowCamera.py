import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import cv2
import os
import math
import pandas as pd

GPU_ID = 0
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print("Device:", device)

env = gym.make('LunarLander-v3', render_mode="human")
print('State shape: Image-based')
print('Number of actions: ', env.action_space.n)

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-3
LR = 5e-5
UPDATE_EVERY = 4
IMG_SIZE = 64
STACK_SIZE = 4
N_STEPS = 3
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 1e5
EPSILON_PER = 1e-2

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(std * self.sigma_init)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(std * self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

class DuelingQNetworkCNN(nn.Module):
    def __init__(self, action_size, seed):
        super(DuelingQNetworkCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        with torch.no_grad():
            test_in = torch.zeros(1, STACK_SIZE, IMG_SIZE, IMG_SIZE)
            test_out = self._forward_conv(test_in)
            conv_out_size = test_out.view(1, -1).size(1)
        self.conv_out_size = conv_out_size

        self.fc_shared = NoisyLinear(conv_out_size, 512)

        self.value = NoisyLinear(512, 1)

        self.advantage = NoisyLinear(512, action_size)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, state):
        x = self._forward_conv(state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        v = self.value(x)
        a = self.advantage(x)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q

    def reset_noise(self):
        self.fc_shared.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.buffer = []
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done, td_error=1.0):
        max_prio = self.priorities.max() if self.buffer else 1.0
        idx = self.pos % self.buffer_size
        e = self.experience(state, action, reward, next_state, done)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[idx] = e

        self.priorities[idx] = max(max_prio, td_error)
        self.pos += 1

    def sample(self, beta=0.4):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = prios.sum()
        probs = prios / total

        indices = np.random.choice(len(prios), self.batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]

        N = len(prios)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        states = torch.from_numpy(np.stack([e.state for e in experiences], axis=0)).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences], axis=0)).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        weights = torch.from_numpy(weights).float().to(device)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = (abs(prio) + EPSILON_PER) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class NstepHelper:
    def __init__(self, n_steps, gamma):
        self.n_steps = n_steps
        self.gamma = gamma
        self.deque = deque()

    def add(self, state, action, reward, done):
        self.deque.append((state, action, reward, done))

    def get_n_step(self, next_state):
        cum_reward = 0
        for i, (_, _, r, d) in enumerate(self.deque):
            cum_reward += (self.gamma ** i) * r
            if d:
                break

        state, action, _, _ = self.deque[0]
        done = self.deque[-1][3]
        return (state, action, cum_reward, next_state, done)

    def pop_left(self):
        self.deque.popleft()

    def __len__(self):
        return len(self.deque)

def preprocess_frame(frame):
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

class FrameStacker:
    def __init__(self, stack_size, img_size):
        self.stack_size = stack_size
        self.img_size = img_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self):
        self.frames.clear()

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_stacked_frames(self):
        while len(self.frames) < self.stack_size:
            zero_frame = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
            self.frames.append(zero_frame)
        return np.concatenate(self.frames, axis=0)

class RainbowAgent:
    def __init__(self, action_size, seed):
        self.action_size = action_size
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.qnetwork_local = DuelingQNetworkCNN(action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetworkCNN(action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, ALPHA, seed)

        self.t_step = 0
        self.nstep_helper = NstepHelper(N_STEPS, GAMMA)

    def step(self, state, action, reward, next_state, done, beta):
        self.nstep_helper.add(state, action, reward, done)

        if len(self.nstep_helper) >= N_STEPS:
            n_state, n_action, n_reward, n_next_state, n_done = self.nstep_helper.get_n_step(next_state)
            self.memory.add(n_state, n_action, n_reward, n_next_state, n_done, td_error=1.0)
            self.nstep_helper.pop_left()

        if done:
            while len(self.nstep_helper) > 0:
                n_state, n_action, n_reward, n_next_state, n_done = self.nstep_helper.get_n_step(next_state)
                self.memory.add(n_state, n_action, n_reward, n_next_state, n_done, td_error=1.0)
                self.nstep_helper.pop_left()

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(beta=beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        self.qnetwork_local.reset_noise()

        if random.random() > eps:
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                q_values = self.qnetwork_local(state_t)
            self.qnetwork_local.train()
            action = q_values.argmax(dim=1).item()
        else:
            action = random.randrange(self.action_size)
        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, indices, weights = experiences

        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        Q_local_next = self.qnetwork_local(next_states).detach()
        best_actions = Q_local_next.argmax(dim=1, keepdim=True)

        Q_target_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        y = rewards + (gamma ** N_STEPS) * Q_target_next * (1 - dones)

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        td_errors = y - Q_expected

        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(Q_expected, y, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors_cpu = td_errors.detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors_cpu)

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

def dqn(n_episodes=200000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9998):
    calibrate_camera()
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    agent = RainbowAgent(action_size=env.action_space.n, seed=0)
    scores = []
    scores_window = deque(maxlen=100)
    data = {"Episode":[], "Total Reward": [], "Avg100 Reward": [], "Epsilon":[]}
    frame_stacker = FrameStacker(STACK_SIZE, IMG_SIZE)

    eps = eps_start
    global_step = 0
    reward_count = 0
    solved = 0
    max_avg = -500
    for i_episode in range(1, n_episodes+1):
        frame_stacker.reset()
        obs, _ = env.reset()
        ret, camera_frame = cap.read()
        frame = preprocess_frame(camera_frame)
        frame_stacker.add_frame(frame)
        state = frame_stacker.get_stacked_frames()
        score = 0

        for t in range(max_t):
            global_step += 1
            beta = min(1.0, BETA_START + (1.0 - BETA_START)*(global_step / BETA_FRAMES))

            action = agent.act(state)
            next_obs, reward, done, truncated, _ = env.step(action)

            ret, camera_frame = cap.read()
            next_frame = preprocess_frame(camera_frame)
            frame_stacker.add_frame(next_frame)
            next_state = frame_stacker.get_stacked_frames()
            if reward >= 100:
                reward_count += 1

            agent.step(state, action, reward, next_state, done or truncated, beta)
            state = next_state
            score += reward

            if done or truncated:
                break

        if i_episode % 5000 == 0:
            torch.save(agent.qnetwork_local.state_dict(), f"rainbow_checkpoint_{i_episode}.pth")
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        if (np.mean(scores_window) > max_avg):
            max_avg = np.mean(
                scores_window)
        data["Episode"].append(i_episode)
        data["Total Reward"].append(score)
        data["Avg100 Reward"].append(np.mean(scores_window))
        data["Epsilon"].append(eps)

        print(f"Episode {i_episode}  Score: {score:.2f}  Avg100: {np.mean(scores_window):.2f}  eps: {eps:.3f}  beta:{beta:.3f} reward_count: {reward_count} solved: {solved} max_avg: {max_avg}")

        if np.mean(scores_window) >= 200.0:
            print(f"\nEnvironment solved in {i_episode-100} episodes! Average Score: {np.mean(scores_window):.2f}")
            solved += 1
            
            if solved % 1000 or solved == 0 or (solved % 100 == 0 and solved <=1000):
                torch.save(agent.qnetwork_local.state_dict(), f"rainbow_checkpoint_solved_{solved}.pth")

        if i_episode % 5000 == 0:
            df = pd.DataFrame(data)
            df.to_csv(f"training_data_{i_episode}.csv", index=False)
            data["Episode"].clear()
            data["Total Reward"].clear()
            data["Avg100 Reward"].clear()
            data["Epsilon"].clear()

    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    torch.save(agent.qnetwork_local.state_dict(), "rainbow_checkpoint.pth")

    return scores

def calibrate_camera():    
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Nie udało się otworzyć kamery!")
        return False

    tmp_env = gym.make('LunarLander-v3', render_mode = "human")

    print("Kalibracja kamery... Ustaw kamerę i naciśnij 'q', aby rozpocząć uczenie.")
    tmp_env.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd pobierania obrazu z kamery!")
            break
        tmp_env.render()
        next_obs, reward, done, truncated, _ = tmp_env.step(random.randint(0,3))
        
        cv2.putText(frame, "Kalibracja: nacisnij 'q', aby rozpoczac", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Podglad kamery", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Kalibracja zakonczona. Rozpoczynamy uczenie...")
            break

        if done or truncated:
            tmp_env.reset()

    tmp_env.close()
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    scores = dqn()
    print("Done!")