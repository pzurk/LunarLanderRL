import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import cv2
import random
import pandas as pd

GPU_ID = 1
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

env = gym.make('LunarLander-v3', render_mode="rgb_array")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4
UPDATE_EVERY = 4
STACK_SIZE = 4
IMG_SIZE = 64

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        with torch.no_grad():
            test_in = torch.zeros(1, STACK_SIZE, IMG_SIZE, IMG_SIZE)
            test_out = self._forward_conv(test_in)
            conv_out_size = test_out.view(1, -1).size(1)
        self.conv_out_size = conv_out_size

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def _forward_conv(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x

    def forward(self, state):
        x = self._forward_conv(state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            self.learn(self.memory.sample(), GAMMA)
    
    def act(self, state, eps=0.0):
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
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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

    def preprocess_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9998):
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)
    scores = []
    data = {"Episode":[], "Total Reward": [], "Avg100 Reward": [], "Epsilon":[]}
    scores_window = deque(maxlen=100)
    frame_stacker = FrameStacker(STACK_SIZE, IMG_SIZE)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        frame_stacker.reset()
        obs, _ = env.reset()
        
        frame = frame_stacker.preprocess_frame(env.render())
        frame_stacker.add_frame(frame)
        state = frame_stacker.get_stacked_frames()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            _,  reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frame = frame_stacker.preprocess_frame(env.render())
            frame_stacker.add_frame(next_frame)
            next_state = frame_stacker.get_stacked_frames()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        if i_episode%5000 == 0:
            torch.save(agent.qnetwork_local.state_dict(), f"dqn_checkpoint_{i_episode}.pth")

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        data["Episode"].append(i_episode)
        data["Total Reward"].append(score)
        data["Avg100 Reward"].append(np.mean(scores_window))
        data["Epsilon"].append(eps)
        
        print(f"Episode {i_episode}  Score: {score:.2f}  Avg100: {np.mean(scores_window):.2f}  eps: {eps:.3f}")        

        if np.mean(scores_window) >= 200.0:
            print(f"\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}")

        if i_episode%5000 == 0:
            df = pd.DataFrame(data)
            df.to_csv(f"training_data_dqn_{i_episode}.csv", index = False)
            data["Episode"].clear()
            data["Total Reward"].clear()
            data["Avg100 Reward"].clear()
            data["Epsilon"].clear()
    
    df = pd.DataFrame(data)
    df.to_csv("training_data_dqn.csv", index = False)
    torch.save(agent.qnetwork_local.state_dict(), "dqn_checkpoint.pth")
    return scores

if __name__ == "__main__":
    scores = dqn()
    print("Done!")