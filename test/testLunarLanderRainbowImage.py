import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt

GPU_ID = 1
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64
STACK_SIZE = 4
isCreateVideo = False

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

def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()

def test_model(model_path, n_episodes=10, max_t=1000):
    env = gym.make('LunarLander-v3', render_mode="rgb_array")
    action_size = env.action_space.n

    model = DuelingQNetworkCNN(action_size, seed=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    data = {"Episode": [], "Total Reward": [], "Avg100 Reward": []}
    done_counter = 0
    frame_stacker = FrameStacker(STACK_SIZE, IMG_SIZE)
    sum = 0
    scores = []
    for i_episode in range(1, n_episodes + 1):
        frame_stacker.reset()
        _, _ = env.reset()
        frame = preprocess_frame(env.render())
        frame_stacker.add_frame(frame)
        state = frame_stacker.get_stacked_frames()
        if isCreateVideo:
            frames = []
        score = 0
        scores_window = deque(maxlen=100)
        for t in range(max_t):
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_t)
            action = q_values.argmax(dim=1).item()

            _, reward, terminated, truncated, _ = env.step(action)
            env_render = env.render()
            if isCreateVideo:
                frames.append(env_render)
            next_frame = preprocess_frame(env_render)
            frame_stacker.add_frame(next_frame)
            next_state = frame_stacker.get_stacked_frames()
            score += reward
            state = next_state
            if reward >=100:
                print(f"Episode: {i_episode} reward: {reward}")
            if terminated or truncated:
                break
        scores.append(score)
        if isCreateVideo:
            create_video(frames, 30, str(i_episode))
        scores_window.append(score)
        data["Episode"].append(i_episode)
        data["Total Reward"].append(score)
        data["Avg100 Reward"].append(np.mean(scores_window))

        if score > 200:
            done_counter += 1
        print(f"Episode {i_episode}: Total Reward = {score:.2f} Done = {done_counter} Solve rate = {done_counter/i_episode}")
        sum = sum + score
    print(f"Average score: {sum/1000}")
    print(f"Success rate{done_counter/1000}")
    df = pd.DataFrame(data)
    df.to_csv("training_data_full_rainbow.csv", index=False)
    env.close()
    plt.plot(scores, linewidth = 1)
    plt.title("Scores in Test Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()


if __name__ == "__main__":
    model_path = r"C:\LunarLanderRL\rainbow_checkpoint_600000.pth"
    test_model(model_path, 1000)