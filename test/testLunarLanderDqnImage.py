import gymnasium as gym
import torch
import numpy as np
import cv2
from collections import deque
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMG_SIZE = 64
STACK_SIZE = 4

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

def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()

def test_model(model_path, n_episodes=10, max_t=1000):
    env = gym.make('LunarLander-v3', render_mode="rgb_array")
    action_size = env.action_space.n

    model = QNetwork(env.observation_space.shape[0], action_size, seed=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    data = {"Episode": [], "Total Reward": [], "Avg100 Reward": []}
    done_counter = 0
    frame_stacker = FrameStacker(STACK_SIZE, IMG_SIZE)
    sum = 0
    for i_episode in range(1, n_episodes + 1):
        frame_stacker.reset()
        obs, _ = env.reset()
        frame = preprocess_frame(env.render())
        frame_stacker.add_frame(frame)
        state = frame_stacker.get_stacked_frames()
        frames = []
        score = 0
        scores_window = deque(maxlen=100)
        for t in range(max_t):
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_t)
            action = q_values.argmax(dim=1).item()

            next_obs, reward, done, truncated, _ = env.step(action)
            env_render = env.render()
            next_frame = preprocess_frame(env_render)
            frame_stacker.add_frame(next_frame)
            next_state = frame_stacker.get_stacked_frames()
            frames.append(env_render)
            score += reward
            state = next_state
            if reward >=100:
                print(f"Episode: {i_episode} reward: {reward}")
            if done or truncated:
                break
        scores_window.append(score)
        data["Episode"].append(i_episode)
        data["Total Reward"].append(score)
        data["Avg100 Reward"].append(np.mean(scores_window))
        create_video(frames,30, str(i_episode))
        if score > 200:
            done_counter += 1
        print(f"Episode {i_episode}: Total Reward = {score:.2f} Done = {done_counter}")
        sum = sum + score
    print(sum/100)
    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    env.close()

if __name__ == "__main__":
    model_path = r"C:\LunarLanderRL\dqn_checkpoint_330000.pth"
    test_model(model_path, 100)