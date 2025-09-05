import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()

def test_trained_model(env, model_path, n_episodes=10, render=True):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = QNetwork(state_size, action_size, seed=0)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    done_counter = 0
    scores = []
    scores_window = deque(maxlen=100)
    data = {"Episode": [], "Total Reward": [], "Avg100 Reward": []}
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        frames = []
        while not done:
            if render:
                env.render()

            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = model(state)

            action = np.argmax(action_values.cpu().data.numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            if reward  >=100:
                print(reward)
            frames.append(env.render())
        data["Episode"].append(i_episode)
        data["Total Reward"].append(score)
        data["Avg100 Reward"].append(np.mean(scores_window))
        scores.append(score)
        scores_window.append(score)
        if score > 200:
            done_counter += 1
        print(f"Episode {i_episode}/{n_episodes}: Score = {score:.2f} Done: {done_counter} Succes rate: {done_counter*100/i_episode}") 
    df = pd.DataFrame(data)
    df.to_csv("training_data_sensor.csv", index=False)
    env.close()
    print(f"\nAverage Score over {n_episodes} episodes: {np.mean(scores):.2f}")
    return scores

model_path =  r"C:\LunarLanderRL\999_checkpoint.pth"

env = gym.make('LunarLander-v3', render_mode="rgb_array")

test_scores = test_trained_model(env, model_path, n_episodes=1000, render=True)

plt.plot(test_scores)
plt.title("Scores in Test Episodes")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
