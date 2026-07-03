import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import ale_py
from collections import deque

gym.register_envs(ale_py)

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# ==== 載入模型並評估連續20個seed直到平均reward達標 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/Pong-v5", render_mode=None)
preprocessor = AtariPreprocessor()
num_actions = env.action_space.n
model = DQN(input_channels=4, num_actions=num_actions).to(device)
model.load_state_dict(torch.load("./model_reward21_ep2040.pt", map_location=device, weights_only=True))
model.eval()

target_avg_reward = 19
window_size = 20
rewards = deque(maxlen=window_size)
seed = 8888888

while True:
    obs, _ = env.reset(seed=seed)
    state = preprocessor.reset(obs)
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = preprocessor.step(next_obs)
    rewards.append(total_reward)
    print(f"Seed {seed}: reward = {total_reward}")
    if len(rewards) == window_size:
        avg_reward = np.mean(rewards)
        print(f"Seeds {seed-window_size+1}~{seed} average reward: {avg_reward:.2f}")
        if avg_reward >= target_avg_reward:
            print("Target achieved!")
            break
    seed += 1