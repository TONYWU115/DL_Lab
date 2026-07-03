import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    def forward(self, x):
        return self.network(x)

class IdentityPreprocessor:
    def reset(self, obs):
        return obs

    def step(self, obs):
        return obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode=None)
preprocessor = IdentityPreprocessor()
num_actions = env.action_space.n
model = DQN(num_actions).to(device)
# model.load_state_dict(torch.load("./results/best_model.pt", map_location=device))
model.load_state_dict(torch.load("./LAB5_s11182048_task1.pt", map_location=device, weights_only=True))
model.eval()

rewards = []
for seed in range(20):
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
print(f"Average reward: {np.mean(rewards)}")