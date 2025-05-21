import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 🧪 模擬的 player mode 環境
class DummyPlayerEnv:
    def __init__(self):
        self.observation_space = type('Space', (), {'shape': (4,)})
        self.action_space = type('Space', (), {'n': 2})
    def reset(self):
        return np.random.rand(4)  # 起點隨機
    def step(self, action):
        next_state = np.random.rand(4)
        reward = 1.0 if random.random() > 0.2 else 0.0
        done = random.random() > 0.8
        return next_state, reward, done, {}

env = DummyPlayerEnv()

# Q-Network 架構
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# 🧠 Double DQN 訓練
def train_double_dqn(env, episodes=100, gamma=0.99, epsilon=0.1, batch_size=32, buffer_capacity=1000, target_update_freq=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())  # 初始同步
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    for episode in range(episodes):
        state = torch.FloatTensor(env.reset())
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.stack(next_states)
                dones = torch.FloatTensor(dones)

                # Double DQN Target 計算
                next_actions = policy_net(next_states).argmax(1)
                next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + gamma * next_q_values * (1 - dones)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = criterion(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新 target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{episodes} 完成")

    return policy_net

# 測試流程
def test_policy(env, policy_net, episodes=5):
    for episode in range(episodes):
        state = torch.FloatTensor(env.reset())
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
            state, reward, done, _ = env.step(action)
            state = torch.FloatTensor(state)
            total_reward += reward
        print(f"測試 Episode {episode + 1}: 總回報 = {total_reward}")

# ⏱ 執行訓練與測試
agent = train_double_dqn(env)
test_policy(env, agent)
