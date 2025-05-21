import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 這是你需要準備的環境，請確認有正確實作或替換
# 例如 from your_env_file import StaticGridWorldEnv
class DummyEnv:
    def __init__(self):
        self.observation_space = type('Space', (), {'shape': (4,)})  # 假設觀測值是4維
        self.action_space = type('Space', (), {'n': 2})  # 假設有2個動作
    def reset(self):
        return np.zeros(4)
    def step(self, action):
        next_state = np.zeros(4)
        reward = 1.0
        done = True
        return next_state, reward, done, {}
        
env = DummyEnv()  # 替換為 StaticGridWorldEnv() 如果你已經有環境的話

# Q-Network 定義
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

# 經驗回放緩衝區
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

# 訓練流程
def train_naive_dqn(env, episodes=50, gamma=0.99, epsilon=0.1, batch_size=16, buffer_capacity=500):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters())
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_capacity)

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)

        done = False
        while not done:
            # ε-greedy 策略
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
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

                batch_states = torch.stack(batch_states)
                batch_actions = torch.LongTensor(batch_actions)
                batch_rewards = torch.FloatTensor(batch_rewards)
                batch_next_states = torch.stack(batch_next_states)
                batch_dones = torch.FloatTensor(batch_dones)

                q_values = policy_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                next_q_values = policy_net(batch_next_states).max(1)[0]
                expected_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)

                loss = criterion(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode + 1}/{episodes} 完成")

    return policy_net

# 測試流程
def test_policy(env, policy_net, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
            state, reward, done, _ = env.step(action)
            state = torch.FloatTensor(state)
            total_reward += reward
        print(f"測試 Episode {episode + 1}: 總回報 = {total_reward}")

# 執行訓練與測試
trained_policy = train_naive_dqn(env)
test_policy(env, trained_policy)
