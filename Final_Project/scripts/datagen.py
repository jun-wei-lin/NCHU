import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RealisticDiscountEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)  # 折扣: [0%, 5%, 10%, 20%]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.state = self._sample_customer()
        return self.state, {}

    def step(self, action):
        discount = [0.0, 0.05, 0.1, 0.2][action]
        age, region, has_history = self.state

        # 更現實的轉換率模型
        prob = (
            0.05 +  # base
            0.01 * has_history +
            0.002 * (60 - age) +  # 年紀越小越願意買
            2.5 * discount  # 折扣提升購買意願
        )
        prob = min(max(prob, 0), 1)  # 限制在 [0, 1]
        converted = np.random.rand() < prob

        reward = 100 * (1 - discount) - 30 if converted else 0
        next_state = self._sample_customer()
        done = False
        info = {'converted': int(converted)}

        self.state = next_state
        return next_state, reward, done, False, info

    def _sample_customer(self):
        age = np.random.randint(20, 61) / 60.0  # 正規化
        region = np.random.randint(0, 4) / 3.0
        history = np.random.choice([0, 1])
        return np.array([age, region, history], dtype=np.float32)

def generate_realistic_discount_data(env, n_steps=1000, csv_file="real_discount_data.csv"):
    records = []
    state, _ = env.reset()

    for _ in range(n_steps):
        action = env.action_space.sample()  # 也可以換成策略選擇
        next_state, reward, done, _, info = env.step(action)

        age = state[0] * 60
        region = int(state[1] * 3)
        history = int(state[2])
        discount = [0.0, 0.05, 0.10, 0.20][action]

        records.append({
            "age": int(age),
            "region": region,
            "history": history,
            "action": action,
            "discount": discount,
            "converted": info['converted'],
            "reward": reward
        })

        state = next_state

    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    print(f"✅ 已生成並儲存 {n_steps} 筆模擬資料到 {csv_file}")

env = RealisticDiscountEnv()
generate_realistic_discount_data(env, n_steps=5000)