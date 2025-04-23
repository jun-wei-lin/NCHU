
import numpy as np
import matplotlib.pyplot as plt

# 固定隨機種子
np.random.seed(42)

# 參數設置
n_arms = 10
horizon = 500
n_runs = 200
true_probs = np.random.rand(n_arms)
epsilon = 0.1
temperature = 0.1

def simulate_mab(algo):
    rewards = np.zeros(horizon)
    estimates = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    successes = np.zeros(n_arms)
    failures = np.zeros(n_arms)
    
    for t in range(horizon):
        if algo == 'epsilon_greedy':
            arm = np.random.randint(n_arms) if np.random.rand() < epsilon else np.argmax(estimates)
        elif algo == 'ucb1':
            arm = t if t < n_arms else np.argmax(estimates + np.sqrt(2 * np.log(t + 1) / counts))
        elif algo == 'softmax':
            exp_values = np.exp(estimates / temperature)
            probs = exp_values / exp_values.sum()
            arm = np.random.choice(n_arms, p=probs)
        elif algo == 'thompson':
            samples = np.random.beta(successes + 1, failures + 1)
            arm = np.argmax(samples)
        else:
            raise ValueError("Unsupported algorithm")
        
        reward = np.random.rand() < true_probs[arm]
        rewards[t] = reward

        if algo == 'thompson':
            successes[arm] += reward
            failures[arm] += 1 - reward
        else:
            counts[arm] += 1
            estimates[arm] += (reward - estimates[arm]) / counts[arm]
    
    return rewards

# 執行模擬並收集數據
algos = ['epsilon_greedy', 'ucb1', 'softmax', 'thompson']
results = {}

for algo in algos:
    cum_rewards = np.zeros(horizon)
    for _ in range(n_runs):
        r = simulate_mab(algo)
        cum_rewards += np.cumsum(r)
    results[algo] = cum_rewards / n_runs

# 儲存圖表
plt.figure(figsize=(8, 5))
for algo in algos:
    plt.plot(results[algo], label=algo)
plt.title("Multi-Armed Bandit Algorithm Comparison")
plt.xlabel("Time Step")
plt.ylabel("Average Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mab_result_plot.png")
plt.show()
