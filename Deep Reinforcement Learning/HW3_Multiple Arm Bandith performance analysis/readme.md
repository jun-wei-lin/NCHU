# HW3: Explore and Exploit for Multi-Armed Bandit Problem

本作業實作並比較四種多臂強盜（Multi-Armed Bandit, MAB）演算法：**Epsilon-Greedy**、**UCB1**、**Softmax (Boltzmann)**、**Thompson Sampling**。每個演算法皆包含公式推導、ChatGPT prompt、Python 程式碼與視覺化結果。

---

## 📌 問題設定

- 臂數（Arms）：10
- 每臂真實成功機率：隨機初始化
- 每次實驗時間步長：500
- 重複實驗次數：200

---

## 🔢 1. Epsilon-Greedy

### 📘 演算法公式

$$
a_t =
\\begin{cases}
\\arg\\max_a Q_t(a), & \\text{以機率 } 1 - \\varepsilon \\\\
\\text{隨機選擇 } a, & \\text{以機率 } \\varepsilon
\\end{cases}
$$

$$
Q_{t+1}(a) = Q_t(a) + \\frac{1}{N_t(a)} (R_t - Q_t(a))
$$

### 💡 ChatGPT Prompt

> 「請說明 epsilon-greedy 多臂強盜算法如何在探索與利用間取得平衡，並推導其更新公式。」

---

## 🔢 2. UCB1 (Upper Confidence Bound)

### 📘 演算法公式

$$
a_t = \\arg\\max_a \\left[ Q_t(a) + \\sqrt{\\frac{2 \\ln t}{N_t(a)}} \\right]
$$

### 💡 ChatGPT Prompt

> 「請解釋 UCB1 演算法如何利用上界項進行動作選擇，達到探索與利用的權衡。」

---

## 🔢 3. Softmax (Boltzmann Exploration)

### 📘 演算法公式

$$
P(a_t = a) = \\frac{\\exp(Q_t(a)/\\tau)}{\\sum_b \\exp(Q_t(b)/\\tau)}
$$

### 💡 ChatGPT Prompt

> 「請說明 Softmax 策略是如何根據值函數動態調整每個行動的選擇機率，並控制探索程度。」

---

## 🔢 4. Thompson Sampling

### 📘 演算法公式

$$
\\theta_a \\sim \\text{Beta}(S_a + 1, F_a + 1), \quad
a_t = \\arg\\max_a \\theta_a
$$

### 💡 ChatGPT Prompt

> 「請解釋 Thompson Sampling 是如何透過貝氏推論來達成探索與利用的平衡。」

---

## 💻 Python 程式碼摘要

以下程式碼模擬四種演算法在相同 MAB 環境下的表現：

```python
def simulate_mab(algo):
    # 初始化參數
    ...
    for t in range(horizon):
        # 各種演算法的行動選擇邏輯
        ...
        # 回饋與估計值更新
        ...
```
## 📊 實驗結果圖

比較四種演算法於 500 個時間步內的平均累積獎勵表現：

![output](https://github.com/user-attachments/assets/a4abedb4-58ae-4a22-9eb2-4e5f753f6ead)


---

## 🧠 結果分析
![123](https://github.com/user-attachments/assets/98631f55-a2ab-4df0-abcd-756de57c17f0)
---

## 📁 專案結構
 - 📦 HW3-MAB
 - ┣ 📜 README.md
 - ┣ 📜 mab_simulation.py
 - ┣ 📊 mab_result_plot.png

---

## 📚 參考資料

[Introduction to Thompson Sampling: the Bernoulli bandit](https://gdmarmerola.github.io/ts-for-bernoulli-bandit/)

---

