# HW4-2: Enhanced DQN – Double DQN for Player Mode 🎮🧠

本作業在隨機起點（player mode）環境中，實作與測試 **Double DQN**，目標是改善 Naive DQN 的 Q-value 高估問題，提升學習穩定性與泛化能力。

---

## 📌 任務目標

- ✅ 實作 Double DQN 增強版
- ✅ 應用於 player mode 環境（起點隨機）
- ✅ 分析與 Naive DQN 的差異
- ✅ 評估訓練穩定性與測試成功率

---

## 🧠 Double DQN 架構重點

- 引入 **目標網路 target_net**，與主網路 policy_net 分開
- 用主網路選動作 (`argmax`)，用目標網路估值（`gather`）
- 每 N 回合同步一次 target_net，降低 Q-value 高估問題

以下為核心更新邏輯：

```python
# Double DQN 更新邏輯
next_actions = policy_net(next_states).argmax(1)
next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
target = rewards + gamma * next_q_values * (1 - dones)
```

---

## ⚙️ 執行方式

### 套件安裝

請先安裝所需套件：

```bash
pip install torch numpy
```

（若你使用 OpenAI Gym 環境，也可額外安裝）

```bash
pip install gym
```

---

### 執行訓練與測試

```bash
python hw4_2_double_dqn_player.py
```

成功執行後，將看到如下輸出：

```
Episode 97/100 完成
Episode 98/100 完成
Episode 99/100 完成
Episode 100/100 完成
測試 Episode 1: 總回報 = 1.0
測試 Episode 2: 總回報 = 11.0
測試 Episode 3: 總回報 = 5.0
測試 Episode 4: 總回報 = 1.0
測試 Episode 5: 總回報 = 1.0
```
![image](https://github.com/user-attachments/assets/cc1122d2-e5f2-4a9b-86ae-31714e56f355)

---

## 🧪 實驗結果（player mode）

- 訓練 Episode 數：100
- 測試回合數：5
- 測試結果顯示，在 player mode 隨機起點下，Double DQN 仍能學會穩定策略，雖有回報差異，但大多成功完成任務。

---

## 📂 檔案說明

| 檔案名稱                    | 說明                                   |
|-----------------------------|----------------------------------------|
| `hw4_2_double_dqn_player.py` | Double DQN 實作與訓練流程              |
| `requirements.txt`           | Python 執行環境所需套件（與 HW4-1 相同）|

---

## 📝 心得與反思

本作業讓我深入理解 Q-learning 過度估計的問題，以及為何需要使用 Double DQN 架構進行修正。  
透過 player mode 的隨機起點設計，更能測試 agent 的泛化能力與穩定性。  
實作過程中也學會如何在 PyTorch 中切分主網路與目標網路，並使用 `.gather()` 處理動作選擇與評估分離。

