# HW4-1: Naive DQN for Static Mode 🧠🎯

本專案實作 Naive DQN 強化學習演算法，並應用於靜態格子環境（Static GridWorld）中，完成基本訓練與測試流程，驗證 Q-Learning 的效果。

---

## 📌 任務目標

> ✅ 實作基本 DQN 演算法  
> ✅ 搭配 Experience Replay 增強穩定性  
> ✅ 應用於 `static` 模式，驗證策略是否學習成功  

---

## 🧠 DQN 架構簡述

- 使用 **MLP（Multi-Layer Perceptron）** 當作 Q-Network
- 採用 **MSE Loss + Adam Optimizer**
- 使用 **ε-greedy** 策略進行探索與利用
- 經驗重放（Experience Replay Buffer）避免樣本相關性

---

## 🧪 執行結果

共訓練 50 回合，測試 5 次皆回報為 **+1.0**，表示 agent 能成功找到正確策略。

### 訓練紀錄（節錄）：
![image](https://github.com/user-attachments/assets/94c2e5e8-132d-49a8-bb45-26b85c01f8c7)
![image](https://github.com/user-attachments/assets/885ec943-e188-4dbe-8cbc-2053331aa55b)

---


## 📂 檔案說明

| 檔案名稱 | 說明 |
|----------|------|
| `hw4_1_naive_dqn_static.py` | 主程式，包含 Q-network、ReplayBuffer、訓練與測試邏輯 |
| `requirements.txt` | 套件安裝清單（建議包含 `gym`, `numpy`, `torch`） |

---

## 🚀 執行方式

### 安裝套件
```bash
pip install gym numpy torch
```

### 執行主程式
python hw4_1_naive_dqn_static.py

