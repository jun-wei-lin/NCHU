# HW4-3: DQN with Keras for Random Mode 🧠🎮

本作業實作強化學習中的 DQN（Deep Q-Network）演算法，並使用 **Keras** 於 `random mode` 環境中進行訓練與測試。
同時，整合兩項訓練穩定技巧：

- ✅ **Gradient Clipping**：防止梯度爆炸
- ✅ **Learning Rate Scheduler**：幫助模型穩定收斂

---

## 📌 作業目標

- 使用 TensorFlow/Keras 建構 Q-network
- 應用於起點、障礙隨機的 random mode 環境
- 藉由 Replay Buffer 儲存經驗並提升訓練穩定性
- 評估訓練結果與泛化能力

---

## ⚙️ 執行方式

### Google Colab 執行建議：

1. 開啟 Colab：[https://colab.research.google.com](https://colab.research.google.com)
2. 上傳 `HW4_3_DQN_Keras_RandomMode.ipynb`
3. 建議切換執行環境為 GPU：`Runtime` → `Change runtime type` → 選擇 GPU

或於本機端安裝執行：

```bash
pip install tensorflow numpy
python dqn_keras_random.py
```

---

## 📂 檔案說明

| 檔案名稱                         | 說明                          |
|----------------------------------|-------------------------------|
| `dqn_keras_random.py`            | 主程式，完整訓練與測試流程    |
| `HW4_3_DQN_Keras_RandomMode.ipynb` | Colab Notebook 版本，適合分段執行與上傳 |
| `requirements.txt`               | 所需套件列表                  |

---

## 🧪 實驗結果

### 訓練 100 回合：
- 初期 reward 接近 0
- 後期 reward 可達 10~28，模型成功學會策略

### 測試 5 回合結果：
```
Test Episode 1: Total Reward = 1.0
Test Episode 2: Total Reward = 2.0
Test Episode 3: Total Reward = 12.0
Test Episode 4: Total Reward = 4.0
Test Episode 5: Total Reward = 9.0
```

> 顯示 agent 即使在隨機起點下，仍具備穩定決策能力。

---

## 💡 訓練技巧說明

- **Gradient Clipping**：使用 `clipnorm=1.0` 限制每次反向傳播梯度範圍
- **Learning Rate Scheduler**：使用 `ExponentialDecay` 讓學習率隨訓練次數遞減

---

## 📝 心得與反思

此作業讓我熟悉如何在 TensorFlow/Keras 中構建強化學習系統。與傳統 DQN 相比，額外整合訓練穩定技巧後，能有效提升在隨機環境下的學習效率與表現穩定性。  
此外透過 Google Colab 執行訓練任務，簡化了環境安裝與加速運算，是實驗與開發的理想選擇。

