# Hw5 系列作業說明

本專案包含三個作業項目，分別為 **Hw5-1**、**Hw5-2** 和 **Hw5-3**。以下為各作業的詳細內容與執行說明。

---

## 📂 Hw5-1: iris classification problem
### 使用 tf.keras, pytorch, pytorch lightning 實現了經典的 Iris 資料集分類問題，並採用了 Dropout、Batch Normalization、Early Stopping 和學習率調整等技術，進一步提升模型效能。


### 使用技術
目標是進行基礎影像處理操作，包含：
- PyTorch 深度學習框架
- Dropout、Batch Normalization
- 學習率調整 (StepLR)

### 功能
- 支援資料集的標準化與拆分
- 支援模型訓練、驗證與測試
- 繪製準確率變化圖表

### 資料集
- 使用 Scikit-learn 提供的 Iris 資料集，該資料集包含 150 筆資料，分為 3 種分類，每筆資料包含 4 個特徵。

### 程式邏輯
### 資料處理
- 使用 StandardScaler 將資料標準化。
- 資料集拆分為訓練集 (70%)、驗證集 (15%) 與測試集 (15%)。
### 模型架構
模型包含以下部分：
1.  兩層隱藏層，每層包含：
    - 線性層 (Linear)
    - 批量正規化 (BatchNorm1d)
    - 激活函數 (ReLU)
    - Dropout
2.  最終輸出層：
    - 線性層輸出 3 個分類
### 訓練與驗證
  - 損失函數：交叉熵損失 (CrossEntropyLoss)
  - 優化器：Adam
  - 學習率調整：每 30 個 epoch 衰減至原來的 50%
### 測試
獨立測試數據集，計算並輸出測試準確率。

### 準確率變化圖表
訓練完成後，自動繪製訓練與驗證準確率變化圖表。

---

## 📂 Hw5-2: 手寫辨認
### 功能描述
Hw5-2 專注於影像分析技術，功能包含：
- 邊使用 MNIST 數據集進行手寫數字辨識。
- 支援兩種模型架構：
  - Dense Neural Network (DNN)： 全連接神經網路。
  - Convolutional Neural Network (CNN)： 卷積神經網路，適合影像處理。
- 自動列出錯誤分類的樣本，顯示其正確標籤與預測標籤。
- 可視化錯誤分類的手寫數字。

### 執行方式

- 在終端機列出錯誤分類樣本的索引值、正確標籤與預測標籤。

- 彈出視覺化圖表顯示部分錯誤分類的手寫數字及其預測結果。

### 修改參數

- 可透過調整程式內的 num_examples 變數設定視覺化錯誤分類樣本的數量。

---

## 📂 Hw5-3: Cifar 圖片分類 vgg19  pretrained

### 功能描述
1. 使用 Hugging Face 提供的模型與工具實現 Grad-CAM 的可視化。
2. 解決 RuntimeError 相關錯誤，並調整代碼以適配不同影像尺寸（例如 800x600）。
3. 使用 VGG19 作為預訓練模型進行影像分類，針對 Cifar 數據集（10 類別）進行模型訓練與預測。

### 執行方式
1. 確保安裝必要套件。

2. 執行程式以載入影像並生成 Grad-CAM 視覺化。
  
3. 程式將下載影像並生成 Grad-CAM 熱力圖。

4. 訓練完成後，程式會輸出模型性能評估結果。

---

## ⚙️ 環境需求
- colab


## 📮 聯絡方式
若有任何問題，歡迎透過以下方式聯繫：
- Email: junweilin975@gmail.com
- GitHub Issues

歡迎提交 Pull Request 或意見反饋！
