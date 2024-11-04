## Hw3: SVM作業說明

HW3-1. 1D compare to  logistic regression with SVM on simple case

HW3-2. 2D SVM with streamlit deployment (3D plot) -dataset 分布在feature plane上圓形

HW3-3. 2D dataset 分布在feature plane上非圓形

## HW3_1: 使用 SVM 與一維邏輯迴歸進行比較
### 目標
生成隨機數據並根據條件進行二元分類，然後可視化結果。

### 步驟
1. 數據生成：生成 300 個隨機變數 X(i)，範圍為 [0, 1000]。
2. 目標變量生成：根據條件定義 𝑌(𝑖)Y(i)：當 500<𝑋(𝑖)<800500<X(i)<800 時 𝑌(𝑖)=1Y(i)=1，否則 𝑌(𝑖)=0Y(i)=0。
3. 視覺化：使用 Matplotlib 繪製散點圖來展示X與Y的分佈

### 圖片示例
![001](https://github.com/user-attachments/assets/ed8ca1b1-5b1b-4c9c-ab35-31e8bf12e68e)
![002](https://github.com/user-attachments/assets/3d7cafdd-03f4-44cc-9316-d861ace7d1b5)

## HW3_2: 使用 SVM 觀察二維邏輯迴歸，feature plane 為圓形
### 目標
建立邏輯回歸模型來預測二元分類結果，並評估模型表現。

### 步驟
1. 數據準備：從 HW3_1 生成的數據中標準化 𝑋 變量。
2. 模型訓練：使用 scikit-learn 的 LogisticRegression 訓練邏輯回歸模型。
3. 模型預測：在訓練數據上進行預測並可視化預測結果。
4. 評估模型：計算並輸出模型的準確率和分類報告。

### 圖片示例
![003](https://github.com/user-attachments/assets/8d7ced8e-53fc-4087-bb20-8313ac4a095c)
![004](https://github.com/user-attachments/assets/a484553d-fba2-4e56-ab87-7b078e5e3acc)

## HW3_3: 使用 SVM 觀察二維邏輯迴歸，feature plane 為非圓形
### 目標
使用 SVM 模型進行非線性分類，以解決邏輯回歸無法處理的非線性問題。

### 步驟
1. 數據準備：使用 HW3_1 生成的數據，並對 𝑋 變量進行標準化。
2. 模型訓練：使用 RBF 核心的 SVM 模型進行訓練。
3. 模型預測：在訓練數據上進行預測並可視化結果，展示非線性決策邊界。
4. 評估模型：計算並輸出模型的準確率和分類報告。

### 圖片示例
![005](https://github.com/user-attachments/assets/80f21fb2-fa13-4ea0-8a4e-f5db37b94e06)
![006](https://github.com/user-attachments/assets/9ff107e0-c0c4-4f70-af1e-ca7207cbd691)

