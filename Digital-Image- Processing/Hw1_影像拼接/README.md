# 作業：影像拼接

## 專案說明
此專案使用 Python 編寫，透過 OpenCV 實現影像拼接的功能。我們採用了 SIFT 特徵檢測與 RANSAC 方法進行兩張或多張影像的拼接處理。

---

## 使用環境
- **語言**：Python
- **執行環境**：Google Colab
- **主要函式庫**：OpenCV

---

## 流程說明
![流程](https://github.com/jun-wei-lin/NCHU/blob/main/Digital-Image-%20Processing/Hw1_%E5%BD%B1%E5%83%8F%E6%8B%BC%E6%8E%A5/Hw1_%E5%BD%B1%E5%83%8F%E6%8B%BC%E6%8E%A5_%E6%B5%81%E7%A8%8B.png?raw=true)
### 1. 載入影像
- 因為影像存儲於 GitHub 中，我們先對 URL 的影像進行下載與預處理。
- 轉換為灰階圖像（供 SIFT 使用）及彩色圖像（供可視化處理）。
- 原始圖片為 3024x4032 像素，為降低處理時間，將其縮放為 768x1024。

### 2. 檢測特徵點 (SIFT)
- 使用 **SIFT (Scale-Invariant Feature Transform)**，進行影像中的局部特徵檢測。
- 特徵點檢測過程需要灰階圖像，處理流程將基於灰階影像進行。

### 3. 描述符匹配 (BFMatcher)
- 利用 **暴力匹配法 (Brute-Force Matcher)**，比較兩組特徵描述子的歐幾里得距離，生成匹配對。

### 4. 過濾匹配點 (Ratio Test)
- 應用 **最近鄰和次近鄰的距離比值測試**，篩選出準確的匹配點，排除錯誤匹配。

### 5. 隨機採樣一致性 (RANSAC)
- 使用 **RANSAC (Random Sample Consensus)** 演算法：
  - 隨機選取匹配點，計算內點集合。
  - 排除誤差較大的匹配點。

### 6. 計算單應性矩陣 (Homography)
- 根據內點集合，計算單應性矩陣 \( H \)，描述影像之間的幾何關係。

### 7. 透視變換與平移
- 利用單應性矩陣 \( H \)，對左影像進行透視變換。
- 對右影像進行平移處理，確保兩張影像對齊。

### 8. 融合圖像
- 根據像素值對兩張影像進行融合，避免黑色邊緣出現。

### 9. 輸出拼接結果
- 將最終拼接的影像保存或顯示。

---

## 結果展示
![擷取](https://github.com/user-attachments/assets/d77ecf85-ad44-40d4-82c9-fcdc32584a51)


---

## 如何運行程式
1. 確保已安裝以下依賴項目：
   - `opencv-python`
   - `numpy`

2. 在 Google Colab 中運行程式碼。

3. 載入影像 URL，並執行拼接程式。

---

## 聯絡方式
如有問題，請提交 Issue 或聯絡作者。
