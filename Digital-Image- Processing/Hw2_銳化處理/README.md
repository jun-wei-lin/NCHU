
# Unsharp Masking & Image Sharpening

此專案主要實現了影像的銳化處理，使用了常見的 **Unsharp Masking** 技術。透過一系列的影像處理步驟，我們可以提升影像的邊緣細節，使影像看起來更清晰。以下是每個步驟的具體操作及其實現方法。

## 1. 安裝必要的工具

在執行專案之前，您需要安裝以下 Python 套件：

```bash
pip install opencv-python-headless matplotlib
```

## 2. 流程簡介
<p align="center">
  <img src="https://raw.githubusercontent.com/jun-wei-lin/NCHU/refs/heads/main/Digital-Image-%20Processing/Hw2_%E9%8A%B3%E5%8C%96%E8%99%95%E7%90%86/Hw2_%E6%B5%81%E7%A8%8B%E5%9C%96drawio.png" alt="Example Image">
</p>

### 步驟 1: 讀取影像
首先，載入影像並轉換為灰階影像，這有助於簡化後續處理，避免色彩信息影響銳化結果。

```python
gray_img, source_image = read_image_from_url(source_url)
```

### 步驟 2: 進行 Sobel Operator
Sobel 濾波器用於計算影像的邊緣。這裡，我們分別對影像進行水平方向和垂直方向的梯度運算，並將兩者合併以獲得梯度幅值圖像。

```python
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度
diff_1_img = cv2.magnitude(sobel_x, sobel_y)  # 合併梯度
```

### 步驟 3: 進行 Laplace Operator
使用 Laplace 濾波器計算二階微分，這可以揭示影像的細節部分，通常有助於更精確地捕捉邊緣。

```python
diff_2_img = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
```

### 步驟 4: 算術平均濾波 (Mean Filter)
對影像進行算術平均濾波，以減少雜訊並平滑影像。這是用來構建 mask 的步驟，後面會與 Laplacian 濾波結果結合。

```python
kernel = np.ones((3, 3), np.float32) / 9
mean_img = cv2.filter2D(diff_1_img, -1, kernel)
```

### 步驟 5: 正規化處理
將結果進行正規化，使其落在 [0, 1] 範圍內，避免在後續運算中出現數值超過範圍的問題。

```python
mean_img = cv2.normalize(mean_img, None, 0, 1, cv2.NORM_MINMAX)
```

### 步驟 6: 結合 Laplace 與 Mean Filter 的結果
將 Laplace 濾波結果與算術平均濾波結果進行乘法運算，這樣可以獲得我們需要的增強細節圖像。

```python
e = diff_2_img * (mean_img / 255)
e_scaled = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX)
```

### 步驟 7: 銳化處理
將結果與原始影像結合，並進行加權操作，實現銳化增強。這一步是 Unsharp Masking 技術的核心，將增強的邊緣細節應用到原始影像中。

```python
e_scaled_color = cv2.merge([e_scaled] * 3)  # 將灰階影像轉換為 3 通道影像
enhanced_image = cv2.addWeighted(source_image.astype(np.float32), 1.0, 
                                 e_scaled_color.astype(np.float32), 0.3, 0)
```

### 步驟 8: 顯示影像結果
最後，將所有中間結果與最終增強的影像顯示出來，以便比較不同步驟的效果。

```python
show_images(
    [gray_img, diff_1_img, diff_2_img, mean_img, enhanced_image],
    ['Gray Image', 'Sobel Operator', 'Laplace Operator', 'Mean Filter', 'Unsharp Masking Result']
)
```

## 3. 結果展示
最後將原始影像與增強影像進行並排顯示，您可以直觀地看到銳化效果。

```python
total_kp = np.concatenate((source_image, enhanced_image), axis=1)
show_images([total_kp], ['V.S.'])
```

## 4. 額外的銳化方法

除了傳統的 **Unsharp Masking** 外，還有一些其他的影像銳化技術，例如：

- **Laplacian of Gaussian (LoG)**: 通過結合高斯模糊和拉普拉斯運算來加強影像邊緣。
- **High-pass Filtering**: 利用高通濾波來增強邊緣細節。
- **Bilateral Filtering**: 在保持邊緣細節的同時去除雜訊。
- **Wiener Filter**: 一種自適應濾波方法，可以去除噪音的同時保持影像細節。

您可以根據需求選擇合適的銳化方法，並在專案中進行實驗。

## 5. 參考資料
- [【影像處理】非銳化濾鏡 Unsharp Masking](https://jason-chen-1992.weebly.com/home/-unsharp-masking)
- [Nick.NCHU.DIP Public 非銳化濾鏡 Unsharp Masking](https://github.com/nicktien007/Nick.NCHU.DIP?tab=readme-ov-file)
