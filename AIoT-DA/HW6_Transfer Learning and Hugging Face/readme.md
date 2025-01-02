# NLP 語言生成圖像應用

此專案展示如何使用 Hugging Face 的 Stable Diffusion 模型在 Google Colab 中實現文本生成圖像功能。用戶可以輸入自定義描述，模型將根據描述生成對應的圖像。

---

## 功能簡介
1. 接收用戶輸入的文本描述。
2. 使用 Hugging Face 的 Stable Diffusion 模型生成高品質圖像。
3. 直接在 Colab 中顯示生成的圖像。

---

## 使用說明

### 1. 環境準備
請確保您已經安裝了必要的 Python 庫。可以在 Google Colab 中執行以下指令來安裝依賴：

```bash
!pip install diffusers transformers torch matplotlib
```

### 2. 載入模型
加載 Hugging Face 的 Stable Diffusion 模型：

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")  # 使用 GPU 加速（如果可用）
```

### 3. 輸入文字描述並生成圖像
以下為完整程式碼：

```python
from PIL import Image
import matplotlib.pyplot as plt

# 函數：根據描述生成圖像
def generate_image_from_text(text_prompt):
    image = pipe(text_prompt).images[0]
    
    # 顯示生成的圖像
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")  # 隱藏座標軸
    plt.show()
    
    return image

# 手動輸入描述並生成圖像
text_input = input("請輸入您想要生成的圖像描述: ")
generated_image = generate_image_from_text(text_input)
```

### 4. 測試範例
您可以輸入以下描述進行測試：

- 中文描述：`"一間有貓當店員的桌遊店"`
- 英文描述：`"A board game store with cats as shopkeepers"`

模型將根據描述生成對應的圖像，並直接在 Colab 中顯示。

---

## 注意事項
1. 請確保在 Google Colab 中開啟 GPU 加速（`Runtime > Change runtime type > GPU`）。
2. 模型可能需要 Hugging Face 登錄憑證。如有需要，請使用以下指令進行登錄：

    ```python
    from huggingface_hub import login
    login(token="YOUR_HUGGINGFACE_API_KEY")
    ```
3. 圖像生成的質量和精確性取決於描述的詳細程度。

---

## 參考資源
- [Hugging Face 官方文檔](https://huggingface.co/docs)
- [Stable Diffusion 模型](https://huggingface.co/stabilityai/stable-diffusion-2-1)

---

## 授權
此專案基於 MIT 授權協議，歡迎自由使用和修改。
