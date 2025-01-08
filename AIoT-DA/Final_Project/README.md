# 智慧社群洞察與熱門話題分析平台

## 專案簡介
本專案是一個基於 PTT 的智慧社群洞察工具，透過即時爬蟲、情感分析、趨勢預測與用戶行為分群，幫助使用者快速掌握社群輿情與熱門話題走向。

專案採用 **Streamlit Cloud** 部署，並以互動式網頁形式提供服務，讓使用者輸入關鍵字即可查看分析結果。

---

## 功能描述
1. **即時爬蟲與清洗**
   - 爬取 PTT 八卦板關鍵字相關文章，進行基礎文本清洗（移除標點、合併空格、限制長度等）。
2. **情感分析**
   - 使用 Hugging Face 模型分析文章與回文的正負向情感，輸出結果可視化。
3. **趨勢預測**
   - 利用 ARIMA 時間序列模型分析歷史文章數據，預測未來幾期話題熱度走勢。
4. **用戶行為分群**
   - 基於 K-means 演算法，分析用戶發文與回文行為，分群並視覺化呈現用戶互動模式。
5. **前端可視化**
   - 以長條圖、折線圖等形式直觀呈現分析結果，支援互動篩選。

---

## 技術架構
- **後端技術**
  - 爬蟲：`requests`、`BeautifulSoup`
  - NLP 分析：Hugging Face 的 `transformers`
  - 時間序列分析：`statsmodels` (ARIMA)
  - 分群分析：`sklearn` (K-means, PCA)
- **前端技術**
  - 可視化框架：`Streamlit`
  - 部署：`Streamlit Cloud`
- **版本控管**
  - **GitHub**：程式碼管理與協作

---

## 專案架構
```plaintext
AIoT-DA/
├── Final_Project/
│   ├── streamlit_app.py               # Streamlit 前端應用
│   ├── modules/
│   │   ├── sentiment_analysis.py      # 情感分析模組
│   │   ├── user_clustering.py         # 用戶分群模組
│   ├── trend_prediction.py            # 趨勢預測模組
│   ├── fonts/
│   │   ├── kaiu.ttf                   # 字型檔案
│   ├── requirements.txt               # 依賴函式庫清單
│   ├── crawler.py                     # 爬蟲程式
```
## Demo
👉 [點擊此處查看 Demo](https://eefzbzjg62yh54cyzxez5q.streamlit.app/) 

## Video
🎥 [點擊此處觀看示範影片](https://youtu.be/4Oh9ubYgO8k)

## 範例截圖
### 1. 情感分析
![情感分析001](https://github.com/user-attachments/assets/434f8412-264d-490b-a6da-b12a0182b6dc)
![情感分析002](https://github.com/user-attachments/assets/9681f775-47ad-4c68-8cbc-d9d1175c69cc)
---
### 2. 趨勢預測
![驅勢預測001](https://github.com/user-attachments/assets/ad97f823-9553-4589-a2de-d8342d27d243)
![驅勢預測002](https://github.com/user-attachments/assets/fc4ecc5e-f8a4-4829-8204-23b43f29139f)
---
### 3. 用戶行為分析
![用戶行為分析001](https://github.com/user-attachments/assets/d074a657-a0db-4168-a589-a36babb71fa6)
![用戶行為分析002](https://github.com/user-attachments/assets/f2c8c657-dcb6-4e58-83bf-8c56c5ec8dab)
![用戶行為分析003](https://github.com/user-attachments/assets/dd9c77d3-195b-4773-86df-c6927891d444)
![用戶行為分析004](https://github.com/user-attachments/assets/fccbf0a1-89ff-4eba-92c4-8e1b29aa1107)
![用戶行為分析005](https://github.com/user-attachments/assets/c7dcd9e3-c804-497f-9d00-cb4ac7ea7210)
## 未來方向
1. 多平台支持：納入 Dcard、Twitter 等社群平台數據源。
2. 進階 NLP 處理：
-- 支持繁體中文斷詞與客製化詞典，提升分析準確性。
-- 探索多語言支持，適用更廣泛的使用場景。
3. 輿情報表自動化：支持報表生成與即時警示功能。
4. 即時事件監測：搭建即時監測系統，捕捉突發熱點並生成預警通知。

---
## 感謝
感謝所有團隊成員的努力與貢獻，並感謝指導老師提供的寶貴建議！

