# 📘 智慧折扣推薦系統 - 強化學習 DQN 專案

本專案為國立中興大學「深度強化學習」課程的期末專題。透過模擬顧客面對不同折扣的購買行為，使用 Deep Q-Network (DQN) 強化學習演算法學習推薦最適折扣策略，並以 Streamlit 建構 Web 應用程式供互動展示與分析。

---

## 🎯 專案目標

- 模擬顧客與系統的互動過程，根據顧客資訊（如年齡、地區、歷史購買紀錄）來推薦適當的折扣。
- 使用強化學習（DQN）學習折扣決策策略，使得長期收益最大化。
- 建立一個易於互動的 Streamlit Web 應用，展示模型推薦、預測結果及模擬資料視覺化。

---

## 🚀 線上展示（Streamlit Cloud）

👉 [點擊此處查看 Demo](https://dynamic-discount-strategy-project.streamlit.app/)

---

## 🧠 使用技術與方法

- 強化學習模型：Deep Q-Network (DQN)
- 資料生成：模擬環境以 Gymnasium 類似方式建構，用轉換機率模擬真實顧客行為。
- 預測評估：分類準確率（Accuracy）、精準率（Precision）、召回率（Recall）、F1-score、混淆矩陣。
- Web 應用介面：Streamlit
- 開發工具：Python 3.10, PyTorch, Pandas, scikit-learn, Matplotlib, Seaborn

---

## 📁 專案結構

```bash
Final_Project/
├── streamlit_app.py              # 主應用程式（Streamlit）
├── requirements.txt              # 套件需求檔案
├── data/
│   └── real_discount_data.csv    # 模擬生成的顧客購買資料
└── scripts/
    ├── datagen.py                # Gym 風格模擬器與資料生成腳本
    └── train.py                  # DQN 訓練與 Q-Network 建構
```

---

## 🔍 功能分頁說明（Streamlit App）

### 1️⃣ 顧客折扣推薦
使用者可透過介面輸入三個參數：
- 顧客年齡（20~60）
- 地區（北部、中部、南部、東部）
- 是否曾有購買紀錄（是/否）

系統將會：
- 將輸入轉換為 Q-Network 的狀態向量（state）
- 輸出每個折扣（0%、5%、10%、20%）對應的 Q 值
- 選擇 Q 值最高者作為推薦折扣並顯示
- 額外顯示所有 Q 值供分析參考

### 2️⃣ 模型預測評估
讀取 real_discount_data.csv 資料後進行以下分析：
- 對每筆資料，使用 Q-Network 模型預測最佳折扣（Q 值最大）
- 與實際資料中使用的折扣做比對，若一致則採用實際 converted 值，否則視為預測為未購買
- 顯示分類指標：
  - Accuracy：整體預測準確度
  - Precision / Recall / F1-score：針對「是否購買」的分類品質
  - 混淆矩陣圖（confusion matrix）：視覺化 TP、FP、FN、TN 結果

### 3️⃣ 模擬資料視覺化
展示模擬資料的統計特性：
- 折扣 vs 購買轉換率：各折扣下的平均轉換率（柱狀圖）
- 顧客年齡分布：所有樣本中顧客年齡的直方圖分布
- 可視化協助判斷折扣與顧客屬性對購買行為的影響

---

## 📈 模型與資料說明

| 欄位名稱 | 說明                         |
|----------|------------------------------|
| age      | 顧客年齡（20~60）            |
| region   | 地區（0: 北部, 1: 中部, 2: 南部, 3: 東部）|
| history  | 是否曾購買過（0 或 1）       |
| action   | 提供的折扣類別（0~3）        |
| discount | 對應折扣百分比（0%, 5%, 10%, 20%）|
| converted| 是否完成購買（0 或 1）       |
| reward   | 該次互動的利潤（扣除成本後）  |

Q-Network 架構：
- 輸入：state = [age_norm, region_norm, history]
- 輸出：4 個 Q 值（每種折扣的長期價值）
- 策略：選擇 Q 值最大的動作作為推薦折扣

---

## 📦 安裝與執行（本地端）

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動 Streamlit App
streamlit run streamlit_app.py
```

---

## ☁️ 如何部署到 Streamlit Community Cloud

1. 確保 GitHub 專案包含以下內容：
   - streamlit_app.py
   - requirements.txt
   - data/real_discount_data.csv

2. 前往：https://streamlit.io/cloud
   - 登入 GitHub 帳號
   - 建立新 App，選擇 repo 和指定主程式：Final_Project/streamlit_app.py

3. 點選 Deploy，等待啟動完成即可分享連結 🎉

---

## 📌 未來可擴充方向

- 加入使用者長期價值（LTV）作為 reward shaping
- 支援上下文多臂強盜（Contextual Bandit）對照
- 強化環境為多步互動歷史（Episode 長度 > 1）
- 將模型儲存為 .pth，部署前載入避免重新訓練

---

## 🙌 作者資訊

- 🧑‍💻 第五組：林君瑋、王均維、羅振豪
- 📘 專案：深度強化學習期末報告（2025）
- ✉️ 聯絡方式：請透過學校課程平台或 GitHub 聯繫

---

感謝指導老師與助教協助完成本專題！若您對此專案有建議或想法，歡迎開啟 issue 或 star ⭐ 支持 🙌
