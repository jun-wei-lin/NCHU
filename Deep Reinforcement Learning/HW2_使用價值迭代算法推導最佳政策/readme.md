# HW2: 使用價值迭代算法推導最佳政策

這個專案是針對強化學習環境中網格地圖 (Gridworld) 的擴充，主要展示如何利用 Value Iteration 算法計算各狀態的最優價值 V(s)，並根據最優策略顯示每個單元格上的最佳行動（以箭頭符號呈現），最後從起始單元格出發依據最優策略模擬出一條最佳路徑。

## 功能說明

- **隨機策略生成與顯示**  
  對於每個非障礙物與非終點的單元格，隨機產生一個初始行動（上下左右），並以對應的箭頭符號顯示於該單元格中。

- **策略評估**  
  使用 Value Iteration 算法來計算每個狀態的價值 V(s)：  
  - 每步行動成本為 -1。  
  - 終點狀態獎勵為 +20（terminal state）。  
  - 障礙物狀態固定值為 -1。  
  - 透過多次迭代更新，直到收斂或達到預設的最大迭代次數。

- **最終路徑模擬**  
  根據評估出的最優策略，從使用者設定的起始單元格開始模擬走出一條路徑，並高亮顯示最終路徑；如果模擬過程中遇到循環或無法前進則會中斷並提示使用者。

- **程式碼結構與可讀性**  
  程式碼採用模組化設計，並利用 IIFE 包裝前端邏輯，各功能區塊（如初始化、價值更新、策略提取與模擬）均有充分註解，方便維護與擴充。

## 資料夾結構

- HW2_使用價值迭代算法推導最佳政策/
- ├── app.py # Flask 應用主程式（負責網格生成與頁面渲染）
- ├── templates/
- │ └── index.html # 網格介面 HTML 模板 (含設定、重設與「找出最終路徑」按鈕)
- └── static/
  - ├── css/
  - │ └── styles.css # 前端樣式表 (包含格子狀態、箭頭、數值與最佳路徑高亮效果)
  - └── js/
  - │ └── script.js # 前端互動邏輯 (設定單元格、Value Iteration 算法、策略提取與模擬最佳路徑)
 
  
## 使用說明

1. **設定網格**  
   使用者先輸入網格尺寸（5~9），點擊生成網格後，可以透過滑鼠點選設定：
   - **起始單元格** (綠色)
   - **終點單元格** (紅色)
   - **障礙物** (灰色，最多 n-2 個)

2. **計算與顯示策略與價值**  
   點擊「找出最終路徑」按鈕後，系統將：
   - 利用 Value Iteration 算法計算各狀態的最優價值 V(s)。
   - 根據 V(s) 提取出最優策略，並在各單元格上以箭頭與狀態值 (V(s)) 形式顯示。
   - 根據最優策略從起始單元格模擬走出一條最佳路徑，並以高亮效果顯示。

3. **模擬最佳路徑**  
   若模擬途中遇到循環或無法前進狀況，將會中斷並提示使用者。

## 參數與評估

- **隨機策略生成：** 初始策略以隨機行動產生，用於後續策略評估的基礎。
- **折扣因子 (γ)：** 設定為 0.9，反映未來獎勵的重要性。
- **每步獎懲設定：**  
  - 每步成本 -1  
  - 終點獎勵 +20  
  - 障礙物固定值 -1

## 執行結果
![001](https://github.com/user-attachments/assets/5d581a54-7aba-4a79-bba4-4c56beb49181)
![002](https://github.com/user-attachments/assets/c8d304e7-d0b9-4975-9a4d-10f0a654f6ae)


---

這個 HW2 專案展示了如何利用策略評估與 Value Iteration 算法推導出最優的狀態價值與策略，並以直觀的方式在網格上呈現，適合用於強化學習環境的初步驗證與演示。

## License
此專案採用 MIT License 授權，歡迎自由使用與修改。
