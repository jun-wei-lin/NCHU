import streamlit as st

# 引入爬蟲函數
from your_module import scrape_ptt  # 替換為爬蟲函數的正確文件路徑

st.title("熱門關鍵字分析")

# 使用者輸入
keyword = st.text_input("請輸入關鍵字：")
period = st.number_input("請輸入搜尋期間（月）", min_value=1, value=3)

if st.button("開始分析"):
    # 爬蟲執行
    st.write("正在抓取文章連結，請稍候...")
    links = scrape_ptt(keyword, period)

    if links:
        st.write(f"共找到 {len(links)} 篇相關文章")
        st.write(links)  # 列出所有文章連結
    else:
        st.write("未找到相關文章")
