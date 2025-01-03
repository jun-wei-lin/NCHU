import streamlit as st

# 定義主題與頁籤
st.set_page_config(page_title="智慧社群洞察與熱門話題分析平台", layout="wide")

# 頁籤選單
st.sidebar.title("功能選單")
option = st.sidebar.radio(
    "選擇功能",
    ["首頁", "情感分析", "趨勢預測", "用戶行為分析", "個性化推薦", "文本摘要"]
)

# 各頁籤對應功能
if option == "首頁":
    st.title("智慧社群洞察與熱門話題分析平台")
    st.write("歡迎使用本平台！這裡提供基於 PTT 熱門話題的智能分析與趨勢預測工具。")
elif option == "情感分析":
    st.title("情感分析模組")
    st.write("此模組將分析 PTT 文章的情感傾向（正面、中立、負面）。")
     # 使用者輸入
    keyword = st.text_input("請輸入關鍵字：")
    period = st.number_input("搜尋期間（月）", min_value=1, value=3)
    
    if st.button("開始分析"):
        from modules.sentiment_analysis import analyze_sentiment
        from crawler import scrape_ptt

        # 爬取文章
        st.write("正在抓取文章內容...")
        links = scrape_ptt(keyword, period)
        articles = [link["content"] for link in links]  # 假設抓取內容

        # 分析情感
        st.write("正在分析情感...")
        sentiment_results = analyze_sentiment(articles)

        # 展示結果
        st.write("分析結果：")
        for result in sentiment_results:
            st.json(result)
elif option == "趨勢預測":
    st.title("趨勢預測模組")
    st.write("此模組將分析熱門關鍵字的趨勢並進行未來預測。")
    # 暫時占位，未整合功能
elif option == "用戶行為分析":
    st.title("用戶行為分析模組")
    st.write("此模組將分析 PTT 用戶發文行為模式與特性。")
    # 暫時占位，未整合功能
elif option == "個性化推薦":
    st.title("個性化推薦模組")
    st.write("此模組將根據您的興趣推薦相關文章或話題。")
    # 暫時占位，未整合功能
elif option == "文本摘要":
    st.title("文本摘要模組")
    st.write("此模組將為每篇文章生成簡短摘要，幫助快速掌握重點。")



    # 暫時占位，未整合功能
