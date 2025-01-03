import streamlit as st
import matplotlib.pyplot as plt

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

if option == "情感分析":
    st.title("情感分析模組")
    st.write("此模組將分析 PTT 文章的情感傾向（正面、中立、負面）。")
    
    # 使用者輸入
    keyword = st.text_input("請輸入關鍵字：")
    period = st.number_input("搜尋期間（月）", min_value=1, max_value=12, value=3)  # 限制最大值為 12 個月
    max_articles = 100  # 固定最大抓取文章數量為 100 篇

if st.button("開始分析"):
    from modules.sentiment_analysis import analyze_sentiment
    from crawler import scrape_ptt

    st.write("正在抓取文章內容...（最多抓取 100 篇文章）")
    articles = scrape_ptt(keyword, period, max_articles)

    if not articles:
        st.write("未找到相關文章")
    else:
        st.write(f"總共抓取到 {len(articles)} 篇文章")

        # 檢查文章的 Token 長度
        st.write("檢查文章長度（前 5 篇）：")
        for i, article in enumerate(articles[:5]):
            encoded_input = tokenizer(article, truncation=True, max_length=512, return_tensors="pt")
            st.write(f"文章 {i+1} 原始字符數：{len(article)}，截斷後 Token 數：{len(encoded_input['input_ids'][0])}")

        try:
            # 分析情感
            st.write("正在分析情感...")
            sentiment_results = analyze_sentiment(articles)

            # 顯示分析結果
            st.write("分析結果：")
            for article, sentiment in zip(articles, sentiment_results):
                st.write(f"文章內容：{article[:100]}...")  # 顯示前 100 字
                st.json(sentiment)

        except ValueError as e:
            st.error(f"錯誤：{e}")


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
