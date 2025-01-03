import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

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
        from transformers import AutoTokenizer  # 新增 tokenizer

        st.write("正在抓取文章內容...（最多抓取 100 篇文章）")
        articles, links = scrape_ptt(keyword, period, max_articles)  # 確保返回連結
        if not articles:
            st.write("未找到相關文章")
        else:
            st.write(f"總共抓取到 {len(articles)} 篇文章")

            # 初始化 tokenizer
            tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

            try:
                # 分析情感
                st.write("正在分析情感...")
                sentiment_results = analyze_sentiment(articles)

                # 統計情感分佈
                st.write("情感分佈統計：")
                labels = [result["label"] for result in sentiment_results]
                label_counts = {label: labels.count(label) for label in set(labels)}

                for label, count in label_counts.items():
                    st.write(f"{label}: {count} 篇")
                    
                # 獲取字體文件的相對路徑
                current_dir = os.path.dirname(__file__)
                font_path = os.path.join(current_dir, "fonts", "kaiu.ttf")  # 字體文件相對路徑

                # 確認字體文件是否存在
                if not os.path.exists(font_path):
                    raise FileNotFoundError(f"字體文件未找到，請確認路徑是否正確：{font_path}")

                # 加載字體
                my_font = fm.FontProperties(fname=font_path)
        
               # 繪製柱狀圖
                fig, ax = plt.subplots()
                ax.bar(label_counts.keys(), label_counts.values(), color=['green', 'red', 'blue'])
                ax.set_title("情感分佈", fontproperties=my_font)
                ax.set_xlabel("情感類別", fontproperties=my_font)
                ax.set_ylabel("文章數量", fontproperties=my_font)
                st.pyplot(fig)

                # 展示各種情感標籤的文章三篇
                st.write("展示各情感類別的文章：")
                for label in label_counts.keys():
                    st.write(f"**{label} 類文章**（展示三篇）：")
                    count = 0
                    for i, result in enumerate(sentiment_results):
                        if result["label"] == label and count < 3:
                            st.write(f"文章 {count+1}：{articles[i][:100]}...")  # 顯示前 100 字
                            st.json(result)  # 顯示情感分析結果
                            st.markdown(f"<span style='color:blue; font-weight:bold;'>[查看原文]({links[i]})</span>", unsafe_allow_html=True)  # 提供原文連結，使用樣式
                            count += 1
                    if count == 0:
                        st.write("無符合條件的文章")

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
