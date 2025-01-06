import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import streamlit as st
from crawler import scrape_user_behavior  # 調用新增的爬蟲函數

# 設定中文字體
def set_chinese_font():
    """設定 Matplotlib 的中文字體"""
    current_dir = os.path.dirname(__file__)  # 當前模組所在目錄
    font_path = os.path.join(current_dir, "../fonts/kaiu.ttf")  # 字體相對路徑
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字體文件未找到，請確認路徑是否正確：{font_path}")
    return fm.FontProperties(fname=font_path)

def perform_clustering(data, n_clusters=3):
    """
    執行 K-means 分群。
    根據數據樣本數動態調整分群數量，避免錯誤。
    """
    features = data[['post_count', 'reply_count']]
    n_samples = features.shape[0]

    # 動態調整 n_clusters，確保 n_samples >= n_clusters
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)  # 至少分 1 群

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    return data, kmeans


def visualize_clusters(data):
    """視覺化分群結果"""
    features = data[['post_count', 'reply_count']]
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # 設定中文字體
    chinese_font = set_chinese_font()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=data['cluster'], cmap='viridis')
    
    # 設置 colorbar 並修改字體
    cbar = plt.colorbar(scatter)
    cbar.set_label('分群', fontproperties=chinese_font)

    plt.title("用戶分群結果", fontproperties=chinese_font)
    plt.xlabel("PCA 組件 1", fontproperties=chinese_font)
    plt.ylabel("PCA 組件 2", fontproperties=chinese_font)
    st.pyplot(plt)

def run_user_clustering():
    """用戶分群分析流程"""
    keyword = st.text_input("請輸入關鍵字：")
    period = st.number_input("搜尋期間（月）", min_value=1, max_value=12, value=3)

    if st.button("開始分析"):
        st.info("正在抓取 PTT 數據...")

        # 初始化進度條和進度信息
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(message):
            progress_text.text(message)

        # 爬取數據
        user_data = scrape_user_behavior(keyword, period, on_progress=update_progress)

        if not user_data:
            st.error("未能抓取到相關數據，請嘗試其他關鍵字。")
            return

        st.success(f"成功爬取 {len(user_data)} 篇文章！")

        # 轉換為 DataFrame
        df = pd.DataFrame(user_data)
        df['post_count'] = 1  # 每篇文章計為一次發文

        # 檢查數據量
        if df.shape[0] < 2:
            st.warning("數據量不足，無法進行分群分析。請嘗試增加搜尋期間或更換關鍵字。")
            return

        st.write("用戶數據預覽：", df.head())

        # 分群分析
        st.info("正在執行用戶分群...")
        clustered_data, kmeans_model = perform_clustering(df)
        st.success("分群完成！")
        st.write("分群結果：", clustered_data)

        # 可視化分群結果
        visualize_clusters(clustered_data)


        # 可視化分群結果
        visualize_clusters(clustered_data)
