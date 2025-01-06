import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
from crawler import scrape_user_behavior  # 調用新增的爬蟲函數

def perform_clustering(data, n_clusters=3):
    """執行 K-means 分群"""
    features = data[['post_count', 'reply_count']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    return data, kmeans

def visualize_clusters(data):
    """視覺化分群結果"""
    features = data[['post_count', 'reply_count']]
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=data['cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.title("用戶分群結果")
    plt.xlabel("PCA 組件 1")
    plt.ylabel("PCA 組件 2")
    st.pyplot(plt)

def run_user_clustering():
    """用戶分群分析流程"""
    keyword = st.text_input("請輸入關鍵字：")
    period = st.number_input("搜尋期間（月）", min_value=1, max_value=12, value=3)
    max_articles = 100

    if st.button("開始分析"):
        st.info("正在抓取 PTT 數據...")
        user_data = scrape_user_behavior(keyword, period, max_articles)

        if not user_data:
            st.error("未能抓取到相關數據，請嘗試其他關鍵字。")
            return

        st.success(f"成功抓取到 {len(user_data)} 篇文章！")
        
        # 轉換為 DataFrame
        df = pd.DataFrame(user_data)
        df['post_count'] = 1  # 每篇文章計為一次發文
        st.write("用戶數據預覽：", df.head())

        # 分群分析
        st.info("正在執行用戶分群...")
        clustered_data, kmeans_model = perform_clustering(df)
        st.success("分群完成！")
        st.write("分群結果：", clustered_data)

        # 可視化分群結果
        visualize_clusters(clustered_data)
