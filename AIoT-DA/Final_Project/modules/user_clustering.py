import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
from crawler import fetch_user_data  # 從現有的爬蟲模組獲取用戶數據

def perform_clustering(data, n_clusters=3):
    """執行 K-means 分群"""
    features = data[['post_count', 'reply_count', 'sentiment_score_avg']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    return data, kmeans

def visualize_clusters(data):
    """視覺化分群結果"""
    features = data[['post_count', 'reply_count', 'sentiment_score_avg']]
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
    """執行用戶分群分析流程"""
    st.info("正在從 PTT 獲取用戶數據...")
    data = fetch_user_data()  # 調用爬蟲模組
    if data is None or data.empty:
        st.error("未能獲取有效的用戶數據，請檢查爬蟲程式。")
        return
    
    st.success("數據獲取成功！")
    st.write("數據預覽：", data.head())
    
    # 分群分析
    st.info("正在執行用戶分群...")
    clustered_data, kmeans_model = perform_clustering(data)
    st.success("分群完成！")
    st.write("分群結果：", clustered_data)
    
    # 可視化分群結果
    st.info("生成分群可視化...")
    visualize_clusters(clustered_data)
