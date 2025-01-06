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


def perform_clustering(data, max_clusters=5):
    """
    執行用戶分群並返回分群結果。

    Args:
        data (DataFrame): 包含用戶行為特徵的數據。
        max_clusters (int): 最大分群數量，用於肘部法則。

    Returns:
        DataFrame: 包含分群結果的數據。
        KMeans: 訓練完成的 KMeans 模型。
        PCA: 主成分分析模型。
    """
    # 標準化數據
    features = data[['post_count', 'reply_count']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 肘部法則確定最佳分群數
    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # 可視化肘部法則
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title("肘部法則：分群數 vs. 惰性")
    plt.xlabel("分群數 (k)")
    plt.ylabel("惰性 (Inertia)")
    plt.show()

    # 選擇分群數（此處暫設為 3，建議根據肘部圖調整）
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)

    # 主成分分析 (PCA) 降維
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    data['pca_1'] = reduced_features[:, 0]
    data['pca_2'] = reduced_features[:, 1]

    return data, kmeans, pca


def visualize_clusters(data, kmeans_model):
    """
    可視化用戶分群結果。

    Args:
        data (DataFrame): 包含分群和 PCA 特徵的數據。
        kmeans_model (KMeans): KMeans 模型。
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data['pca_1'], data['pca_2'], c=data['cluster'], cmap='viridis', alpha=0.7
    )
    plt.colorbar(scatter, label='分群')
    plt.title("用戶分群結果")
    plt.xlabel("PCA 組件 1")
    plt.ylabel("PCA 組件 2")
    plt.show()

    # 顯示每個群的中心點
    cluster_centers = kmeans_model.cluster_centers_
    print("每個群的中心點（標準化後）:")
    print(cluster_centers)

def run_user_clustering():
    """用戶分群分析流程"""
    keyword = st.text_input("請輸入關鍵字：")
    period = st.number_input("搜尋期間（月）", min_value=1, max_value=12, value=3)

    if st.button("開始分析"):
        # 初始化進度條和進度信息
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # 停止信號
        if "stop_signal" not in st.session_state:
            st.session_state.stop_signal = False

        stop_button = st.button("停止爬取")
        if stop_button:
            st.session_state.stop_signal = True

        # 回調函數：更新進度
        def update_progress(message):
            progress_text.text(message)

        # 回調函數：檢查停止信號
        def check_stop_signal():
            return st.session_state.stop_signal

        # 開始爬取
        st.info("正在抓取 PTT 數據...")
        user_data = scrape_user_behavior(keyword, period, on_progress=update_progress, stop_signal=check_stop_signal)

        # 清理停止信號
        st.session_state.stop_signal = False

        if not user_data:
            st.error("未能獲取足夠的數據。請嘗試其他關鍵字或時間範圍。")
            return

        st.success(f"已完成爬取 {len(user_data)} 篇文章，正在進行分析...")

        # 數據轉換
        df = pd.DataFrame(user_data)
        df['post_count'] = 1  # 每篇文章計為一次發文

        # 分群分析
        st.info("正在執行用戶分群...")
        clustered_data, kmeans_model, pca_model = perform_clustering(df)
        st.success("分群完成！")

        # 分群統計
        cluster_summary = clustered_data.groupby('cluster').agg({
            'reply_count': ['mean', 'sum', 'count']
        })
        st.write("分群統計：", cluster_summary)

        # 視覺化
        visualize_clusters(clustered_data, kmeans_model)
