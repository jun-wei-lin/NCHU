import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import streamlit as st
from crawler import scrape_user_behavior
from sklearn.preprocessing import StandardScaler

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
    # Step 1: 重新計算用戶的總發文數和回文數
    user_stats = data.groupby('author').agg(
        total_post_count=('post_count', 'count'),
        total_reply_count=('reply_count', 'sum')
    ).reset_index()

    # Step 2: 標準化數據
    scaler = StandardScaler()
    user_stats[['scaled_post_count', 'scaled_reply_count']] = scaler.fit_transform(
        user_stats[['total_post_count', 'total_reply_count']]
    )
    # 檢查樣本數並調整分群數
    n_samples = user_stats.shape[0]
    if n_samples < 2:
        st.warning("樣本數過少，無法進行分群分析。請嘗試延長搜尋日期或更改關鍵字。")
        return pd.DataFrame(), None, None

    if n_samples < max_clusters:
        max_clusters = n_samples
        st.warning(f"樣本數（{n_samples}）少於設定的最大分群數，已自動調整分群數為 {max_clusters}。")


    # Step 3: 肘部法則確定最佳分群數
    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(user_stats[['scaled_post_count', 'scaled_reply_count']])
        inertia.append(kmeans.inertia_)

    # 可視化肘部法則
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title("肘部法則：分群數 vs. 惰性", fontproperties=set_chinese_font())
    plt.xlabel("分群數 (k)", fontproperties=set_chinese_font())
    plt.ylabel("惰性 (Inertia)", fontproperties=set_chinese_font())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Step 4: K-means 分群
    n_clusters = 3  # 可根據肘部法則調整
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_stats['cluster'] = kmeans.fit_predict(user_stats[['scaled_post_count', 'scaled_reply_count']])

    # Step 5: 主成分分析 (PCA) 降維
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(user_stats[['scaled_post_count', 'scaled_reply_count']])
    user_stats['pca_1'] = reduced_features[:, 0]
    user_stats['pca_2'] = reduced_features[:, 1]

    return user_stats, kmeans, pca

def visualize_clusters_with_summary(data, cluster_summary, kmeans_model):
    """
    增強的用戶分群結果可視化，包括數據摘要和特徵分佈圖。

    Args:
        data (DataFrame): 包含分群和 PCA 特徵的數據。
        cluster_summary (DataFrame): 每個群的摘要數據（平均發文數、回文數、用戶數量）。
        kmeans_model (KMeans): KMeans 模型。
    """
    import matplotlib.pyplot as plt

    # 數據摘要表格
    st.subheader("分群數據摘要")
    st.write(cluster_summary)

    # 群內說明文字
    st.subheader("分群數據的應用價值")
    st.markdown("""
    - **群 0**：以普通用戶為主，活躍度較低，可能是大部分用戶的行為模式。
    - **群 1**：回文數明顯較高，這些用戶可能是社群的高互動參與者，對回應討論感興趣。
    - **群 2**：活躍度最高，發文和回文數都非常高，可能是熱門話題的主要貢獻者或核心用戶。
    """)
    
    # 可視化群內特徵分佈（柱狀圖）
    st.subheader("各群平均特徵比較")
    cluster_summary[['avg_post_count', 'avg_reply_count']].plot(
        kind='bar', figsize=(8, 6), legend=True, alpha=0.8
    )
    plt.title("各群的平均發文數與回文數", fontproperties=set_chinese_font())
    plt.xlabel("群集編號 (Cluster)", fontproperties=set_chinese_font())
    plt.ylabel("數值 (平均數)", fontproperties=set_chinese_font())
    st.pyplot(plt)

    # 箱線圖：回文數分佈
    st.subheader("回文數分佈（箱線圖）")
    st.markdown("""
    ### 箱線圖的意義：
    - 箱線圖展示了各群回文數的分佈情況。
    - 其中：
      - 箱體代表數據的 **四分位範圍 (IQR)**。
      - 箱體內的線代表數據的 **中位數**。
      - 箱體外的點代表數據中的 **異常值 (outliers)**。
    ### 觀看方式：
    - **群集編號 (X 軸)** 表示不同用戶群。
    - **回文數 (Y 軸)** 表示該群用戶的回文行為數據。
    - 可以觀察：
      - **群間的回文數分佈差異**。
      - 是否存在某些群有極高或極低的異常值。
    """)
    plt.figure(figsize=(8, 6))
    data.boxplot(column='total_reply_count', by='cluster', grid=False)
    plt.title("回文數分佈（按群）", fontproperties=set_chinese_font())
    plt.suptitle("")  # 移除默認標題
    plt.xlabel("群集編號 (Cluster)", fontproperties=set_chinese_font())
    plt.ylabel("回文數", fontproperties=set_chinese_font())
    st.pyplot(plt)

    # 群內特徵分佈圖（PCA）
    st.subheader("用戶分群結果可視化（PCA 降維）")
    st.markdown("""
    ### PCA 降維圖的意義：
    - PCA 降維圖通過將高維數據壓縮到 2 維，展示各群用戶的分佈情況。
    - 每個點表示一位用戶，點的顏色對應該用戶所屬的群。
    - X 軸和 Y 軸分別表示數據在第一主成分 (PCA 組件 1) 和第二主成分 (PCA 組件 2) 上的投影。
    ### 觀看方式：
    - **群集 (顏色區分)**：顏色對應用戶的分群結果。
    - **點的位置 (分佈)**：表示用戶行為的特徵相似性，距離越近的點表明用戶行為越相似。
    - 可觀察：
      - **各群是否有明顯分界**（不同顏色的點是否聚集）。
      - 某些群是否存在異常行為（例如遠離其他點的用戶）。
    """)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data['pca_1'], data['pca_2'], c=data['cluster'], cmap='viridis', alpha=0.7
    )
    # 設置 colorbar 並調整字體
    cbar = plt.colorbar(scatter)
    cbar.set_label('分群標籤 (群集編號)', fontproperties=set_chinese_font())
    
    plt.title("用戶行為分群結果（PCA 降維）", fontproperties=set_chinese_font())
    plt.xlabel("PCA 組件 1", fontproperties=set_chinese_font())
    plt.ylabel("PCA 組件 2", fontproperties=set_chinese_font())
    st.pyplot(plt)

def preprocess_data(data):
    """
    預處理數據，處理極端值，並返回處理後的數據。
    Args:
        data (DataFrame): 包含用戶行為特徵的原始數據。
    Returns:
        DataFrame: 預處理後的數據。
    """
    # 設置極端值篩選的上限（均值 + 3 * 標準差）
    mean_reply = data['reply_count'].mean()
    std_reply = data['reply_count'].std()
    threshold_upper = mean_reply + 3 * std_reply

    # 統計極端值前的數據分佈
    st.write("處理前的回文數摘要：")
    st.write(data['reply_count'].describe())

    # 處理極端值
    data['reply_count'] = data['reply_count'].clip(upper=threshold_upper)

    # 統計處理後的數據分佈
    st.write(f"已將回文數的極端值截斷至上限 {threshold_upper:.2f}。")
    st.write("處理後的回文數摘要：")
    st.write(data['reply_count'].describe())

    return data

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

        # 數據預處理
        st.info("正在進行數據預處理（包括極端值處理）...")
        df = pd.DataFrame(user_data)
        df['post_count'] = 1  # 每篇文章計為一次發文

        # **新增：檢查樣本數量**
        if len(df) < 2:
            st.warning("數據樣本數過少，無法進行有效分群分析。請嘗試延長搜尋期間或更改關鍵字。")
            return
            
        df = preprocess_data(df)  # 處理極端值

        # 分群分析
        st.info("數據預處理完成，開始用戶分群分析...")
        clustered_data, kmeans_model, _ = perform_clustering(df)

        # 如果分群結果為空，提示用戶並結束
        if clustered_data.empty:
            st.warning("分群結果不可用，請嘗試增加搜尋期間或更改關鍵字。")
            return
            
        # 分群統計
        cluster_summary = clustered_data.groupby('cluster').agg({
            'total_post_count': 'mean',
            'total_reply_count': 'mean',
            'author': 'count'
        }).rename(columns={'author': 'user_count'}).reset_index()
        cluster_summary.rename(
            columns={
                'total_post_count': 'avg_post_count',
                'total_reply_count': 'avg_reply_count'
            },
            inplace=True
        )

        # 視覺化增強
        visualize_clusters_with_summary(clustered_data, cluster_summary, kmeans_model)

