import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 設定 Streamlit 介面
st.title("2D SVM 與 3D 視覺化展示")
st.write("使用拉桿調整決策函數的距離閾值，以觀察不同閾值對決策邊界的影響。")

# 載入一組非圓形的樣本數據集
X, y = datasets.make_moons(n_samples=200, noise=0.1, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用 RBF 核心訓練 SVM 模型
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X, y)

# 添加拉桿來調整距離閾值
distance_threshold = st.slider("調整距離閾值", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)



# 創建網格以供繪圖
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# 計算網格中每個點的決策函數值，並調整為新閾值
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) - distance_threshold

# 3D 繪圖
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap="coolwarm", edgecolor="k", alpha=0.7)

# 繪製數據點
ax.scatter(X[:, 0][y == 0], X[:, 1][y == 0], -1, c='blue', marker='o', s=50, label="Class 0")
ax.scatter(X[:, 0][y == 1], X[:, 1][y == 1], -1, c='red', marker='^', s=50, label="Class 1")

# 設定圖表參數
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Decision Function")
ax.view_init(elev=30, azim=120)
ax.legend()


# 在 Streamlit 中顯示圖表
st.pyplot(fig)
