import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.preprocessing import PolynomialFeatures

# Streamlit UI 標題
st.title("2D SVM 與 3D 視覺化")

# 步驟 1: 生成 2D 數據集
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)

# 步驟 2: 在 2D 數據上訓練線性 SVM
model = svm.SVC(kernel='linear')
model.fit(X, y)

# 步驟 3: 將數據轉換為 3D 以進行視覺化（使用多項式特徵進行提升）
poly = PolynomialFeatures(degree=2)
X3d = poly.fit_transform(X)[:, 1:]  # 忽略偏差項（第一列）

# 定義用於決策邊界視覺化的網格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = model.decision_function(xy).reshape(xx.shape)

# 步驟 4: 使用 matplotlib 繪製 3D 圖
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 繪製決策平面
ax.plot_surface(xx, yy, Z, color='lightblue', alpha=0.5, rstride=100, cstride=100)

# 3D 散點圖顯示數據點
ax.scatter(X3d[y == 0][:, 0], X3d[y == 0][:, 1], X3d[y == 0][:, 2], color='blue', label='Class 0', alpha=0.6)
ax.scatter(X3d[y == 1][:, 0], X3d[y == 1][:, 1], X3d[y == 1][:, 2], color='red', label='Class 1', alpha=0.6)

# 標籤和標題
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3 (Transformed)')
ax.set_title('3D Visualization of 2D SVM Decision Boundary')
ax.legend()

# 在 Streamlit 中顯示圖形
st.pyplot(fig)
