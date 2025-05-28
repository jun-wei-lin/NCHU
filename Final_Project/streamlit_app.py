import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Q-Network 定義 === #
class QNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# === 輔助函數 === #
def load_data(path='data/real_discount_data.csv'):
    return pd.read_csv(path)

def train_simple_qnet(df, episodes=5, batch_size=128):
    state = df[["age", "region", "history"]].values / [60, 3, 1]
    action = df["action"].values
    reward = df["reward"].values
    next_state = np.vstack([state[1:], state[-1]])
    done = np.array([0] * (len(state)-1) + [1])

    Xs = torch.tensor(state, dtype=torch.float32)
    Xa = torch.tensor(action, dtype=torch.long)
    Xr = torch.tensor(reward, dtype=torch.float32)
    Xs_next = torch.tensor(next_state, dtype=torch.float32)
    Xdone = torch.tensor(done, dtype=torch.float32)

    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        for i in range(0, len(Xs), batch_size):
            s = Xs[i:i+batch_size]
            a = Xa[i:i+batch_size]
            r = Xr[i:i+batch_size]
            s_next = Xs_next[i:i+batch_size]
            d = Xdone[i:i+batch_size]

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_q_next = target_net(s_next).max(1)[0]
                target = r + 0.99 * max_q_next * (1 - d)
            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        target_net.load_state_dict(q_net.state_dict())

    return q_net

def evaluate_model(q_net, df):
    X = df[["age", "region", "history"]].values / [60, 3, 1]
    y_true = df["converted"].values
    state_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = q_net(state_tensor)
        actions = torch.argmax(preds, dim=1).numpy()

    predicted_converted = []
    for i, a in enumerate(actions):
        rec_discount = [0.0, 0.05, 0.10, 0.20][a]
        actual_discount = df["discount"].iloc[i]
        if np.isclose(rec_discount, actual_discount):
            predicted_converted.append(df["converted"].iloc[i])
        else:
            predicted_converted.append(0)

    acc = accuracy_score(y_true, predicted_converted)
    report = classification_report(y_true, predicted_converted, output_dict=True)
    cm = confusion_matrix(y_true, predicted_converted)
    return acc, report, cm

# === Streamlit App === #
st.set_page_config(page_title="動態折扣強化學習 Demo", layout="wide")
st.title("🔁 智慧折扣推薦系統 - 強化學習 DQN")

menu = st.sidebar.radio("選擇功能：", [
    "1️⃣ 顧客折扣推薦",
    "2️⃣ 模型預測評估",
    "3️⃣ 模擬資料視覺化"
])

df = load_data()
q_net = train_simple_qnet(df)

if menu.startswith("1"):
    st.subheader("📍 模擬顧客折扣推薦")
    age = st.slider("年齡", 20, 60, 35)
    region = st.selectbox("地區 (0:北, 1:中, 2:南, 3:東)", [0, 1, 2, 3])
    history = st.radio("是否曾購買過", [0, 1])

    state = torch.tensor([[age/60, region/3, history]], dtype=torch.float32)
    with torch.no_grad():
        q_values = q_net(state)
        best_action = torch.argmax(q_values).item()
        discount = ["0%", "5%", "10%", "20%"][best_action]

    st.success(f"推薦折扣為：💡 {discount}")
    st.json({"Q-Values": q_values.numpy().round(2).tolist()[0]})

elif menu.startswith("2"):
    st.subheader("📊 模型預測與分類準確率")
    acc, report, cm = evaluate_model(q_net, df)

    st.metric("預測準確率", f"{acc*100:.2f}%")
    st.text("分類報告：")
    st.json(report)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("預測"); ax.set_ylabel("實際")
    st.pyplot(fig)

elif menu.startswith("3"):
    st.subheader("📈 模擬資料統計分析")
    st.write("轉換率 vs 折扣：")
    chart = df.groupby("discount")["converted"].mean().reset_index()
    st.bar_chart(chart.set_index("discount"))

    st.write("年齡分布：")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["age"], bins=20, edgecolor='black')
    st.pyplot(fig2)
