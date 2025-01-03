from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 匯入字體管理
import os  # 確保匯入 os 模組

def prepare_data(data):
    """清理並準備數據."""
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

def train_arima_model(data, order=(1, 1, 1)):
    """訓練 ARIMA 模型."""
    model = ARIMA(data['value'], order=(2, 1, 2))  # 更高階的模型
    fit = model.fit()
    return fit

def plot_trends(data, forecast, font_path):
    """繪製改進後的趨勢圖並支持中文標籤."""
    my_font = fm.FontProperties(fname=font_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['value'], label="歷史數據", linestyle='-', marker='o', color='blue')
    forecast_index = pd.date_range(data.index[-1], periods=len(forecast)+1, freq="D")[1:]
    ax.plot(forecast_index, forecast, label="預測數據", linestyle='--', marker='x', color='orange')

    # 避免標籤擁擠，每隔一定間隔顯示
    interval = max(1, len(data) // 10)
    for i, value in enumerate(data['value']):
        if i % interval == 0:
            ax.text(data.index[i], int(value), f"{int(value)}", fontsize=10, color='blue', ha='right', fontproperties=my_font)
    for i, value in enumerate(forecast):
        if i % interval == 0:
            ax.text(forecast_index[i], int(value), f"{int(value)}", fontsize=10, color='orange', ha='left', fontproperties=my_font)

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.legend(prop=my_font, loc="upper left")
    ax.set_title("趨勢預測", fontproperties=my_font, fontsize=16)
    ax.set_xlabel("日期", fontproperties=my_font, fontsize=12)
    ax.set_ylabel("文章數量", fontproperties=my_font, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig
