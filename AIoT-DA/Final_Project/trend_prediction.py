from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm  # 匯入字體管理
import os  # 確保匯入 os 模組

def prepare_data(data):
    """清理並準備數據."""
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['value'] = pd.to_numeric(data['value'], errors='coerce')  # 非數字轉為 NaN
    data.dropna(subset=['value'], inplace=True)  # 移除 NaN
    data['value'] = data['value'].clip(lower=0, upper=data['value'].quantile(0.95))  # 限制最大值在95分位
    return data

def train_arima_model(data, order=(1, 1, 1)):
    """訓練 ARIMA 模型."""
    model = ARIMA(data['value'], order=(1, 1, 1))  # 更高階的模型
    fit = model.fit()
    return fit

def predict_trends(fit, steps=30):
    """預測未來趨勢並返回整數結果."""
    forecast = fit.forecast(steps=steps)
    forecast = forecast.clip(lower=0).round().astype(int)  # 限制最小值為 0 並轉為整數
    return forecast



def plot_trends(data, forecast, font_path, history_months=3):
    """
    繪製趨勢圖，限制歷史數據範圍，並支持中文標籤。

    Args:
        data (pd.DataFrame): 包含日期和數值的歷史數據。
        forecast (pd.Series): 預測數據。
        font_path (str): 字體文件路徑。
        history_months (int): 要顯示的歷史數據月份數。

    Returns:
        fig: Matplotlib 圖像對象。
    """
    # 加載字體
    my_font = fm.FontProperties(fname=font_path)

    # 限制歷史數據範圍（過去指定月數）
    now = pd.Timestamp.now()
    start_date = now - pd.DateOffset(months=history_months)
    limited_data = data[data.index >= start_date]  # 過濾數據

    # 生成預測日期範圍
    forecast_index = pd.date_range(start=limited_data.index[-1], periods=len(forecast) + 1, freq="D")[1:]

    # 繪圖
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(limited_data.index, limited_data['value'], label="歷史數據", linestyle='-', marker='o', color='blue')
    ax.plot(forecast_index, forecast, label="預測數據", linestyle='--', marker='x', color='orange')

    # 格式化 X 軸日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()  # 自動旋轉日期標籤

    # 自動設置 X 軸範圍，從過去指定月份到預測結束日期
    ax.set_xlim([limited_data.index.min(), forecast_index[-1]])

    # 中文標籤
    ax.legend(prop=my_font, loc="upper left")
    ax.set_title("趨勢預測", fontproperties=my_font, fontsize=16)
    ax.set_xlabel("日期", fontproperties=my_font, fontsize=12)
    ax.set_ylabel("文章數量", fontproperties=my_font, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig

