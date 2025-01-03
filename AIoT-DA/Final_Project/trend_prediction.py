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
    model = ARIMA(data['value'], order=order)
    fit = model.fit()
    return fit

def predict_trends(fit, steps=30):
    """預測未來趨勢."""
    forecast = fit.forecast(steps=steps)
    return forecast

def plot_trends(data, forecast, font_path):
    """繪製趨勢圖並支持中文標籤."""
    # 加載字體
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError(f"字體文件未找到：{font_path}")
    my_font = fm.FontProperties(fname=font_path)

    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['value'], label="歷史數據", linestyle='-', marker='o')
    forecast_index = pd.date_range(data.index[-1], periods=len(forecast)+1, freq="D")[1:]
    ax.plot(forecast_index, forecast, label="預測數據", linestyle='--', marker='x')
    
    # 中文標籤
    ax.legend(prop=my_font)
    ax.set_title("趨勢預測", fontproperties=my_font)
    ax.set_xlabel("日期", fontproperties=my_font)
    ax.set_ylabel("值", fontproperties=my_font)
    return fig
