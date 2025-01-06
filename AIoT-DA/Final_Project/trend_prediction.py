from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm  # 匯入字體管理
import os  # 確保匯入 os 模組

def prepare_data(data):
    """清理並準備數據，按月處理。"""
    if 'month' not in data.columns:
        raise ValueError("數據中缺少 'month' 列，無法進行處理。")

    data['month'] = pd.to_datetime(data['month'])
    data.set_index('month', inplace=True)

    # 填充缺失值
    data['value'] = data['value'].fillna(0)
    return data

    # 處理缺失值
    data['value'] = data['value'].fillna(0)  # 將缺失值填充為 0
    return data

def train_arima_model(data, order=(1, 1, 1)):
    """訓練 ARIMA 模型."""
    try:
        model = ARIMA(data['value'], order=order)
        fit = model.fit()
        return fit
    except Exception as e:
        raise ValueError(f"ARIMA 模型訓練失敗：{e}")

def predict_trends(fit, steps=6):
    """預測未來每月趨勢並返回整數結果."""
    forecast = fit.forecast(steps=steps)
    forecast = forecast.clip(lower=0).round().astype(int)  # 限制最小值為 0 並轉為整數
    return forecast

def plot_trends(data, forecast, font_path):
    """
    繪製按月的趨勢圖，包括歷史數據和未來預測。

    Args:
        data (pd.DataFrame): 包含每月數據的歷史數據。
        forecast (pd.Series): 預測數據，每月的預測值。
        font_path (str): 字體文件路徑。

    Returns:
        fig: Matplotlib 圖像對象。
    """
    if data.empty or forecast.empty:
        raise ValueError("無有效數據進行繪圖。")

    # 加載字體
    my_font = fm.FontProperties(fname=font_path)

    # 限制歷史數據範圍（最多 12 個月）
    limited_data = data[-12:]  # 僅保留過去 12 個月數據

    # 預測數據
    forecast_index = pd.date_range(start=limited_data.index[-1] + pd.DateOffset(months=1), periods=len(forecast), freq="M")

    # 繪圖
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(limited_data.index, limited_data['value'], label="歷史數據", linestyle='-', marker='o', color='blue')
    ax.plot(forecast_index, forecast, label="預測數據", linestyle='--', marker='x', color='orange')

    # 加入分界線
    ax.axvline(x=limited_data.index[-1], color='red', linestyle='--', label="預測起點")

    # 格式化 X 軸日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()  # 自動旋轉日期標籤

    # 中文標籤
    ax.legend(prop=my_font, loc="upper left")
    ax.set_title("按月趨勢預測", fontproperties=my_font, fontsize=16)
    ax.set_xlabel("月份", fontproperties=my_font, fontsize=12)
    ax.set_ylabel("文章數量", fontproperties=my_font, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig
