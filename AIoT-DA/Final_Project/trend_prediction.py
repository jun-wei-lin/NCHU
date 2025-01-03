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
    if len(data['value'].dropna()) < max(order):
        raise ValueError(f"數據量不足以支持 ARIMA({order}) 模型，至少需要 {max(order)} 條數據。")
    try:
        model = ARIMA(data['value'], order=order)
        fit = model.fit()
        return fit
    except Exception as e:
        raise ValueError(f"ARIMA 模型訓練失敗: {e}")

def predict_trends(fit, steps=30):
    """預測未來趨勢並返回置信區間."""
    forecast_obj = fit.get_forecast(steps=steps)
    if forecast_obj is None:
        raise ValueError("無法生成預測對象，檢查模型輸入數據。")
    
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int()
    if forecast_ci is None or forecast_mean is None:
        raise ValueError("預測置信區間或預測均值為空，檢查模型結果。")

    # 使用模型的最後一個日期作為起點
    last_date = fit.data.dates[-1] if hasattr(fit.data, 'dates') and fit.data.dates is not None else fit.data.index[-1]
    forecast_index = pd.date_range(last_date, periods=steps + 1, freq="D")[1:]

    forecast_df = pd.DataFrame({
        "forecast": forecast_mean,
        "lower_bound": forecast_ci.iloc[:, 0],
        "upper_bound": forecast_ci.iloc[:, 1]
    }, index=forecast_index)

    return forecast_df

def plot_trends(data, forecast, font_path):
    """繪製改進後的趨勢圖並支持中文標籤."""
    my_font = fm.FontProperties(fname=font_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['value'], label="歷史數據", linestyle='-', marker='o', color='blue', alpha=0.8)
    forecast_index = pd.date_range(data.index[-1], periods=len(forecast)+1, freq="D")[1:]
    ax.plot(forecast_index, forecast, label="預測數據", linestyle='--', marker='x', color='orange', alpha=0.8)

    # 添加數據標籤，每隔一定間隔顯示
    interval = max(len(data) // 10, 1)  # 每 10 個點顯示一個標籤
    for i, value in enumerate(data['value']):
        if i % interval == 0:
            ax.text(data.index[i], value, f"{value}", fontsize=10, color="blue", ha="right", fontproperties=my_font)

    for i, value in enumerate(forecast):
        if i % interval == 0:
            ax.text(forecast_index[i], value, f"{value:.1f}", fontsize=10, color="orange", ha="left", fontproperties=my_font)

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.legend(prop=my_font, loc="upper left", fontsize=12)
    ax.set_title("趨勢預測", fontproperties=my_font, fontsize=16, pad=15)
    ax.set_xlabel("日期", fontproperties=my_font, fontsize=12)
    ax.set_ylabel("文章數量", fontproperties=my_font, fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.7)

    return fig


