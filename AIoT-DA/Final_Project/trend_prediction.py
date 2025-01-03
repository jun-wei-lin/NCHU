from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_trends(data, forecast):
    """繪製趨勢圖."""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['value'], label="歷史數據")
    forecast_index = pd.date_range(data.index[-1], periods=len(forecast)+1, freq="D")[1:]
    plt.plot(forecast_index, forecast, label="預測數據")
    plt.legend()
    plt.title("趨勢預測")
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.show()
