from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(data):
    """清理並準備數據."""
    data.rename(columns={"date": "ds", "value": "y"}, inplace=True)
    return data

def train_prophet_model(data):
    """訓練 Prophet 模型."""
    model = Prophet()
    model.fit(data)
    return model

def predict_trends(model, periods=30):
    """預測未來趨勢."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_trends(data, forecast):
    """繪製趨勢圖."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['ds'], data['y'], label="歷史數據")
    plt.plot(forecast['ds'], forecast['yhat'], label="預測數據")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    plt.legend()
    plt.show()
