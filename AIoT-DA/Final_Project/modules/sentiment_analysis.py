from transformers import pipeline

# 測試情感分析管道
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment_pipeline("我非常喜歡這個產品！"))

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。
    
    Args:
        articles (list): 文章內容列表
    
    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 明確指定使用 PyTorch 框架
    sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")  # 或 "tf" 用於 TensorFlow
    results = [sentiment_pipeline(article)[0] for article in articles]
    return results
