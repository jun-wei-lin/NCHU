from transformers import pipeline

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 使用 PyTorch 框架
    sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")
    results = [sentiment_pipeline(article)[0] for article in articles]
    return results
