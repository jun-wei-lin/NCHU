from transformers import pipeline

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。
    
    Args:
        articles (list): 文章內容列表
    
    Returns:
        list: 每篇文章的情感標籤及分數
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = [sentiment_pipeline(article) for article in articles]
    return results
