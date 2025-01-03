from transformers import pipeline

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 初始化情感分析管道，啟用自動截斷
    sentiment_pipeline = pipeline("sentiment-analysis", framework="pt", truncation=True)

    # 確保輸入是字符串列表
    if not isinstance(articles, list) or not all(isinstance(article, str) for article in articles):
        raise ValueError("輸入應為包含字符串的列表 (List[str])")

    # 分析情感
    results = sentiment_pipeline(articles)
    return results
