from transformers import pipeline, AutoTokenizer

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 初始化模型與分詞器
    sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    results = []
    for article in articles:
        # 使用分詞器對文本進行截斷
        truncated_text = tokenizer.decode(
            tokenizer.encode(article, truncation=True, max_length=512)
        )
        # 分析情感
        sentiment = sentiment_pipeline(truncated_text)[0]
        results.append(sentiment)

    return results
