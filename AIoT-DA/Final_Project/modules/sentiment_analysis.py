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
        # 截斷長文本
        encoded_input = tokenizer(
            article,
            truncation=True,
            max_length=512,  # 最大序列長度
            return_tensors="pt"
        )
        # 分析情感
        sentiment = sentiment_pipeline(encoded_input["input_ids"].tolist()[0])[0]
        results.append(sentiment)

    return results
