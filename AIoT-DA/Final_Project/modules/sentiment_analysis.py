from transformers import pipeline, AutoTokenizer

def analyze_sentiment(articles):
    """
    分析文章情感傾向（正面、中立、負面）。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 初始化情感分析管道，啟用自動截斷
    sentiment_pipeline = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese", framework="pt", truncation=True)
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

    results = []
    for article in articles:
        # 使用分詞器進行截斷
        encoded_input = tokenizer(article, truncation=True, max_length=512, return_tensors="pt")
        truncated_text = tokenizer.decode(encoded_input["input_ids"][0])
        

    # 分析情感
    sentiment = sentiment_pipeline(truncated_text)[0]
    results.append(sentiment)
  
    return results
