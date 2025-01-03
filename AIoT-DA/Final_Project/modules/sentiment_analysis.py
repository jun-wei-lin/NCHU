from transformers import pipeline, AutoTokenizer

def analyze_sentiment(articles):
    """
    使用情感分析管道分析文章的情感。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤及分數
    """
    # 初始化模型與分詞器
    sentiment_pipeline = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

    # 確保所有文本為字符串
    articles = [str(article) for article in articles]

    # 分詞並啟用截斷與填充
    encoded_inputs = tokenizer(articles, truncation=True, padding=True, max_length=512, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"].tolist()  # 提取編碼後的 token id

    # 分析情感
    results = []
    for ids in input_ids:
        try:
            # 嘗試分析情感
            sentiment = sentiment_pipeline({"text": tokenizer.decode(ids, skip_special_tokens=True)})
            if sentiment and isinstance(sentiment, list) and len(sentiment) > 0:
                results.append(sentiment[0])
            else:
                # 如果分析結果異常，加入空結果
                results.append({"label": "UNKNOWN", "score": 0.0})
        except Exception as e:
            # 捕獲異常，加入錯誤標記
            results.append({"label": "ERROR", "score": 0.0, "error": str(e)})

    return results
