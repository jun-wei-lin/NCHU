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

    # 分詞並啟用截斷與填充
    encoded_inputs = tokenizer(articles, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # 分析情感
    results = sentiment_pipeline(encoded_inputs["input_ids"].tolist())
    return results
