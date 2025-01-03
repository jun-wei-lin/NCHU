from transformers import pipeline, AutoTokenizer

def analyze_sentiment(articles):
    """
    使用情感分析管道逐篇分析文章的情感，並提供解釋。

    Args:
        articles (list): 文章內容列表

    Returns:
        list: 每篇文章的情感標籤、分數及詳細分數解釋
    """
    # 初始化模型與分詞器
    sentiment_pipeline = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

    results = []
    for article in articles:
        # 使用分詞器進行截斷
        encoded_input = tokenizer(article, truncation=True, max_length=512, return_tensors="pt")
        truncated_text = tokenizer.decode(encoded_input["input_ids"][0], skip_special_tokens=True)
        
        # 分析情感，**新增返回所有分數的功能**
        analysis = sentiment_pipeline(truncated_text, return_all_scores=True)  # **新增**
        best_match = max(analysis[0], key=lambda x: x["score"])  # **選擇最高分的情感類別**
        
        # 保存結果，**包括詳細解釋**
        result = {
            "label": best_match["label"],  # 最終分類
            "score": best_match["score"],  # 信心分數
            "details": analysis[0]  # **包括所有類別分數的詳細信息**
        }
        results.append(result)

    return results
