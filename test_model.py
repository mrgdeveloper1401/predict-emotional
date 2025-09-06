# test_model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# load model
model_path = "./my_custom_fa_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# move into GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def predict_sentiment(texts):
    """
    predict for list of texts
    """
    # tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    # process result
    results = []
    for i, text in enumerate(texts):
        sentiment = "positive" if predictions[i] == 1 else "negative"
        confidence = probabilities[i][predictions[i]].item()

        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                "negative": round(probabilities[i][0].item(), 4),
                "positive": round(probabilities[i][1].item(), 4)
            }
        })

    return results


# example test
test_texts = [
    "این محصول واقعا عالی بود",
    "بدترین خرید عمرم بود",
    "نه خوب بود نه بد",
    "سرویس دهی خوبی داشتند",
    "خیلی دیر رسید و کیفیت پایینی داشت",
    "از پشتیبانی بسیار راضی هستم",
    "پیشنهاد نمی‌کنم اصلا",
    "قیمت مناسبی داشت"
]

print("🧪 Testing the trained model:\n")
predictions = predict_sentiment(test_texts)

for pred in predictions:
    print(f"📝 Text: {pred['text']}")
    print(f"🎯 Sentiment: {pred['sentiment']}")
    print(f"📊 Confidence: {pred['confidence']:.4f}")
    print(
        f"📈 Probs - Negative: {pred['probabilities']['negative']:.4f}, Positive: {pred['probabilities']['positive']:.4f}")
    print("-" * 50)
