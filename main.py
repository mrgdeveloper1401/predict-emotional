# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn

app = FastAPI(
    title="Persian Sentiment Analysis API",
    description="API for analyzing sentiment of Persian texts using BERT",
    version="1.0.0"
)



def load_model():
    try:
        model_path = "./my_custom_fa_sentiment_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


model, tokenizer, device = load_model()


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict


class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    success: bool = True


@app.get("/")
async def root():
    return {
        "message": "Persian Sentiment Analysis API",
        "status": "active",
        "model": "BERT Persian Sentiment"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=SentimentResult)
async def predict_single(request: TextRequest):
    """Sentiment analysis of a single text"""
    try:
        results = predict_sentiment([request.text])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=SentimentResponse)
async def predict_batch(request: BatchRequest):
    """Batch sentiment analysis of multiple texts"""
    try:
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts per request")

        results = predict_sentiment(request.texts)
        return SentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def predict_sentiment(texts: List[str]):
    """Sentiment prediction function"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

    results = []
    for i, text in enumerate(texts):
        sentiment = "negative" if predictions[i] == 1 else "positive"
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
