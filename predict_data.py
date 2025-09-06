import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, f1_score
import gc

# clean memory
torch.cuda.empty_cache()
gc.collect()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

try:
    df_3 = pd.read_csv("datasets/Snappfood-Sentiment-Analysis_drop_null_label_id_to_int.csv")
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df_3.shape}")
    print(f"Label distribution:\n{df_3['label_id'].value_counts()}")
except FileNotFoundError as fe:
    raise fe

# div data and train test data
X_train, X_val, y_train, y_val = train_test_split(
    df_3['comment'].tolist(),
    df_3['label_id'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_3['label_id']
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# load model
model_name = "HooshvareLab/bert-fa-base-uncased"
print(f"Loading tokenizer and model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# move into GPU
model.to(device)

# tokenize data
print("Tokenizing data...")
def tokenize_data(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )

train_encodings = tokenize_data(X_train)
val_encodings = tokenize_data(X_val)

# define Dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# create Dataset and DataLoader
train_dataset = SentimentDataset(train_encodings, y_train)
val_dataset = SentimentDataset(val_encodings, y_val)


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# optimize
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # decrease learning rate
epochs = 3

scaler = torch.amp.GradScaler('cuda')

print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        # move into gpu for none blocking
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # user mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Scal gradient for prevent underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # calc accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Average training loss: {avg_train_loss:.4f}')
    print(f'Training accuracy: {train_accuracy:.4f}')

    # calc
    model.eval()
    val_predictions = []
    val_true_labels = []
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_predictions.extend(preds.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions, average='binary')
    avg_val_loss = val_loss / len(val_loader)

    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation F1-Score: {val_f1:.4f}')
    print('-' * 50)

print("Training complete!")

# save model
output_dir = "./my_custom_fa_sentiment_model"
print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model saved successfully!")

# test model
def predict_sentiment(texts):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    results = []
    for i, text in enumerate(texts):
        sentiment = "positive" if predictions[i] == 1 else "negative"
        confidence = probabilities[i][predictions[i]].item()
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })

    return results

test_texts = [
    "این محصول واقعا عالی بود",
    "بدترین خرید عمرم بود",
    "نه خوب بود نه بد",
    "سرویس دهی خوبی داشتند",
    "خیلی دیر رسید و کیفیت پایینی داشت"
]

print("\nTesting model on sample texts:")
predictions = predict_sentiment(test_texts)
for pred in predictions:
    print(f"Text: {pred['text']}")
    print(f"Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.4f})")
    print()
