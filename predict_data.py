import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
import torch


# read data
# df = pd.read_csv("datasets/Snappfood-Sentiment-Analysis.csv", on_bad_lines="skip")

# drop column and save new csv-file
# del df['Unnamed: 0']
# df.to_csv("datasets/Snappfood-Sentiment-Analysis.csv", index=False)
#-------------------------------------------------------------------------------------

# print sum null data
# print(df.isnull().sum())

# drop null data
# df = df.dropna(subset=["label_id"])
# print(df.isnull().sum())

# save after drop null data
# df.to_csv("datasets/Snappfood-Sentiment-Analysis_drop_null.csv", index=False)
#-------------------------------------------------------------------------------------

# read new dataset after drop null data
# df_2 = pd.read_csv('datasets/Snappfood-Sentiment-Analysis_drop_null.csv')

# read Dimensions data
# print("shape data is: ", df_2.shape)
# print("ndim is: ", df_2.ndim)
# print("columns data is", df_2.columns)
# print("min data is: ", df_2['label_id'].min())
# print("max data is: ", df_2['label_id'].max())
# print(df_2.head())
# print(df_2.isnull().sum())

# convert label_id into integer
# df_2['label_id'] = df_2['label_id'].astype(int)
# df_2.to_csv("datasets/Snappfood-Sentiment-Analysis_drop_null_label_id_to_int.csv", index=False)
#-------------------------------------------------------------------------------------

# read data after convert label_id into int
df_3 = pd.read_csv("datasets/Snappfood-Sentiment-Analysis_drop_null_label_id_to_int.csv")
# print(df_3.head())
# print(df_3.columns)

# Check the distribution of columns
# print(df_3['label'].value_counts())
# print(df_3['label_id'].value_counts())

# div train test data
X_train, X_val, y_train, y_val = train_test_split(
    df_3['comment'].tolist(),
    df_3['label_id'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_3['label_id']
)
# print("train data is: ", len(X_train))
# print("val data is: ", len(X_val))
# -----------------------------------------------------------------

# tokenize and create database
model_name = "HooshvareLab/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

# Convert our data into a torch Dataset
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

train_dataset = SentimentDataset(train_encodings, y_train)
val_dataset = SentimentDataset(val_encodings, y_val)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# ---------------------------------------------------------------------

# training model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 4
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs} - Average training loss: {avg_train_loss:.4f}')

print("Training complete!")

# save model
model.save_pretrained('./my_custom_fa_sentiment_model')
tokenizer.save_pretrained('./my_custom_fa_sentiment_model')
