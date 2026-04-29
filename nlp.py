# =========================
# 1. IMPORTS
# =========================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# =========================
# 2. LOAD DATASET (5000 SAMPLES)
# =========================

df = pd.read_csv("IMDB Dataset.csv", encoding="latin-1")

df = df[['review', 'sentiment']]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ⚡ BALANCED DATASET
df = df.groupby("sentiment").sample(2500, random_state=42)

texts = df['review'].values
labels = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# =========================
# 🔥 MODEL 1: LOGISTIC REGRESSION
# =========================

print("\n--- Logistic Regression ---")

vectorizer = TfidfVectorizer(max_features=3000)

X_train_lr = vectorizer.fit_transform(X_train)
X_test_lr = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_lr, y_train)

lr_pred = lr_model.predict(X_test_lr)

lr_acc = accuracy_score(y_test, lr_pred)
print("LR Accuracy:", lr_acc)

# SAVE
with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("LR model saved ✔")

# =========================
# 🔥 MODEL 2: LSTM
# =========================

print("\n--- LSTM ---")

max_words = 8000
max_len = 64

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

lstm_model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model.fit(
    X_train_pad, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_test_pad, y_test)
print("LSTM Accuracy:", lstm_acc)

# SAVE
lstm_model.save("lstm_model.h5")

with open("lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("LSTM model saved ✔")

# =========================
# 🔥 MODEL 3: BERT
# =========================

print("\n--- BERT ---")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = bert_tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=64
)

test_encodings = bert_tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=64
)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, y_train)
test_dataset = IMDbDataset(test_encodings, y_test)

bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    logging_steps=50,
    save_strategy="no"
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

bert_eval = trainer.predict(test_dataset)
bert_preds = np.argmax(bert_eval.predictions, axis=1)

bert_acc = accuracy_score(y_test, bert_preds)
print("BERT Accuracy:", bert_acc)

# SAVE
bert_model.save_pretrained("bert_model")
bert_tokenizer.save_pretrained("bert_tokenizer")

print("BERT model saved ✔")

# =========================
# FINAL RESULTS
# =========================

print("\n================ FINAL RESULTS ================")
print("Logistic Regression:", lr_acc)
print("LSTM:", lstm_acc)
print("BERT:", bert_acc)