# 🎬 IMDB Sentiment Analysis (ML + LSTM + BERT)

This project is a **Natural Language Processing (NLP) system** that performs sentiment analysis on IMDB movie reviews using three different approaches:

- Logistic Regression (Machine Learning)
- LSTM (Deep Learning)
- BERT (Transformer Model)

It also includes a **Streamlit web app** for real-time predictions.

---

# 🚀 Features

- 🧠 3 AI models comparison (ML vs DL vs Transformer)
- ⚡ Fast prediction system
- 🌐 Streamlit web interface
- 💾 Model saving & reuse
- 📊 Performance comparison

---

# 📊 Models Used

## 1. Logistic Regression
- TF-IDF based text classification
- Fast and strong baseline model

## 2. LSTM (Deep Learning)
- Sequential text learning
- Captures context in sentences

## 3. BERT (Transformer)
- Pretrained language model
- Highest accuracy among all models

---

# 📁 Project Structure


---

# 📥 Dataset

The dataset is stored in Google Drive and is automatically downloaded when the project runs.

Link:
https://drive.google.com/file/d/11HHaTKNBipgieEe6TvjPkxn85PSS3XPw/view?usp=sharing

The dataset is loaded using `gdown`:

```python
import gdown
import pandas as pd

file_id = "11HHaTKNBipgieEe6TvjPkxn85PSS3XPw"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, "IMDB Dataset.csv", quiet=False)

df = pd.read_csv("IMDB Dataset.csv", encoding="latin-1")
