# src/evaluate.py
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Load model & tokenizer
model = load_model("saved_model/toxicity_model.h5")
tokenizer = joblib.load("saved_model/tokenizer.pkl")

# Load test data
df = pd.read_csv("data/dataset.csv")
df["comment_text"] = df["comment_text"].apply(clean_text)

X = tokenizer.texts_to_sequences(df["comment_text"])
X_pad = pad_sequences(X, maxlen=150, padding="post")

y_true = df[labels]
y_pred = (model.predict(X_pad) > 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=labels))
