# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Apply cleaning
df["comment_text"] = df["comment_text"].apply(clean_text)

X = df["comment_text"]

# Multi-label targets
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[labels]

# Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=150, padding="post")

# Save tokenizer
joblib.dump(tokenizer, "saved_model/tokenizer.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(20000, 128, input_length=150),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dense(len(labels), activation="sigmoid")  # Multi-label output
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, callbacks=[es])

# Save model
model.save("saved_model/toxicity_model.h5")
