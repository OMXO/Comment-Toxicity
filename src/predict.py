# src/predict.py
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import clean_text

# Load everything once (no need to pass in every time)
MODEL_PATH = "saved_model/toxicity_model.h5"
TOKENIZER_PATH = "saved_model/tokenizer.pkl"
MAX_LEN = 150

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# Define your labels (make sure they match your training order)
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_comment(text: str) -> dict:
    """Predict multi-label toxicity for a single comment."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    preds = model.predict(padded)[0]

    # return as dictionary {label: probability}
    return {label: float(prob) for label, prob in zip(LABELS, preds)}
