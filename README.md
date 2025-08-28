# Comment-Toxicity
This project implements a deep learning-based multi-label text classification model to detect different types of toxic comments. The model is trained to classify user-generated comments into multiple categories of toxicity, such as:  ✅ Toxic  ✅ Severe Toxic  ✅ Obscene  ✅ Threat  ✅ Insult  ✅ Identity Hate
Preprocessed raw text data using NLP techniques (tokenization, stopword removal, lemmatization, etc.)
Converted comments into numerical sequences using Tokenizer + Padding
Built a Bidirectional LSTM model with embeddings to capture contextual meaning
Applied multi-label classification with sigmoid activation for independent label predictions
Optimized training using EarlyStopping and evaluation metrics like Accuracy, Precision, Recall, and F1-Score

Model Performance
Training Accuracy: ~94%
Loss: ~0.15
Evaluated using a held-out test set with balanced performance across classes

Predicts multiple toxicity labels for a single comment
Can be deployed in web apps (Flask/Streamlit) for real-time predictions
Includes saved model (.h5) and tokenizer (.pkl) for reuse
