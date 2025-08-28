# app.py
import streamlit as st
import pandas as pd
from src.predict import predict_comment

st.title("🛡️ Multi-label Toxic Comment Detection")
st.write("Enter a comment and get predictions for multiple toxicity categories.")

# ---- Single prediction ----
user_input = st.text_area("Enter your comment:")
if st.button("Predict"):
    results = predict_comment(user_input)
    st.write("### Results:")
    for label, prob in results.items():
        st.write(f"- **{label}**: {prob:.4f}")  # show 4 decimals

# ---- Bulk prediction ----
st.subheader("📂 Bulk Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with 'comment_text' column", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_results = df["comment_text"].apply(predict_comment)  # list of dicts
    results_df = pd.DataFrame(df_results.tolist())
    output = pd.concat([df, results_df], axis=1)

    st.write("### Bulk Results")
    st.dataframe(output)

    st.download_button(
        "⬇️ Download Results",
        output.to_csv(index=False),
        "results.csv",
        "text/csv"
    )
