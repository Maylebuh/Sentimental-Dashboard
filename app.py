import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# === Load model ===
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", device=-1)

model = load_model()

# === App title with style ===
st.markdown(
    """
    <h1 style="color:#4CAF50;text-align:center;">🌟 Sentiment Analysis Dashboard 🌟</h1>
    """,
    unsafe_allow_html=True
)

# === Input section ===
st.markdown("### <span style='color:#2196F3'>Enter text manually or upload a file</span>", unsafe_allow_html=True)

user_input = st.text_area("Manual input (one text per line):", "")

uploaded_file = st.file_uploader("Or upload a TXT or CSV file", type=["txt", "csv"])

texts = []

if user_input.strip():
    texts.extend([line.strip() for line in user_input.strip().split("\n") if line.strip()])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        texts.extend([line.strip() for line in content.split("\n") if line.strip()])
    elif uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
        st.info("Preview of uploaded CSV file:")
        st.dataframe(df_upload.head())
        
        column = st.selectbox("Select column for sentiment analysis", df_upload.columns)
        texts.extend(df_upload[column].dropna().astype(str).tolist())

# === Analyze button ===
if st.button("🚀 Analyze"):
    if not texts:
        st.warning("Please provide input or upload a file.")
    else:
        with st.spinner("Analyzing sentiments..."):
            results = model(texts)
        
        df = pd.DataFrame(results)
        df["Text"] = texts
        df["Confidence"] = df["score"].round(2)
        df = df[["Text", "label", "Confidence"]].rename(columns={"label": "Sentiment"})

        st.markdown("## 🎯 Sentiment Analysis Results")
        st.dataframe(df)

        # === Visuals ===
        sentiment_counts = df["Sentiment"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Sentiment Distribution (Bar Chart)")
            st.bar_chart(sentiment_counts)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["#4CAF50", "#F44336", "#9E9E9E"])
            ax.set_title("Sentiment Breakdown")
            st.pyplot(fig)

        # === Export buttons ===
        st.markdown("### 💾 Download results")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv", mime="text/csv")
        st.download_button("Download JSON", df.to_json(orient="records"), file_name="sentiment_results.json", mime="application/json")

# === Footer ===
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:gray">Built with ❤️ using Streamlit and Hugging Face</p>
    """,
    unsafe_allow_html=True
)
