# %%
import streamlit as st
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "models/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# =========================
# NLP TOOLS
# =========================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =========================
# CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    tokens = text.split()

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_tokens)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")

st.write("Paste a news article below.")

news_input = st.text_area("News Article")

if st.button("Predict"):

    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        cleaned_news = clean_text(news_input)

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)

        probability = model.predict_proba(news_vector)

        confidence = probability.max() * 100

        if prediction[0] == 1:
            st.success(f"TRUE News ({confidence:.2f}% confidence)")
        else:
            st.error(f"FAKE News ({confidence:.2f}% confidence)")


