# %%
# ==========================================
# FAKE NEWS DETECTION MODEL
# ==========================================

# ========= IMPORT LIBRARIES =========

import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ========= DOWNLOAD NLTK DATA =========

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ========= LOAD DATASETS =========

fake_df = pd.read_csv('/Users/zihad/Desktop/Project/News_detection/Fake.csv')
true_df = pd.read_csv('/Users/zihad/Desktop/Project/News_detection/True.csv')

# ========= CREATE LABELS =========

# Fake news = 0
fake_df["label"] = 0

# True news = 1
true_df["label"] = 1

# ========= COMBINE DATASETS =========

df = pd.concat([fake_df, true_df], axis=0)

# ========= SHUFFLE DATA =========

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========= COMBINE TITLE + TEXT =========

df["content"] = df["title"] + " " + df["text"]

# ========= TEXT CLEANING =========

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # tokenize text
    tokens = word_tokenize(text)

    # remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # join tokens back into text
    return " ".join(cleaned_tokens)


# ========= APPLY CLEANING =========

df["clean_content"] = df["content"].apply(clean_text)

# ========= FEATURES AND LABELS =========

X = df["clean_content"]
y = df["label"]

# ========= TRAIN TEST SPLIT =========

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ========= TF-IDF VECTORIZATION =========

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ========= TRAIN MODEL =========

model = LogisticRegression()

model.fit(X_train_vec, y_train)

# ========= MAKE PREDICTIONS =========

y_pred = model.predict(X_test_vec)

# ========= EVALUATE MODEL =========

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# TEST WITH CUSTOM NEWS
# ==========================================

news = [
    "The government announces a new climate change policy today."
]

# clean custom news
cleaned_news = [clean_text(text) for text in news]

# vectorize
news_vector = vectorizer.transform(cleaned_news)

# prediction
prediction = model.predict(news_vector)

# output result
if prediction[0] == 1:
    print("\nPrediction: True News")
else:
    print("\nPrediction: Fake News")



joblib.dump(
    model,
    "/Users/zihad/Desktop/Project/News_detection/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "/Users/zihad/Desktop/Project/News_detection/tfidf_vectorizer.pkl"
)


