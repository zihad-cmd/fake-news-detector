# Fake News Detector

A machine learning-based fake news detection system built using NLP techniques and Logistic Regression.

## Overview

This project classifies news articles as:

- Fake News
- True News

The model was trained using TF-IDF vectorization and Logistic Regression on a public fake news dataset.

## Dataset

The project uses two datasets:

- `Fake.csv` → fake news articles
- `True.csv` → real news articles

The datasets contain:
- title
- text
- subject
- date

## NLP Pipeline

The preprocessing pipeline includes:

- Lowercasing
- URL removal
- HTML tag removal
- Punctuation removal
- Number removal
- Tokenization
- Stopword removal
- Lemmatization

## Machine Learning Model

Model:
- Logistic Regression

Vectorization:
- TF-IDF

## Model Performance

Accuracy achieved:

```text
98.5%