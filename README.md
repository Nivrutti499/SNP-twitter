# BrandPulse AI — Twitter Sentiment Dashboard

A professional sentiment analysis dashboard built with Streamlit, powered by three ML models trained on the Sentiment140 dataset (1.6M tweets).

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🧠 Models

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 76.58% | 0.7703 |
| Naïve Bayes | 73.98% | 0.7386 |
| LSTM (Deep Learning) | ~74–76% | ~0.75 |

## 📦 Stack

- **Python 3.11** · Streamlit · TensorFlow · Scikit-learn
- Dataset: [Sentiment140](http://help.sentiment140.com/) — 1.6M labelled tweets

## 🏃 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
