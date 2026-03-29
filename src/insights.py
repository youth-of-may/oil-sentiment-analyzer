import pandas as pd
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CORR_DIR = Path(__file__).parent.parent / "data" / "correlation"

sentiment_price = pd.read_csv(PROCESSED_DIR / 'sentiment_oil.csv')

def count_articles():
    news_data = pd.read_csv(RAW_DIR / 'news_data.csv')
    return len(news_data)


def average_sentiment():
    sentiment_scores = pd.read_csv(PROCESSED_DIR / 'aggregated_sentiment.csv')
    return round(sentiment_scores['sentiment_score'].mean(),2)

def brent_sentimentPrice():
    return sentiment_price[['date','sentiment_score','brent_close']]

def wti_sentimentPrice():
    return sentiment_price[['date','sentiment_score','wti_close']]

def returnTopNNews(top_n):
    news = pd.read_csv(PROCESSED_DIR / "top_negativesentiment_news.csv")
    return news.iloc[0:top_n]