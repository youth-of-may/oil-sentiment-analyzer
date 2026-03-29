import pandas as pd
from pathlib import Path
import os


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def join_df(sentiment: pd.DataFrame, news: pd.DataFrame):
    df = pd.merge(sentiment, news, on="publishedAt")
    df = df.sort_values(by="sentiment_score", ascending=True)
    df.dropna(subset=['title', 'sentiment_score'], inplace=True)
    df.drop_duplicates(inplace=True, keep="first")
    return df[['title', 'description', 'publishedAt', 'sentiment_score']]

def save_data(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__== "__main__":
    sentiment = pd.read_csv(PROCESSED_DIR / 'sentiment_scores.csv')
    news = pd.read_csv(RAW_DIR / 'news_data.csv')
    df = join_df(sentiment, news)
    save_data(df, "top_negativesentiment_news.csv")
    
