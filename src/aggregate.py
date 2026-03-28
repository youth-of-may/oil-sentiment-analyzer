from pathlib import Path
import os
import pandas as pd
from datetime import datetime
import json
import time

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def aggregate(df: pd.DataFrame):
    aggregated = pd.DataFrame()
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
    aggregated = df.groupby('publishedAt')['sentiment_score'].mean()
    return aggregated.reset_index()

def merge(sentiment: pd.DataFrame, prices: pd.DataFrame):
    sentiment.rename(columns={'publishedAt': 'date'}, inplace=True)
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
    prices['date'] = pd.to_datetime(prices['date']).dt.date
    combined = pd.merge(sentiment, prices, on='date', how='left')
    combined['brent_close'] = combined['brent_close'].ffill()
    combined['wti_close'] = combined['wti_close'].ffill()
    return combined


def save_data(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__=="__main__":
    combined = pd.DataFrame()

    sentiment = pd.read_csv(PROCESSED_DIR / 'sentiment_scores.csv')
    sentiment = aggregate(sentiment)

    oil = pd.read_csv(RAW_DIR / 'oil_prices.csv')

    combined = merge(sentiment, oil)

    save_data(combined, 'sentiment_oil.csv')