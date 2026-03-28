from pathlib import Path
import os
import pandas as pd
from datetime import datetime
import json
import time

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def aggregate(df: pd.DataFrame):
    aggregated = pd.DataFrame()
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
    aggregated = df.groupby('publishedAt')['sentiment_score'].mean()
    return aggregated.reset_index()

def save_data(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__=="__main__":
    df = pd.read_csv(PROCESSED_DIR / 'sentiment_scores.csv')
    df = aggregate(df)
    save_data(df, 'aggregated_sentiment.csv')