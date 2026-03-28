from newsapi import NewsApiClient
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

load_dotenv(Path(__file__).parent.parent / ".env")

# Init
newsapi = NewsApiClient(api_key=os.environ.get("NEWS_API"))

def load_news():
    crude_oil = newsapi.get_everything(q='Middle East conflict crude oil',
                                      from_param='2026-02-27',
                                      to='2026-03-26',
                                      language='en',
                                      sort_by='publishedAt',
                                      )
    df = pd.json_normalize(crude_oil['articles'])
    return df 

def clean(df: pd.DataFrame):
    df['title'] = df['title'].astype(str).str.replace("\"", "", regex=False)
    df['description'] = df['description'].astype(str).str.replace("\"", "", regex=False)
    return df

def save_data(df: pd.DataFrame, filename: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df['source'] = df['source.name']    
    df = df[['publishedAt', 'title', 'description',  'source']]
    df.to_csv(RAW_DIR / filename, index=False)


if __name__== "__main__":
    df = load_news()
    df = clean(df)
    save_data(df, "news_data.csv")
    
