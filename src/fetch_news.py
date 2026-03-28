from newsapi import NewsApiClient
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import time


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

load_dotenv(Path(__file__).parent.parent / ".env")

# Initializing collection dates:
start_date = '2026-02-28'
end_date = '2026-03-27'

date_range = pd.date_range(start=start_date, end=end_date)
query = ['Middle East conflict crude oil', 'Iran Israel Strait Hormuz']

# Init
newsapi = NewsApiClient(api_key=os.environ.get("NEWS_API"))

def api_call(dateTime, query) -> pd.DataFrame:
    time.sleep(0.5)
    dateString = dateTime.strftime("%Y-%m-%d")
    crude_oil = newsapi.get_everything(q=query,
                                      from_param=dateString,
                                      to=dateString,
                                      language='en',
                                      sort_by='publishedAt',
                                      page_size=50
                                      )
    df = pd.json_normalize(crude_oil['articles'])
    return df

def load_news():
    global_df = pd.DataFrame()
    for q in query:
        for d in date_range:
            global_df = pd.concat([global_df, api_call(d, q)], ignore_index=True)
    print(global_df.head())
    return global_df

def clean(df: pd.DataFrame):
    df['title'] = df['title'].astype(str).str.replace("\"", "", regex=False)
    df['description'] = df['description'].astype(str).str.replace("\"", "", regex=False)
    df.drop_duplicates(subset=['title', 'source.name'], inplace=True)
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
    
