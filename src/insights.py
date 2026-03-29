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

def newsPerDate(date):
    news = pd.read_csv(PROCESSED_DIR / "top_negativesentiment_news.csv")
    news['publishedAt'] = pd.to_datetime(news['publishedAt'])
    return news[news['publishedAt'].dt.date == date]

def returnBasicCorr():
    df = pd.read_csv(CORR_DIR / 'basic_correlation.csv')
    data = []

    for asset in ['brent', 'wti']:
        pearson_val = round(df[f'{asset}.pearson.value'].iloc[0],3)
        pearson_p = round(df[f'{asset}.pearson.p'].iloc[0],3)
        spearman_val = round(df[f'{asset}.spearman.value'].iloc[0],3)
        spearman_p = round(df[f'{asset}.spearman.p'].iloc[0],6)

        data.append({
            'asset': asset,
            'pearson': pearson_val,
            'spearman': spearman_val,
            'pvalue': min(pearson_p, spearman_p)
        })

    result = pd.DataFrame(data)
    return result.reset_index()

def returnAggDay():
    df = pd.read_csv(PROCESSED_DIR / 'sentiment_oil.csv')
    return df

def brent_lag():
    df = pd.read_csv(CORR_DIR / 'brent_lag_correlation.csv')
    return df

def wti_lag():
    df = pd.read_csv(CORR_DIR / 'wti_lag_correlation.csv')
    return df

def rolling_wti():
    df= pd.read_csv(CORR_DIR / 'rolling_cor_wti.csv' )
    return df

def rolling_brent():
    df= pd.read_csv(CORR_DIR / 'rolling_cor_brent.csv' )
    return df