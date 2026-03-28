from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd


"""
  LLM Sentiment Scoring
  (batch headlines → score -1 to 1 per article)
"""

load_dotenv(Path(__file__).parent.parent / ".env")
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

client = OpenAI(
    api_key=os.environ.get("GROQ_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def analyze_sentiment(news_title: str, news_description: str) -> int:

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content":"You are a news sentiment analyst. Rate the sentiment from -1 (extremely negative) to 1 (extremely positive)."},
                {"role": "user", "content": f""" 
                 You are analyzing the sentiment of a news article related to the Iran–Israel conflict and its impact on global oil prices.
                Article:
                Title: {news_title}
                Description: {news_description}

                Instructions:
                - Focus on tone, word choice, and implied severity (e.g., escalation, disruption, recovery, stability).
                - Consider both the title and description together.
                - Evaluate sentiment specifically in terms of impact on global oil prices.

                Scoring:
                - Return a single number between -1 and 1:
                - -1 = Extremely negative impact (e.g., severe disruption, crisis, price shocks)
                -  0 = Neutral or unclear impact
                - +1 = Extremely positive impact (e.g., stability, recovery, easing prices)

                Output format:
                Return ONLY the number (e.g., -0.75, 0, 0.6)

                Do not include any explanation or additional text."""
                }
            ]
        )
        return response.choices[0].message.content
    
    except RateLimitError:
        return f" Explanation unavailable -- API rate limit reached."

def sentiment_loop(df: pd.DataFrame):
    df['sentiment_score'] = df.apply(lambda row: analyze_sentiment(row['title'], row['description']), axis=1)
    return df

def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__=="__main__":
    df = pd.read_csv(RAW_DIR / 'news_data.csv')
    df = sentiment_loop(df)
    save_processed(df[['publishedAt', 'sentiment_score']], 'sentiment_scores.csv')