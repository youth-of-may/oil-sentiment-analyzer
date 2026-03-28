from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
import json
import time


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


def analyze_sentiment(news_titles: list, news_descriptions: list):

    try:
        time.sleep(2)
        numbered = "\n".join([f"{i+1}. Title: {t} | Description: {d}" 
                      for i, (t, d) in enumerate(zip(news_titles, news_descriptions))])
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages = [
                {
                    "role": "system",
                    "content": "You are a news sentiment analyst. Output ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
            You are analyzing multiple news articles related to the Iran–Israel conflict and its impact on global oil prices.

            Each article is a pair of (Title, Description).

            Articles:
            {numbered}

            Return exactly {len(news_titles)} scores, one per numbered article.

            Instructions:
            - Treat each (Title, Description) pair as a separate article.
            - Evaluate sentiment based on impact on global oil prices.
            - Focus on tone, severity, and implications (e.g., escalation, disruption, recovery, stability).

            Scoring:
            - Return a number between -1 and 1:
            - -1 = Extremely negative impact
            -  0 = Neutral or unclear
            - +1 = Extremely positive impact

            Output format:
            - Return a JSON array of numbers
            - The number of scores MUST match the number of articles
            - Preserve order

            Example output: [-0.75, 0.1, 0.6]


            Do NOT include any explanation or extra text.

            CRITICAL: Return ONLY a raw JSON array. No keys, no objects, no markdown, no explanation.
            Correct: [-0.75, 0.1, 0.6]
            Wrong: {{"scores": [-0.75, 0.1, 0.6]}}
            Wrong: Here are the scores: [-0.75, 0.1, 0.6]
            """
                }
            ]
        )
        #print(str(counter) + " " + response.choices[0].message.content)
        raw = response.choices[0].message.content
        raw = raw.replace('NA', '0').replace('null', '0')
        scores = json.loads(raw)
        counter+=1
        if (len(news_titles) == len(scores)):
            return scores
        else:
            print(f"Length mismatch: expected {len(news_titles)}, got {len(scores)}")
            return [0.0] * len(news_titles) 
    
    except RateLimitError:
        return [0.0] * len(news_titles)


def sentiment_loop(df: pd.DataFrame):
    sentiment_scores = []
    batch_size=15
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        sentiment_scores+= analyze_sentiment(batch['title'].tolist(), batch['description'].tolist())
    df['sentiment_score'] = sentiment_scores
    return df

def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__=="__main__":
    df = pd.read_csv(RAW_DIR / 'news_data.csv')
    df = sentiment_loop(df)
    save_processed(df[['publishedAt', 'sentiment_score']], 'sentiment_scores.csv')