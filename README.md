# Oil Price & Geopolitical Sentiment Analyzer

Analyzes whether geopolitical news sentiment (Iran-Israel conflict) correlates with Brent and WTI crude oil price movements, using LLM-scored headlines and statistical correlation analysis.

## Stack
- **NewsAPI** — geopolitical headline retrieval
- **yfinance** — Brent and WTI historical price data
- **Groq (LLaMA 3.1)** — sentiment scoring per article
- **pandas** — data cleaning, aggregation, merging
- **scipy** — Pearson and Spearman correlation, lag analysis
- **Streamlit + Plotly** — interactive analytics dashboard

## Project Structure
```
oil-sentiment-analyzer/
├── data/
│   ├── raw/               # fetched articles and price CSVs
│   ├── processed/         # merged and aggregated outputs
│   └── correlation/       # basic, lag, and rolling correlation outputs
├── src/
│   ├── fetch_news.py      # NewsAPI retrieval
│   ├── sentiment.py       # Groq LLM scoring
│   ├── fetch_prices.py    # yfinance price pull
│   ├── aggregate.py       # dataframe merging and daily sentiment aggregation
│   └── correlate.py       # Pearson, Spearman, lag, rolling
├── app/
│   ├── dashboard.py       # overview dashboard
│   └── pages/
│       ├── 1_Correlation_Analysis.py  # correlation, lag, and rolling charts
│       └── 2_News_Browser.py          # articles by date with sentiment scores
├── requirements.txt
└── README.md
```

## Pipeline
1. **Fetch** — pull geopolitical headlines from NewsAPI for the target date range
2. **Score** — send each article to Groq (LLaMA 3.1) and retrieve a sentiment score
3. **Aggregate** — average daily sentiment scores across all articles per day
4. **Merge** — join daily sentiment with Brent and WTI closing prices from yfinance
5. **Correlate** — run Pearson and Spearman correlation, lag analysis (0–5 days), and 14-day rolling correlation

## Key Findings
- **Statistically significant negative correlation** between news sentiment and both Brent (Pearson r = -0.48, p = 0.0098) and WTI (r = -0.44, p = 0.021) — more negative sentiment, higher oil prices
- **Brent shows multi-day sensitivity** — significant correlation at lags 0, 2, 4, and 5, suggesting the global benchmark digests geopolitical news over a multi-day window rather than reacting instantaneously
- **WTI only significant at lag 0** — immediate reaction with no statistically significant delayed effect, consistent with its more domestically-driven pricing dynamics
- **Rolling correlation peaked March 18–20** (r ≈ -0.67 to -0.69 for both benchmarks), coinciding with the period of sharpest market sensitivity to conflict developments
- **Brent more responsive to Iran-Israel sentiment than WTI overall**, consistent with Brent's role as the Middle East-linked global benchmark

## Data Sources
- **NewsAPI** — headlines queried for Iran-Israel conflict coverage, ~28 days of articles (Feb 28 – Mar 27, 2025)
- **yfinance** — daily closing prices for Brent Crude (`BZ=F`) and WTI Crude (`CL=F`) for the same date range; weekends forward-filled to align with news data

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file with your API keys:
```
NEWSAPI_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Then run the pipeline in order:
```bash
python src/fetch_news.py
python src/sentiment.py
python src/fetch_prices.py
python src/aggregate.py
python src/correlate.py
```

Then launch the dashboard:
```bash
streamlit run app/dashboard.py
```

## Limitations
- 28 days of data — findings are exploratory, not generalizable
- NewsAPI free tier caps at 100 requests/day and limits historical range
- Weekend oil prices are forward-filled from the previous Friday, which may dampen weekend sentiment signal
- Sentiment is averaged per day across all articles regardless of source weight or recency within the day

## Pending
**Sentiment prediction model** — the next step is to train a supervised model that can predict a sentiment score for a given news article without calling the Groq API. The Groq-annotated scores will serve as training labels. Planned approach:

- Use the existing Groq-scored articles as the labeled dataset
- Fine-tune or train a lightweight text regression/classification model (e.g. a pre-trained transformer or TF-IDF + regression baseline) on the annotated data
- Expose predictions via a new dashboard page where a user can paste a headline and receive a predicted sentiment score