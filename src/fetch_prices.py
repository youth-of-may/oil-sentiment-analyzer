import pandas as pd
from pathlib import Path
import yfinance as yf

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

if __name__=="__main__":
    df = yf.download(['BZ=F', 'CL=F'], start="2026-02-26", end="2026-03-27")
    df = df['Close'].reset_index()
    df.columns = ['date', 'brent_close', 'wti_close']
    df.to_csv(RAW_DIR / 'oil_prices.csv', index=False)