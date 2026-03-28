from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import pandas as pd


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
def basic_correlation(df: pd.DataFrame):
    # drop rows where either column is null
    # run pearson and spearman on sentiment_score vs brent_close
    # run pearson and spearman on sentiment_score vs wti_close
    # return a summary dict or dataframe with r value and p value for each

    # Calculate Pearson
    corr_p, _ = pearsonr(df['sentiment_score'], df['brent_close'])
    print('Brent Close')
    print(f'Pearson Correlation: {corr_p:.3f}')

    # Calculate Spearman
    corr_s, p_value = spearmanr(df['sentiment_score'], df['brent_close'])

    print(f'Spearman Correlation: {corr_s:.3f} p-value: {p_value} \n')
    
    # Calculate Pearson
    corr_p, _ = pearsonr(df['sentiment_score'], df['wti_close'])
    print('WTI Close')
    print(f'Pearson Correlation: {corr_p:.3f}')

    # Calculate Spearman
    corr_s, p_value = spearmanr(df['sentiment_score'], df['wti_close'])

    print(f'Spearman Correlation: {corr_s:.3f} p-value: {p_value}')

def lag_correlation(df, max_lag=5):
    results = []
    for lag in range(0, max_lag + 1):
        shifted = df['sentiment_score'].shift(lag)
        corr, pval = pearsonr(
            shifted.dropna(), 
            df['brent_close'].loc[shifted.dropna().index]
        )
        results.append({'lag_days': lag, 'pearson_r': corr, 'p_value': pval})
    return pd.DataFrame(results)

if __name__=="__main__":
    df = pd.read_csv(PROCESSED_DIR / 'sentiment_oil.csv')
    basic_correlation(df)
    print(lag_correlation(df).head())