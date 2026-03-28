from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import pandas as pd


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CORRELATION_DIR = Path(__file__).parent.parent / "data" / "correlation"
def basic_correlation(df: pd.DataFrame):
    # drop rows where either column is null
    # run pearson and spearman on sentiment_score vs brent_close
    # run pearson and spearman on sentiment_score vs wti_close
    # return a summary dict or dataframe with r value and p value for each

    # Calculate Pearson
    brent_pearson, b_pearson_p_value = pearsonr(df['sentiment_score'], df['brent_close'])

    # Calculate Spearman
    brent_spearman, b_spearman_p_value = spearmanr(df['sentiment_score'], df['brent_close'])

  
    # Calculate Pearson
    wti_pearson, w_pearson_p_value = pearsonr(df['sentiment_score'], df['wti_close'])
  
    # Calculate Spearman
    wti_spearman, w_spearman_p_value = spearmanr(df['sentiment_score'], df['wti_close'])

    dict = {
        'brent':{
            'pearson': {
                'value': brent_pearson,
                'p': b_pearson_p_value
            },
            'spearman': {
                'value': brent_spearman,
                'p': b_spearman_p_value
            }
        },
        'wti': {
           'pearson': {
                'value': wti_pearson,
                'p': w_pearson_p_value
            },
            'spearman': {
                'value': wti_spearman,
                'p': w_spearman_p_value
            }
        }
    }
    return pd.json_normalize(dict)


def lag_correlation(df, stock, max_lag=5):
    results = []
    for lag in range(0, max_lag + 1):
        shifted = df['sentiment_score'].shift(lag)
        corr, pval = pearsonr(
            shifted.dropna(), 
            df[stock].loc[shifted.dropna().index]
        )
        results.append({'lag_days': lag, 'pearson_r': corr, 'p_value': pval})
    return pd.DataFrame(results)

def rolling_correlation(df: pd.DataFrame, window_size: int, stock:str, lag=None):
    if lag:
        df['sentiment_score'] = df['sentiment_score'].shift(lag)
        rolling_corr = df['sentiment_score'].rolling(window=window_size).corr(df[stock])
    else:
        rolling_corr = df['sentiment_score'].rolling(window=window_size).corr(df[stock])
    return rolling_corr.reset_index()



def save_data(df: pd.DataFrame, filename: str, directory):
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / filename, index=False)


if __name__=="__main__":
    df = pd.read_csv(PROCESSED_DIR / 'sentiment_oil.csv')
    basic_corr = basic_correlation(df)
    lag_corr_brent = lag_correlation(df, 'brent_close')
    lag_corr_wti = lag_correlation(df, 'wti_close')
    rolling_corr_brent = rolling_correlation(df, 14, 'brent_close')
    rolling_corr_wti = rolling_correlation(df, 14, 'wti_close')
    save_data(basic_corr, 'basic_correlation.csv', CORRELATION_DIR)
    save_data(lag_corr_brent, 'brent_lag_correlation.csv', CORRELATION_DIR)
    save_data(lag_corr_wti, 'wti_lag_correlation.csv', CORRELATION_DIR)
    save_data(rolling_corr_brent, 'rolling_cor_brent.csv', CORRELATION_DIR)
    save_data(rolling_corr_wti, 'rolling_cor_wti.csv', CORRELATION_DIR)