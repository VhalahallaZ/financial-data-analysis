import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import time
import os

def fetch_yfinance_data(tickers=["SPY", "AGG", "VBMFX"], start="2008-01-01", end=None):
    """
    Fetches daily price data from Yahoo Finance.
    """
    print(f"Fetching yfinance data for {tickers} from {start} to {end or 'present'}...")
    # Add a retry loop for API resilience
    for attempt in range(3):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True)
            if data.empty:
                raise ValueError("No data returned from yfinance")
            
            # With auto_adjust=True, 'Close' is already total-return adjusted for splits and dividends
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    adj_close = data['Close']
                elif 'Adj Close' in data.columns.levels[0]:
                    adj_close = data['Adj Close']
                else:
                    raise KeyError("Could not find 'Close' or 'Adj Close' in yfinance columns")
            else:
                if 'Close' in data.columns:
                    adj_close = data['Close']
                elif 'Adj Close' in data.columns:
                    adj_close = data['Adj Close']
                else:
                    raise KeyError("Could not find 'Close' or 'Adj Close' in yfinance columns")
            return adj_close
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    raise RuntimeError("Failed to fetch yfinance data after 3 attempts")

def fetch_fred_data(start="2008-01-01", end=None):
    """
    Fetches macroeconomic data from FRED using pandas_datareader.
    Falls back to direct URL download if pandas_datareader fails.
    """
    series_ids = {
        "T10Y2Y": "10Y2Y_Yield_Spread",
        "CPIAUCSL": "CPI_Inflation_Index",
        "FEDFUNDS": "Fed_Funds_Rate",
        "UNRATE": "Unemployment_Rate"
    }
    
    print(f"Fetching FRED data for {list(series_ids.keys())}...")
    
    # Try pandas_datareader first
    try:
        df = web.DataReader(list(series_ids.keys()), 'fred', start=start, end=end)
        df = df.rename(columns=series_ids)
        print("FRED data loaded successfully via pandas_datareader.")
        return df
    except Exception as e:
        print(f"pandas_datareader failed: {e}. Attempting direct download fallback...")
        
    # Fallback: Direct CSV download from FRED API (requires no keys for public export link)
    dfs = []
    for sid, name in series_ids.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            temp_df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
            # Convert values to numeric, coercion replaces '.' or bad chars with NaN
            temp_df[sid] = pd.to_numeric(temp_df[sid], errors='coerce')
            temp_df = temp_df.rename(columns={sid: name})
            dfs.append(temp_df)
            time.sleep(1) # Polite delay
        except Exception as ex:
            print(f"Failed to fetch {sid} directly: {ex}")
            
    if not dfs:
        raise RuntimeError("Failed to fetch any FRED data via reader or fallback URL.")
        
    # Merge all dataframes on Date
    df_merged = dfs[0]
    for other_df in dfs[1:]:
        df_merged = df_merged.join(other_df, how='outer')
        
    # Filter by date range
    if start:
        df_merged = df_merged[df_merged.index >= start]
    if end:
        df_merged = df_merged[df_merged.index <= end]
        
    print("FRED data loaded successfully via fallback URLs.")
    return df_merged

def prepare_dataset(start_date="2000-01-01", end_date=None):
    """
    Loads both market and macro data, aligns to monthly frequency,
    engineers features, and labels regimes.
    """
    # Fetch data starting slightly earlier to handle lagged features (e.g. 1 year back)
    start_dt = pd.to_datetime(start_date)
    fetch_start = (start_dt - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    
    mkt = fetch_yfinance_data(start=fetch_start, end=end_date)
    
    # Stitch AGG using VBMFX proxy before AGG launch (September 2003)
    if 'AGG' in mkt.columns and 'VBMFX' in mkt.columns:
        first_agg_idx = mkt['AGG'].first_valid_index()
        if first_agg_idx is not None and mkt.index[0] < first_agg_idx:
            ratio = mkt.loc[first_agg_idx, 'AGG'] / mkt.loc[first_agg_idx, 'VBMFX']
            mkt['AGG'] = mkt['AGG'].fillna(mkt['VBMFX'] * ratio)
            
    macro = fetch_fred_data(start=fetch_start, end=end_date)
    
    # 1. Resample Market Data to Monthly
    # We want monthly close for price assets
    mkt_monthly = mkt.resample('ME').last()
    
    # 2. Resample Macro Data to Monthly
    # Macro data is usually monthly (starts at 1st of month). We align it to end of month.
    macro_monthly = macro.resample('ME').last()
    # Forward fill macro data in case of missing values
    macro_monthly = macro_monthly.ffill()
    
    # Combine datasets
    dataset = mkt_monthly.join(macro_monthly, how='inner')
    
    # 3. Feature Engineering
    # Calculate inflation YoY return (CPI index is CPI_Inflation_Index)
    dataset['CPI_YoY'] = dataset['CPI_Inflation_Index'].pct_change(12) * 100
    
    # Momentum features (using log returns or pct returns)
    dataset['SPY_3M_Ret'] = dataset['SPY'].pct_change(3)
    dataset['SPY_6M_Ret'] = dataset['SPY'].pct_change(6)
    dataset['SPY_12M_Ret'] = dataset['SPY'].pct_change(12)
    
    # Bond Momentum (AGG)
    dataset['AGG_3M_Ret'] = dataset['AGG'].pct_change(3)
    dataset['AGG_12M_Ret'] = dataset['AGG'].pct_change(12)
    
    # Volatility feature: 3-month rolling volatility of daily SPY returns, resampled to monthly
    spy_daily_ret = mkt['SPY'].pct_change()
    spy_daily_vol_3m = spy_daily_ret.rolling(63).std() * np.sqrt(252) # 63 trading days ~ 3 months
    spy_vol_monthly = spy_daily_vol_3m.resample('ME').last()
    dataset['SPY_Vol_3M'] = spy_vol_monthly
    
    # Macro momentum/differences
    dataset['Yield_Spread_Change_3M'] = dataset['10Y2Y_Yield_Spread'].diff(3)
    dataset['Fed_Funds_Change_3M'] = dataset['Fed_Funds_Rate'].diff(3)
    dataset['Unemployment_Change_3M'] = dataset['Unemployment_Rate'].diff(3)
    
    # 4. Target Labeling for ML Classifier
    # We define a "Bear Market" regime (Risk-Off) as when the forward 3-month return of SPY is negative,
    # OR when SPY is in a deep drawdown (> 10% from rolling peak).
    # Let's calculate drawdown first
    rolling_max = dataset['SPY'].cummax()
    drawdown = (dataset['SPY'] - rolling_max) / rolling_max
    dataset['Drawdown'] = drawdown
    
    # Target: 1 if next month's SPY return over the next 3 months is negative, or if current drawdown is deep.
    # To avoid look-ahead bias in features, the features will predict the *future* regime.
    # Target label: 1 (Bear/Risk-Off), 0 (Bull/Risk-On)
    # Let's define it as: forward 3-month return of SPY is less than -2% OR current drawdown is less than -10%
    forward_3m_ret = dataset['SPY'].pct_change(3).shift(-3)
    dataset['Forward_3M_Ret'] = forward_3m_ret
    dataset['Regime_Target'] = np.where((forward_3m_ret < -0.02) | (drawdown < -0.10), 1, 0)
    
    # Drop rows before start_date to remove initialization NaNs for returns
    dataset = dataset[dataset.index >= start_dt]
    
    # Fill remaining NaNs if any
    dataset = dataset.ffill().bfill()
    
    return dataset

if __name__ == "__main__":
    df = prepare_dataset("2010-01-01")
    print(df.tail(10))
    print("Columns:", df.columns)
    print("Dataset shape:", df.shape)
