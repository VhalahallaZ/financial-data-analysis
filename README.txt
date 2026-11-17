================================================================================
DYNAMIC ASSET ALLOCATION REGIME SWITCHING PROJECT - README
================================================================================

1. DATA SOURCES & ASSETS
--------------------------------------------------------------------------------

A. Financial Market Data (Source: Yahoo Finance)
- SPY: SPDR S&P 500 ETF Trust (US Large-Cap Equities).
- AGG: iShares Core U.S. Aggregate Bond ETF (US Total Investment-Grade Bonds).
- VBMFX: Vanguard Total Bond Market Index Fund. 
  Note: AGG started trading on September 22, 2003. To cover the 2000-2002 Dot-Com 
  crash, VBMFX is used as a proxy to stitch bond prices before September 2003.
- Note on Prices: All asset prices are downloaded using total-return adjustment 
  (auto_adjust=True) so that stock dividends and bond interest coupons are reinvested 
  without artificial drops on ex-dates.

B. Macroeconomic Fundamental Data (Source: FRED - St. Louis Federal Reserve)
- T10Y2Y: 10-Year Treasury Yield minus 2-Year Treasury Yield (Yield Curve Spread).
- FEDFUNDS: Effective Federal Funds Rate (Central Bank Policy Rate).
- CPIAUCSL: Consumer Price Index for All Urban Consumers (Headline Inflation).
- UNRATE: Civilian Unemployment Rate (Labor Market Stress).

C. Raw Data CSV File:
- Name: "Dynamic asset allocation raw data.csv"
- How it was constructed: Market prices and economic series are merged using an 
  outer join on Date. Because market prices are daily while economic series are 
  monthly, this keeps all original timestamps and raw values without filling in fake numbers.

--------------------------------------------------------------------------------
2. DATE RANGES & TIME PERIODS
--------------------------------------------------------------------------------

- Total Data Download Window: 1998-01-01 to 2026-09-30
  (Starts in 1998 to allow a 2-year buffer for calculating 12-month trailing returns).
- Feature Matrix Window: 2000-01-31 to 2026-09-30 (321 monthly observations).
- Initial Training Window: 2000-01 to 2009-12 (First 120 months / 10 years).
- Out-of-Sample Backtest Window: 2010-01 to 2026-09 (201 months).

Realistic Publication Lag (Preventing Look-Ahead Bias):
Government economic surveys (CPI and Unemployment) are officially announced by the 
Bureau of Labor Statistics (BLS) with a 1 to 2 week delay after the month ends 
(e.g., September CPI and Unemployment numbers are released in mid-October).
Therefore, an investor rebalancing at the end of September does not have September 
data yet. To make the model realistic, a 1-month shift (.shift(1)) is applied to 
CPI and Unemployment before running the model.

--------------------------------------------------------------------------------
3. STEPS TO REPRODUCE ALL ANALYSES
--------------------------------------------------------------------------------

Google Colab Notebook Link:
https://colab.research.google.com/github/VhalahallaZ/financial-data-analysis/blob/main/notebooks/regime_analysis.ipynb

Steps to Run:
1. Open the Google Colab link above in your web browser.
2. In the top navigation menu, click "Runtime" -> "Run all" (or press Ctrl + F9).
3. The notebook runs automatically from start to finish:
   - Installs all required Python dependencies.
   - Downloads and cleans all market and macroeconomic data.
   - Executes the Exploratory Data Analysis (EDA) and displays all 4 investigation figures.
   - Fits the Markov Regime Switching and walk-forward Random Forest models.
   - Runs the backtests with 10 bps transaction fees and monthly asset drift.
   - Displays the master performance table (returns, volatility, Sharpe, Sortino, Max Drawdown, Calmar).
   - Generates and plots the master cumulative returns, drawdowns, regime switching probabilities, and Gini feature importance charts.
   - Displays the latest prediction date, model probabilities, and recommended portfolio weights at the bottom.
================================================================================
