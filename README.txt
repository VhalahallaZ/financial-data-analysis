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
3. STEPS TO REPRODUCE ALL ANALYSES (METHODOLOGICAL PIPELINE)
--------------------------------------------------------------------------------

Google Colab Notebook Link:
https://colab.research.google.com/github/VhalahallaZ/financial-data-analysis/blob/main/notebooks/regime_analysis.ipynb

The code executes sequentially through the following 7 methodological steps:

Step 1: Data Ingestion & Proxy Stitching
- Downloads daily adjusted prices for SPY, AGG, and VBMFX from Yahoo Finance (1998 to 2026).
- Downloads daily and monthly sovereign series from FRED (T10Y2Y, FEDFUNDS, CPIAUCSL, UNRATE).
- Stitches AGG prices before its September 2003 inception using VBMFX price ratio scaling.

Step 2: Frequency Alignment & Publication Lag Adjustment
- Resamples all series to calendar month-end (ME).
- Applies an explicit 1-month reporting lag (.shift(1)) to CPI and Unemployment to replicate 
  real-world Bureau of Labor Statistics (BLS) release schedules and eliminate look-ahead bias.

Step 3: Feature Engineering & Target Labeling
- Computes 13 stationary features:
  * Equity Momentum: 3M, 6M, and 12M percentage returns of SPY.
  * Bond Momentum: 3M and 12M percentage returns of AGG.
  * Market Volatility: Annualized 3-month rolling realized volatility of daily SPY returns.
  * Macro Cycle: 10Y-2Y yield curve spread, YoY CPI inflation, and Fed Funds rate level.
  * Macro Momentum: 3-month delta of yield spread, Fed Funds rate, and unemployment rate.
- Defines ground-truth Risk-Off regime labels: Target = 1 if forward 3-month SPY return 
  is below -2% OR current trailing drawdown exceeds -10%; else Target = 0.

Step 4: Exploratory Data Analysis (EDA)
- Evaluates historical drawdown asymmetry across 4 major crises (2000, 2008, 2020, 2022).
- Tests yield curve inversion and unemployment acceleration as early-warning crisis indicators.
- Analyzes return skewness, fat tails, volatility clustering, and feature cross-correlations.

Step 5: Regime-Switching Model Fitting & Walk-Forward Inference
- Model 1 (Markov Regime Switching): Fits a 2-state Hamilton switching regression on 
  SPY returns and realized volatility to calculate smoothed posterior Risk-Off probabilities.
- Model 2 (Machine Learning Random Forest): Uses an expanding walk-forward window with 
  120 months initial training. For each month t from 2010 to 2026, the model trains strictly 
  on data up to t-1 and predicts the out-of-sample Risk-Off probability for month t.

Step 6: Backtesting Simulation & Friction Accounting
- Simulates portfolio allocations across 6 strategies: S&P 500 Buy & Hold, Static 60/40, 
  MRS Binary, MRS Continuous, ML Binary, and ML Continuous.
- Shifts model weights by 1 month to reflect month-end rebalancing executed for the following month.
- Deducts an institutional transaction fee of 10 bps (0.10%) per traded portfolio turnover.
- Simulates intra-month organic asset weight drift prior to month-end rebalancing.

Step 7: Performance Evaluation & Chart Generation
- Calculates performance metrics: Cumulative Total Return, Annualized Return (CAGR), 
  Annualized Volatility, Sharpe Ratio (Rf=0%), Sortino Ratio, Maximum Drawdown, and Calmar Ratio.
- Generates master cumulative wealth curves, underwater drawdown profiles, combined regime 
  probability trajectories, and mean Gini feature importance rankings.
- Outputs the latest regime prediction, probabilities, and tactical allocation recommendations.
================================================================================
