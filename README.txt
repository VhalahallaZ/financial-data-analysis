================================================================================
MACRO-DRIVEN REGIME SWITCHING ASSET ALLOCATION PROJECT
================================================================================

PROJECT OVERVIEW:
This project builds and tests a quantitative dynamic asset allocation framework.
Instead of holding a fixed 60% stocks and 40% bonds portfolio, the strategy uses
macroeconomic and market indicators to predict whether the market is entering a
"Risk-Off" (bear market / high stress) regime or a "Risk-On" (bull market) regime.

When the model predicts a Risk-Off regime:
- The portfolio shifts into Core Aggregate Bonds (AGG) to protect capital.
When the model predicts a Risk-On regime:
- The portfolio holds Large-Cap U.S. Equities (SPY) to capture stock market growth.

Two distinct models are tested:
1. Markov Regime Switching (MRS): A traditional statistical 2-state Hamilton model 
   using 3-month stock returns and 3-month realized volatility.
2. Machine Learning Random Forest (ML): An expanding walk-forward decision tree 
   ensemble using 13 macroeconomic and market momentum features.

--------------------------------------------------------------------------------
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
3. PERFORMANCE SUMMARY (OUT-OF-SAMPLE: 2010 TO 2026)
--------------------------------------------------------------------------------

All strategies include a 10 bps (0.10%) fee per trade and monthly asset price drift:

Strategy                            Cumulative  CAGR    Vol    Sharpe  Sortino  Max DD   Calmar
-----------------------------------------------------------------------------------------------
S&P 500 Buy & Hold                    821.14%  14.18%  14.34%   0.988   0.917   -23.93%   0.592
Benchmark (60/40 SPY/AGG)             358.74%   9.52%   9.20%   1.035   0.980   -20.06%   0.475
Markov Switching (MRS - Binary)       292.90%   8.51%   9.09%   0.937   1.040   -21.12%   0.403
Markov Switching (MRS - Continuous)   296.46%   8.57%   8.44%   1.015   1.107   -18.41%   0.466
Machine Learning (RF - Binary)        444.73%  10.65%  11.10%   0.959   0.946   -19.96%   0.534
Machine Learning (RF - Continuous)    320.94%   8.96%   8.45%   1.060   1.022   -18.88%   0.474

Key Conclusions:
- ML Binary gives the highest total growth (444.73% vs 358.74% for 60/40).
- ML Continuous gives the best risk-adjusted performance (Sharpe 1.060 and Vol 8.45%).
- MRS Continuous gives the best downside tail protection (Max Drawdown -18.41% and Sortino 1.107).

--------------------------------------------------------------------------------
4. STEPS TO REPRODUCE ALL ANALYSES
--------------------------------------------------------------------------------

OPTION A: Run Online in Google Colab (Easiest, No Installation)
1. Open Google Colab (https://colab.research.google.com).
2. Go to File -> Open Notebook -> GitHub tab.
3. Paste repository URL: https://github.com/VhalahallaZ/financial-data-analysis
4. Select "notebooks/regime_analysis.ipynb".
5. In the top menu, click "Runtime" -> "Run all" (or press Ctrl + F9).
6. The notebook will automatically download the data, train the models, run the 
   backtests, print the performance table, show all plots, and display the latest 
   prediction at the very bottom.

OPTION B: Run Locally on Your Computer
1. Clone the repository:
   git clone https://github.com/VhalahallaZ/financial-data-analysis.git
   cd financial-data-analysis

2. Create and activate a Python virtual environment:
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

3. Install the required packages:
   pip install -r requirements.txt

4. Run the scripts:
   - python main.py                 (Runs the Binary Allocation system)
   - python main_continuous.py      (Runs the Continuous Allocation system)
   - python plot_master_comparison.py (Generates master plots comparing all 6 strategies)
   - python eda_analysis.py         (Generates Exploratory Data Analysis figures)

All charts will be saved automatically into the "plots/" and "plots_continuous/" folders.

--------------------------------------------------------------------------------
5. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

- Dynamic asset allocation raw data.csv  : Combined raw daily and monthly dataset.
- requirements.txt                       : List of Python package versions needed.
- README.txt                             : This plain text documentation file.
- main.py                                : Script for Binary switching backtest.
- main_continuous.py                     : Script for Continuous allocation backtest.
- plot_master_comparison.py              : Script that produces master return/drawdown charts.
- eda_analysis.py                        : Script for EDA charts.
- master_performance_summary.csv         : Final summary metrics table for all 6 strategies.
- gini_feature_importance.csv            : Importance ranking of the 13 ML features.
- notebooks/
    regime_analysis.ipynb                : Complete Google Colab notebook.
- src/
    data_loader.py                       : Downloads data, stitches AGG/VBMFX, adds 1M lag.
    models.py                            : Markov Switching and Random Forest models.
    backtester.py                        : Rebalancing simulator with 10 bps fee and drift.
    visualizer.py                        : Charting functions.
- plots/                                 : Output folder for generated charts.
================================================================================
