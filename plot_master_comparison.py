import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_loader import prepare_dataset
from src.models import fit_markov_switching, run_expanding_window_ml
from src.backtester import run_backtest, run_benchmark_60_40, calculate_metrics

def main():
    print("=" * 75)
    print("GENERATING MASTER ALL-IN-ONE CUMULATIVE RETURNS & DRAWDOWN COMPARISON")
    print("Strategies: S&P 500 Buy & Hold, Benchmark 60/40, MRS Binary, MRS Continuous, ML Binary, ML Continuous")
    print("=" * 75)
    
    # 1. Load prepared data terminating cleanly on September 30, 2026
    df = prepare_dataset(start_date="2000-01-01", end_date="2026-09-30")
    
    # 2. Features for models
    features_ml = [
        'SPY_3M_Ret', 'SPY_6M_Ret', 'SPY_12M_Ret',
        'AGG_3M_Ret', 'AGG_12M_Ret', 'SPY_Vol_3M',
        '10Y2Y_Yield_Spread', 'CPI_YoY', 'Fed_Funds_Rate',
        'Unemployment_Rate', 'Yield_Spread_Change_3M',
        'Fed_Funds_Change_3M', 'Unemployment_Change_3M'
    ]
    
    # 3. Model Predictions
    mrs_prob = fit_markov_switching(df, features_for_mrs=['SPY_3M_Ret', 'SPY_Vol_3M'])
    ml_prob = run_expanding_window_ml(df, features_ml, target_col='Regime_Target', min_train_months=120)
    
    # 4. Backtest all strategies
    spy_returns = df['SPY'].pct_change().fillna(0)
    bench_returns = run_benchmark_60_40(df, risk_on_asset='SPY', risk_off_asset='AGG', transaction_cost=0.001)
    
    # MRS Binary & Continuous
    mrs_bin_ret, _ = run_backtest(df, mrs_prob, risk_on_asset='SPY', risk_off_asset='AGG', allocation_type='binary', threshold=0.5, transaction_cost=0.001)
    mrs_con_ret, _ = run_backtest(df, mrs_prob, risk_on_asset='SPY', risk_off_asset='AGG', allocation_type='continuous', transaction_cost=0.001)
    
    # ML Binary & Continuous
    ml_bin_ret, _ = run_backtest(df, ml_prob, risk_on_asset='SPY', risk_off_asset='AGG', allocation_type='binary', threshold=0.5, transaction_cost=0.001)
    ml_con_ret, _ = run_backtest(df, ml_prob, risk_on_asset='SPY', risk_off_asset='AGG', allocation_type='continuous', transaction_cost=0.001)
    
    # 5. Compile all 6 series
    all_returns = pd.DataFrame({
        'S&P 500 Buy & Hold': spy_returns,
        'Benchmark (60/40)': bench_returns,
        'Markov Switching (MRS - Binary)': mrs_bin_ret,
        'Markov Switching (MRS - Continuous)': mrs_con_ret,
        'Machine Learning (RF - Binary)': ml_bin_ret,
        'Machine Learning (RF - Continuous)': ml_con_ret
    }, index=df.index)
    
    # Out-of-sample filter (Month 120 onward: Jan 2010 to Sep 2026)
    eval_start_date = df.index[120]
    eval_returns = all_returns[all_returns.index >= eval_start_date]
    
    cum_returns = (1 + eval_returns).cumprod() - 1
    
    # Drawdowns
    drawdowns = pd.DataFrame(index=eval_returns.index)
    for col in eval_returns.columns:
        cum_wealth = (1 + eval_returns[col]).cumprod()
        running_max = cum_wealth.cummax()
        drawdowns[col] = (cum_wealth - running_max) / running_max
        
    # 6. Plotting Master Cumulative Returns Chart
    os.makedirs('plots', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # Distinct, professional color palette
    styles = {
        'S&P 500 Buy & Hold': {'color': '#212529', 'linestyle': '-', 'linewidth': 2.2, 'alpha': 0.95},
        'Benchmark (60/40)': {'color': '#6c757d', 'linestyle': '--', 'linewidth': 1.8, 'alpha': 0.85},
        'Markov Switching (MRS - Binary)': {'color': '#fd7e14', 'linestyle': ':', 'linewidth': 1.6, 'alpha': 0.9},
        'Markov Switching (MRS - Continuous)': {'color': '#e83e8c', 'linestyle': '-.', 'linewidth': 1.6, 'alpha': 0.9},
        'Machine Learning (RF - Binary)': {'color': '#198754', 'linestyle': '-', 'linewidth': 2.3, 'alpha': 0.95},
        'Machine Learning (RF - Continuous)': {'color': '#0d6efd', 'linestyle': '-', 'linewidth': 1.8, 'alpha': 0.9}
    }
    
    for col in cum_returns.columns:
        cfg = styles.get(col, {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 1.0})
        pct_series = cum_returns[col] * 100
        ax.plot(cum_returns.index, pct_series, label=f"{col} ({pct_series.iloc[-1]:.1f}%)", **cfg)
        
    # Crisis Shading
    ax.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-01'), color='#fd7e14', alpha=0.15, label='2020 COVID Shock')
    ax.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-10-01'), color='#ffc107', alpha=0.15, label='2022 Inflation & Rate Hike Cycle')
    
    ax.set_title("Master Out-of-Sample Cumulative Returns Comparison (2010 - 2026)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Cumulative Total Return (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', fontsize=9.5)
    
    plt.tight_layout()
    out_path = 'plots/master_cumulative_returns_comparison.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Master returns plot saved successfully to {out_path}")
    
    # 7. Plotting Master Underwater Drawdown Comparison Chart
    fig_dd, ax_dd = plt.subplots(figsize=(13, 6))
    
    for col in drawdowns.columns:
        cfg = styles.get(col, {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 1.0})
        dd_pct = drawdowns[col] * 100
        min_dd = dd_pct.min()
        ax_dd.plot(drawdowns.index, dd_pct, label=f"{col} (Max: {min_dd:.2f}%)", **cfg)
        
    # Crisis Shading
    ax_dd.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-01'), color='#fd7e14', alpha=0.15, label='2020 COVID Shock')
    ax_dd.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-10-01'), color='#ffc107', alpha=0.15, label='2022 Inflation & Rate Hike Cycle')
    
    ax_dd.set_title("Master Out-of-Sample Underwater Drawdown Comparison (2010 - 2026)", fontsize=13, fontweight='bold', pad=12)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
    ax_dd.set_xlabel("Date", fontsize=11)
    ax_dd.set_ylim(-27, 2)
    ax_dd.grid(True, linestyle='--', alpha=0.5)
    ax_dd.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', fontsize=9.0, ncol=2)
    
    plt.tight_layout()
    out_dd_path = 'plots/master_drawdown_comparison.png'
    plt.savefig(out_dd_path, dpi=300)
    plt.close()
    print(f"Master drawdown plot saved successfully to {out_dd_path}")
    
    # 7. Print Final Master Summary Table
    metrics_list = []
    for col in eval_returns.columns:
        m = calculate_metrics(eval_returns[col])
        m['Strategy'] = col
        metrics_list.append(m)
        
    master_df = pd.DataFrame(metrics_list).set_index('Strategy')
    master_df.to_csv("master_performance_summary.csv")
    
    print("\n" + "=" * 90)
    print("MASTER PERFORMANCE SUMMARY (OUT-OF-SAMPLE: 2010-01 TO 2026-09)")
    print("=" * 90)
    fmt = master_df.copy()
    fmt['Cumulative Return'] = (fmt['Cumulative Return'] * 100).round(2).astype(str) + '%'
    fmt['Annualized Return'] = (fmt['Annualized Return'] * 100).round(2).astype(str) + '%'
    fmt['Annualized Vol'] = (fmt['Annualized Vol'] * 100).round(2).astype(str) + '%'
    fmt['Sharpe Ratio'] = fmt['Sharpe Ratio'].round(3)
    fmt['Max Drawdown'] = (fmt['Max Drawdown'] * 100).round(2).astype(str) + '%'
    print(fmt)
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
