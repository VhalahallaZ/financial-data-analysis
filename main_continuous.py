import os
import pandas as pd
import numpy as np
from src.data_loader import prepare_dataset
from src.models import fit_markov_switching, run_expanding_window_ml
from src.backtester import run_backtest, calculate_metrics, run_benchmark_60_40
from src.visualizer import plot_performance, plot_drawdowns, plot_regimes, plot_combined_regimes

def main():
    print("=" * 60)
    print("STARTING CONTINUOUS ALLOCATION REGIME SWITCHING SYSTEM (V2)")
    print("=" * 60)
    
    # 1. Load Data (from 2000)
    df = prepare_dataset(start_date="2000-01-01")
    
    # Define features for Machine Learning model
    features_ml = [
        'SPY_3M_Ret', 'SPY_6M_Ret', 'SPY_12M_Ret',
        'AGG_3M_Ret', 'AGG_12M_Ret', 'SPY_Vol_3M',
        '10Y2Y_Yield_Spread', 'CPI_YoY', 'Fed_Funds_Rate',
        'Unemployment_Rate', 'Yield_Spread_Change_3M',
        'Fed_Funds_Change_3M', 'Unemployment_Change_3M'
    ]
    
    # 2. Run Models to get probabilities
    mrs_prob = fit_markov_switching(df, features_for_mrs=['SPY_3M_Ret', 'SPY_Vol_3M'])
    ml_prob = run_expanding_window_ml(df, features_ml, target_col='Regime_Target', min_train_months=120)
    
    df['MRS_Prob'] = mrs_prob
    df['ML_Prob'] = ml_prob
    
    # 3. Run Backtests with CONTINUOUS Allocation
    # S&P 500 Buy & Hold returns
    spy_returns = df['SPY'].pct_change().fillna(0)
    
    # Benchmark: 60/40 SPY/AGG Portfolio
    benchmark_returns = run_benchmark_60_40(df, risk_on_asset='SPY', risk_off_asset='AGG', transaction_cost=0.001)
    
    # MRS Dynamic Strategy (Continuous Allocation)
    mrs_returns, mrs_weights = run_backtest(
        df, mrs_prob, risk_on_asset='SPY', risk_off_asset='AGG', 
        allocation_type='continuous', transaction_cost=0.001
    )
    
    # ML Dynamic Strategy (Continuous Allocation)
    ml_returns, ml_weights = run_backtest(
        df, ml_prob, risk_on_asset='SPY', risk_off_asset='AGG', 
        allocation_type='continuous', transaction_cost=0.001
    )
    
    # 4. Compile Returns & Drawdowns
    returns_df = pd.DataFrame({
        'S&P 500 Buy & Hold': spy_returns,
        'Benchmark (60/40)': benchmark_returns,
        'Markov Switching (MRS - Continuous)': mrs_returns,
        'Machine Learning (RF - Continuous)': ml_returns
    }, index=df.index)
    
    eval_start_date = df.index[120]
    eval_returns = returns_df[returns_df.index >= eval_start_date]
    eval_df = df[df.index >= eval_start_date]
    
    cum_returns = (1 + eval_returns).cumprod() - 1
    
    drawdowns = pd.DataFrame(index=eval_returns.index)
    for col in eval_returns.columns:
        cum_wealth = (1 + eval_returns[col]).cumprod()
        running_max = cum_wealth.cummax()
        drawdowns[col] = (cum_wealth - running_max) / running_max
        
    # 5. Calculate Metrics
    metrics_list = []
    for col in eval_returns.columns:
        metrics = calculate_metrics(eval_returns[col])
        metrics['Strategy'] = col
        metrics_list.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
    
    print("\n" + "=" * 80)
    print(f"V2 PERFORMANCE METRICS SUMMARY (Continuous Allocation, Out-of-Sample: {eval_start_date.strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")
    print("=" * 80)
    
    formatted_metrics = metrics_df.copy()
    formatted_metrics['Cumulative Return'] = (formatted_metrics['Cumulative Return'] * 100).round(2).astype(str) + '%'
    formatted_metrics['Annualized Return'] = (formatted_metrics['Annualized Return'] * 100).round(2).astype(str) + '%'
    formatted_metrics['Annualized Vol'] = (formatted_metrics['Annualized Vol'] * 100).round(2).astype(str) + '%'
    formatted_metrics['Sharpe Ratio'] = formatted_metrics['Sharpe Ratio'].round(3)
    formatted_metrics['Max Drawdown'] = (formatted_metrics['Max Drawdown'] * 100).round(2).astype(str) + '%'
    print(formatted_metrics)
    print("=" * 80 + "\n")
    
    # Save CSV
    metrics_df.to_csv("performance_metrics_continuous.csv")
    
    # 6. Save Plots in separate folder
    os.makedirs('plots_continuous', exist_ok=True)
    
    plot_performance(cum_returns, title="Dynamic Asset Allocation (Continuous) - Cumulative Returns Comparison", save_path="plots_continuous/performance_comparison.png")
    plot_drawdowns(drawdowns, title="Dynamic Asset Allocation (Continuous) - Drawdowns Comparison", save_path="plots_continuous/drawdown_comparison.png")
    plot_regimes(eval_df['SPY'], eval_df['MRS_Prob'], title="S&P 500 & Markov Regime-Switching Probabilities (Continuous)", save_path="plots_continuous/regime_switching_mrs.png")
    plot_regimes(eval_df['SPY'], eval_df['ML_Prob'], title="S&P 500 & ML Random Forest Risk-Off Probabilities (Continuous)", save_path="plots_continuous/regime_switching_ml.png")
    plot_combined_regimes(eval_df['SPY'], eval_df['MRS_Prob'], eval_df['ML_Prob'], title="S&P 500 & Combined Regime-Switching Probabilities (Continuous)", save_path="plots_continuous/regime_switching_combined.png")
    
    print("=" * 60)
    print("V2 SYSTEM RUN COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
