import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from src.data_loader import prepare_dataset

# Style setup for institutional, publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#dee2e6'

def run_eda():
    print("=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Load prepared data terminating cleanly on September 30, 2026
    df = prepare_dataset(start_date="2000-01-01", end_date="2026-09-30")
    os.makedirs('plots_eda', exist_ok=True)
    
    # -------------------------------------------------------------
    # Investigation 1: Diversification Breakdown & Drawdown Asymmetry
    # -------------------------------------------------------------
    print("Generating Investigation 1 plot: Drawdown trajectories...")
    spy_cum = (1 + df['SPY'].pct_change().fillna(0)).cumprod()
    agg_cum = (1 + df['AGG'].pct_change().fillna(0)).cumprod()
    
    spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax() * 100
    agg_dd = (agg_cum - agg_cum.cummax()) / agg_cum.cummax() * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df.index, spy_dd, 0, color='#dc3545', alpha=0.3, label='S&P 500 (SPY) Drawdown')
    ax.plot(df.index, spy_dd, color='#dc3545', linewidth=1.5)
    
    ax.fill_between(df.index, agg_dd, 0, color='#0d6efd', alpha=0.3, label='Core Aggregate Bonds (AGG) Drawdown')
    ax.plot(df.index, agg_dd, color='#0d6efd', linewidth=1.5)
    
    # Highlight crises
    ax.axvspan(pd.to_datetime('2007-10-01'), pd.to_datetime('2009-03-01'), color='#6c757d', alpha=0.15, label='2008 Financial Crisis')
    ax.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-01'), color='#fd7e14', alpha=0.15, label='2020 COVID Shock')
    ax.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-10-01'), color='#ffc107', alpha=0.15, label='2022 Rate Hike Cycle')
    
    ax.set_title("Investigation 1: Asymmetric Tail Risk & Historical Peak-to-Trough Drawdowns", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylim(-55, 5)
    ax.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    plt.tight_layout()
    plt.savefig('plots_eda/eda_inv1_drawdown_breakdown.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Investigation 2: Macro Lead Signals (Yield Curve Inversion & UNRATE)
    # -------------------------------------------------------------
    print("Generating Investigation 2 plot: Macro early-warning signals...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    # Top plot: SPY price on left, 10Y-2Y Yield Spread on right
    ax1.plot(df.index, df['SPY'], color='#212529', linewidth=1.8, label='S&P 500 (SPY) Price')
    ax1.set_ylabel("SPY Price ($)", color='#212529', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#212529')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['10Y2Y_Yield_Spread'], color='#0d6efd', linewidth=1.2, label='10Y-2Y Treasury Spread (%)')
    ax1_twin.axhline(0, color='#dc3545', linestyle='--', linewidth=1, alpha=0.8, label='Inversion Threshold (0%)')
    ax1_twin.fill_between(df.index, df['10Y2Y_Yield_Spread'], 0, where=(df['10Y2Y_Yield_Spread'] < 0), color='#dc3545', alpha=0.25, label='Inverted Yield Curve (< 0%)')
    ax1_twin.set_ylabel("Yield Spread (%)", color='#0d6efd', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='#0d6efd')
    ax1_twin.set_ylim(-1.5, 3.5)
    ax1.set_title("Investigation 2A: Yield Curve Inversions as Leading Crisis Signals", fontsize=12, fontweight='bold')
    
    # Bottom plot: Trailing 3-month Unemployment change vs forward return
    ax2.plot(df.index, df['Unemployment_Change_3M'], color='#6f42c1', linewidth=1.5, label='3-Month Change in Unemployment Rate')
    ax2.axhline(0.3, color='#dc3545', linestyle=':', linewidth=1.2, label='Sahm-style Stress Threshold (+0.30%)')
    ax2.fill_between(df.index, df['Unemployment_Change_3M'], 0.3, where=(df['Unemployment_Change_3M'] > 0.3), color='#dc3545', alpha=0.25, label='Labor Deterioration Stress')
    ax2.set_ylabel("3M $\\Delta$ UNRATE (%)", color='#6f42c1', fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(-1.5, 4.0)
    ax2.set_title("Investigation 2B: Labor Market Deterioration & Macro Stress", fontsize=12, fontweight='bold')
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#ffffff')
    ax2.legend(loc='upper left', frameon=True, facecolor='#ffffff')
    
    plt.tight_layout()
    plt.savefig('plots_eda/eda_inv2_macro_signals.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Investigation 3: Volatility Clustering & Return Asymmetry
    # -------------------------------------------------------------
    print("Generating Investigation 3 plot: Return distribution & volatility clustering...")
    fig, (ax_dist, ax_vol) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 3A: Return Distribution vs Normal
    spy_monthly_ret = df['SPY'].pct_change() * 100
    valid_data = pd.DataFrame({'ret': spy_monthly_ret, 'vol': df['SPY_Vol_3M'] * 100}).dropna()
    
    mu, std = valid_data['ret'].mean(), valid_data['ret'].std()
    skew = stats.skew(valid_data['ret'])
    kurt = stats.kurtosis(valid_data['ret'])
    
    sns.histplot(valid_data['ret'], kde=True, color='#0d6efd', bins=35, stat='density', ax=ax_dist, label='Observed Returns')
    x = np.linspace(valid_data['ret'].min(), valid_data['ret'].max(), 100)
    p = stats.norm.pdf(x, mu, std)
    ax_dist.plot(x, p, 'r--', linewidth=1.8, label=f'Normal Fit ($\\mu$={mu:.2f}%, $\\sigma$={std:.2f}%)')
    ax_dist.set_title(f"Investigation 3A: Return Distribution (Skew: {skew:.2f}, Kurt: {kurt:.2f})", fontsize=11, fontweight='bold')
    ax_dist.set_xlabel("Monthly S&P 500 Return (%)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(frameon=True, facecolor='#ffffff')
    
    # 3B: Scatter of Monthly Return vs 3-Month Realized Volatility
    ax_vol.scatter(valid_data['vol'], valid_data['ret'], color='#198754', alpha=0.6, edgecolors='none', s=35)
    sns.regplot(x=valid_data['vol'], y=valid_data['ret'], ax=ax_vol, color='#dc3545', scatter=False, line_kws={'linewidth': 1.5})
    ax_vol.axhline(0, color='#6c757d', linestyle='--', linewidth=0.8)
    ax_vol.axvline(20, color='#ffc107', linestyle=':', linewidth=1.2, label='High-Vol Regime Threshold (20%)')
    ax_vol.set_title("Investigation 3B: Realized Volatility vs. Return Asymmetry", fontsize=11, fontweight='bold')
    ax_vol.set_xlabel("3-Month Annualized Realized Volatility (%)")
    ax_vol.set_ylabel("Monthly S&P 500 Return (%)")
    ax_vol.legend(frameon=True, facecolor='#ffffff')
    
    plt.tight_layout()
    plt.savefig('plots_eda/eda_inv3_volatility_clustering.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Investigation 4: Feature Correlation Heatmap
    # -------------------------------------------------------------
    print("Generating Feature Correlation Heatmap...")
    corr_cols = [
        'SPY_3M_Ret', 'AGG_3M_Ret', 'SPY_Vol_3M', 
        '10Y2Y_Yield_Spread', 'CPI_YoY', 'Fed_Funds_Rate', 
        'Unemployment_Rate', 'Yield_Spread_Change_3M', 'Unemployment_Change_3M'
    ]
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Exploratory Cross-Correlation Matrix of Engineered Features", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots_eda/eda_feature_correlation_matrix.png', dpi=300)
    plt.close()
    
    print("=" * 60)
    print("EDA PLOTS GENERATED SUCCESSFULLY in 'plots_eda/'")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
