import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Apply a clean design style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#dee2e6'

def plot_performance(results_df, title="Strategy Comparison - Cumulative Return", save_path="performance_comparison.png"):
    """
    Plots cumulative returns of the strategies.
    results_df should have columns for each strategy's cumulative returns.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Modern, harmonious colors
    colors = {
        'S&P 500 Buy & Hold': '#343a40',       # Charcoal
        'Benchmark (60/40)': '#6c757d',        # Grey
        'Markov Switching (MRS)': '#0d6efd',  # Blue
        'Markov Switching (MRS - Continuous)': '#0d6efd',
        'Machine Learning (RF)': '#198754',    # Green
        'Machine Learning (RF - Continuous)': '#198754'
    }
    
    for col in results_df.columns:
        color = colors.get(col, None)
        ax.plot(results_df.index, results_df[col] * 100, label=col, color=color, linewidth=2)
        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11, labelpad=10)
    ax.set_xlabel("Date", fontsize=11, labelpad=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, color='#e9ecef')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Performance plot saved to {save_path}")

def plot_drawdowns(drawdown_df, title="Strategy Comparison - Drawdowns", save_path="drawdown_comparison.png"):
    """
    Plots drawdown profile of the strategies.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = {
        'S&P 500 Buy & Hold': '#343a40',
        'Benchmark (60/40)': '#6c757d',
        'Markov Switching (MRS)': '#0d6efd',
        'Markov Switching (MRS - Continuous)': '#0d6efd',
        'Machine Learning (RF)': '#198754',
        'Machine Learning (RF - Continuous)': '#198754'
    }
    
    for col in drawdown_df.columns:
        color = colors.get(col, None)
        ax.fill_between(drawdown_df.index, drawdown_df[col] * 100, 0, label=col, color=color, alpha=0.15)
        ax.plot(drawdown_df.index, drawdown_df[col] * 100, color=color, linewidth=1.5)
        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Drawdown (%)", fontsize=11, labelpad=10)
    ax.set_xlabel("Date", fontsize=11, labelpad=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, color='#e9ecef')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Drawdown plot saved to {save_path}")

def plot_regimes(spy_series, prob_series, title="Regime Probability vs S&P 500", save_path="regime_switching.png"):
    """
    Plots the price of SPY with shaded areas representing periods of high Risk-Off probability.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary axis for SPY
    color_spy = '#343a40'
    ax1.plot(spy_series.index, spy_series.values, color=color_spy, linewidth=1.5, label='SPY Price')
    ax1.set_ylabel('S&P 500 (SPY) Price ($)', color=color_spy, fontsize=11, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color_spy)
    
    # Secondary axis for regime probability
    ax2 = ax1.twinx()
    color_prob = '#dc3545' # Red
    ax2.plot(prob_series.index, prob_series.values, color=color_prob, linewidth=1, alpha=0.5, label='Risk-Off Probability')
    ax2.fill_between(prob_series.index, 0, prob_series.values, where=(prob_series.values > 0.5), 
                     color=color_prob, alpha=0.2, label='Risk-Off State (Prob > 50%)')
    ax2.set_ylabel('Risk-Off Probability', color=color_prob, fontsize=11, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=color_prob)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4, color='#dee2e6')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Regime plot saved to {save_path}")

def plot_combined_regimes(spy_series, mrs_prob, ml_prob, title="Regime Probability Comparison vs S&P 500", save_path="regime_switching_combined.png"):
    """
    Plots the price of SPY (left axis) with both MRS and ML probabilities on the right axis in different colors.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary axis for SPY
    color_spy = '#343a40' # Charcoal
    ax1.plot(spy_series.index, spy_series.values, color=color_spy, linewidth=1.5, label='SPY Price')
    ax1.set_ylabel('S&P 500 (SPY) Price ($)', color=color_spy, fontsize=11, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color_spy)
    
    # Secondary axis for probabilities
    ax2 = ax1.twinx()
    
    # MRS probability in blue with shading
    color_mrs = '#0d6efd' 
    ax2.plot(mrs_prob.index, mrs_prob.values, color=color_mrs, linewidth=1.5, alpha=0.7, label='MRS Risk-Off Prob')
    ax2.fill_between(mrs_prob.index, 0, mrs_prob.values, where=(mrs_prob.values > 0.5), 
                     color=color_mrs, alpha=0.15, label='MRS Risk-Off State')
    
    # ML probability in green with shading
    color_ml = '#198754'
    ax2.plot(ml_prob.index, ml_prob.values, color=color_ml, linewidth=1.5, alpha=0.7, label='ML Risk-Off Prob')
    ax2.fill_between(ml_prob.index, 0, ml_prob.values, where=(ml_prob.values > 0.5), 
                     color=color_ml, alpha=0.15, label='ML Risk-Off State')
    
    # Draw a 50% threshold line
    ax2.axhline(0.5, color='#dc3545', linestyle='--', alpha=0.5, label='50% Threshold')
    
    ax2.set_ylabel('Risk-Off Probability', color='#212529', fontsize=11, labelpad=10)
    ax2.tick_params(axis='y', labelcolor='#212529')
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4, color='#dee2e6')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Combined regime plot saved to {save_path}")
