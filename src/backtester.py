import pandas as pd
import numpy as np

def run_backtest(df, signal_series, risk_on_asset='SPY', risk_off_asset='TLT', 
                 allocation_type='binary', threshold=0.5, transaction_cost=0.001):
    """
    Backtests a dynamic asset allocation strategy using the provided regime probabilities.
    
    Parameters:
    - df: DataFrame containing returns/prices for risk_on_asset and risk_off_asset
    - signal_series: Series containing the probability of the Risk-Off regime [0, 1]
    - risk_on_asset: Column name for equity asset (e.g., 'SPY')
    - risk_off_asset: Column name for bond asset (e.g., 'TLT' or 'AGG')
    - allocation_type: 'binary' (100% allocation shift) or 'continuous' (linear blending)
    - threshold: Signal cutoff for binary switching
    - transaction_cost: Transaction cost rate (default 0.1% or 10 bps per trade)
    """
    # Calculate monthly asset returns
    asset_returns = df[[risk_on_asset, risk_off_asset]].pct_change().fillna(0)
    
    # Generate weights
    weights = pd.DataFrame(index=df.index, columns=[risk_on_asset, risk_off_asset], dtype=float)
    
    if allocation_type == 'binary':
        # 1 if Risk-Off probability > threshold, meaning we hold Bond asset
        weights[risk_off_asset] = np.where(signal_series > threshold, 1.0, 0.0)
        weights[risk_on_asset] = 1.0 - weights[risk_off_asset]
    elif allocation_type == 'continuous':
        # Linear blending: more probability of risk-off = more bonds
        weights[risk_off_asset] = signal_series.clip(0, 1)
        weights[risk_on_asset] = 1.0 - weights[risk_off_asset]
    else:
        raise ValueError("allocation_type must be 'binary' or 'continuous'")
        
    # Shift weights by 1 month to prevent look-ahead bias (rebalance at month-end, earn returns next month)
    weights = weights.shift(1).fillna(0.5) # Start with 50/50 for the first month
    
    # Backtest simulation loop to account for transaction costs accurately
    portfolio_returns = []
    current_weights = np.array([0.5, 0.5]) # starting wealth allocation
    
    for date, row in asset_returns.iterrows():
        target_weights = weights.loc[date].values
        returns = row.values
        
        # 1. Calculate portfolio return before rebalancing
        portfolio_return_no_cost = np.sum(current_weights * returns)
        
        # 2. Calculate end-of-period weights before rebalancing
        end_weights = current_weights * (1.0 + returns)
        sum_end_weights = np.sum(end_weights)
        if sum_end_weights > 0:
            end_weights_normalized = end_weights / sum_end_weights
        else:
            end_weights_normalized = current_weights
            
        # 3. Calculate turnover and transaction costs
        turnover = np.sum(np.abs(target_weights - end_weights_normalized))
        tc = turnover * transaction_cost
        
        # 4. Final net portfolio return
        net_return = portfolio_return_no_cost - tc
        portfolio_returns.append(net_return)
        
        # 5. Set weights for the next period
        current_weights = target_weights
        
    portfolio_returns_series = pd.Series(portfolio_returns, index=df.index)
    return portfolio_returns_series, weights

def calculate_metrics(returns_series):
    """
    Computes annualized return, annualized volatility, Sharpe ratio, and Max Drawdown.
    Assuming monthly returns data.
    """
    # Annualized Return
    n_months = len(returns_series)
    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (12 / n_months) - 1
    
    # Annualized Volatility
    annualized_vol = returns_series.std() * np.sqrt(12)
    
    # Sharpe Ratio (assuming risk free rate = 0)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Max Drawdown
    cum_wealth = (1 + returns_series).cumprod()
    running_max = cum_wealth.cummax()
    drawdowns = (cum_wealth - running_max) / running_max
    max_dd = drawdowns.min()
    
    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Vol": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_dd
    }

def run_benchmark_60_40(df, risk_on_asset='SPY', risk_off_asset='TLT', transaction_cost=0.001):
    """
    Simulates a monthly rebalanced 60/40 benchmark.
    """
    # Fixed target weights
    target_weights = np.array([0.6, 0.4])
    asset_returns = df[[risk_on_asset, risk_off_asset]].pct_change().fillna(0)
    
    portfolio_returns = []
    current_weights = np.array([0.6, 0.4])
    
    for date, row in asset_returns.iterrows():
        returns = row.values
        portfolio_return_no_cost = np.sum(current_weights * returns)
        
        end_weights = current_weights * (1.0 + returns)
        sum_end_weights = np.sum(end_weights)
        if sum_end_weights > 0:
            end_weights_normalized = end_weights / sum_end_weights
        else:
            end_weights_normalized = current_weights
            
        # Rebalance back to 60/40
        turnover = np.sum(np.abs(target_weights - end_weights_normalized))
        tc = turnover * transaction_cost
        
        net_return = portfolio_return_no_cost - tc
        portfolio_returns.append(net_return)
        
        current_weights = target_weights
        
    return pd.Series(portfolio_returns, index=df.index)
