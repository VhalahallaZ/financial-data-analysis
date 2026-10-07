import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def fit_markov_switching(df, features_for_mrs=['SPY_3M_Ret', 'SPY_Vol_3M']):
    """
    Fits a Markov Regime Switching model on SPY returns and volatility.
    Returns the smoothed probability of the high-volatility/low-return (Risk-Off) regime.
    """
    print("Fitting Markov Regime Switching Model...")
    
    # We use a 2-regime model on SPY returns and Volatility
    # Let's fit on the series
    mrs_data = df[features_for_mrs].dropna()
    
    # We can model the variance and mean switching
    # For statsmodels MarkovRegression, we pass endog (e.g., SPY_3M_Ret) and exog (e.g., SPY_Vol_3M)
    model = MarkovRegression(
        endog=mrs_data['SPY_3M_Ret'],
        k_regimes=2,
        exog=mrs_data[['SPY_Vol_3M']],
        switching_variance=True
    )
    
    res = model.fit(disp=False)
    
    # Identify which regime is "Risk-Off".
    # Usually, the regime with the lower mean return or higher volatility is Risk-Off.
    # res.params contains the estimated coefficients.
    # Let's look at the mean of each regime.
    regime_means = res.params.filter(like='const')
    
    # If const[0] > const[1], then Regime 1 is the lower return (Risk-Off) regime
    if len(regime_means) >= 2:
        r0_mean = regime_means.iloc[0]
        r1_mean = regime_means.iloc[1]
        risk_off_regime_index = 1 if r0_mean > r1_mean else 0
    else:
        risk_off_regime_index = 1
        
    # Get the smoothed probability of being in the risk-off regime
    prob_risk_off = res.smoothed_marginal_probabilities[risk_off_regime_index]
    
    # Align back to the original dataframe index
    prob_series = pd.Series(prob_risk_off, index=mrs_data.index).reindex(df.index).ffill().bfill()
    return prob_series

def run_expanding_window_ml(df, feature_cols, target_col='Regime_Target', min_train_months=120):
    """
    Simulates a realistic walk-forward (expanding window) out-of-sample prediction.
    For each month t (starting after min_train_months), we train the Random Forest
    on data up to t-1 and predict the probability of Risk-Off for month t.
    Prevents look-ahead bias completely.
    """
    print(f"Running expanding window ML predictions with {len(feature_cols)} features...")
    
    probabilities = []
    indices = []
    
    # Feature matrix X and target y
    X = df[feature_cols]
    y = df[target_col]
    
    n_samples = len(df)
    
    # Initialize with default/neutral probability for the startup period
    for i in range(min_train_months):
        probabilities.append(0.5) # neutral
        indices.append(df.index[i])
        
    # Walk-forward prediction
    for t in range(min_train_months, n_samples):
        # Train on all history up to t-1
        X_train = X.iloc[:t]
        y_train = y.iloc[:t]
        
        # Test on current sample (month t)
        X_test = X.iloc[[t]]
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict probability of class 1 (Risk-Off)
        prob = rf.predict_proba(X_test)[0][1]
        
        probabilities.append(prob)
        indices.append(df.index[t])
        
    prob_series = pd.Series(probabilities, index=indices)
    return prob_series
