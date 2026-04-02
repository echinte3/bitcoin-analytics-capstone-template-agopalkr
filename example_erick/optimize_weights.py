import logging
import pandas as pd
import numpy as np
from functools import partial
from template.prelude_template import load_data, backtest_dynamic_dca

# Import your specific features and weights function
# Adjust the import path based on your actual file name
from example_erick.model_development_example_1 import precompute_features, compute_window_weights, load_snp_data

def custom_wrapper(df_window: pd.DataFrame, features_df: pd.DataFrame, custom_weights: dict) -> pd.Series:
    """A wrapper that injects custom weights into the window calculator."""
    if df_window.empty:
        return pd.Series(dtype=float)
    start_date = df_window.index.min()
    end_date = df_window.index.max()
    return compute_window_weights(features_df, start_date, end_date, end_date, weights=custom_weights)

def main():
    logging.basicConfig(level=logging.WARNING) # Suppress info logs so we only see results
    print("Loading data and precomputing features...")
    
    # 1. Load Data Once
    btc_df = load_data()
    features_df = precompute_features(btc_df)
    
    # Merge S&P data (like in your run_backtest)
    df_sp500 = load_snp_data()
    features_df = pd.merge(features_df, df_sp500, left_index=True, right_index=True, how='left')
    
    print("Starting optimization grid search...\n")
    best_win_rate = 0.0
    best_weights = {}

    # 2. Iterate through weight combinations
    # We step by 0.1 (10%) to keep it fast. 
    # MVRV is kept between 30% and 70% to maintain strategy logic.
    for mvrv in np.arange(0.3, 0.8, 0.1):
        for ma in np.arange(0.0, 0.4, 0.1):
            for fgi in np.arange(0.0, 0.4, 0.1):
                for snp in np.arange(0.0, 0.4, 0.1):
                    # Polymarket takes whatever is left over to equal 1.0
                    poly = 1.0 - (mvrv + ma + fgi + snp)
                    
                    # Ensure Polymarket is a valid positive weight and not too huge
                    if 0.0 <= poly <= 0.4:
                        current_weights = {
                            'mvrv': round(mvrv, 2),
                            'ma': round(ma, 2),
                            'fgi': round(fgi, 2),
                            'snp': round(snp, 2),
                            'poly': round(poly, 2)
                        }
                        
                        # Inject the specific weights into our wrapper
                        bound_wrapper = partial(
                            custom_wrapper, 
                            features_df=features_df, 
                            custom_weights=current_weights
                        )
                        
                        # Run JUST the backtest engine (No charts!)
                        df_spd, _ = backtest_dynamic_dca(
                            btc_df, 
                            bound_wrapper, 
                            features_df=features_df, 
                            strategy_label="Optimizer"
                        )
                        
                        # Calculate Win Rate
                        wins = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).sum()
                        win_rate = (wins / len(df_spd)) * 100
                        
                        print(f"Tested: {current_weights} | Win Rate: {win_rate:.2f}%")
                        
                        # Track the best performing weights
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_weights = current_weights

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best Win Rate: {best_win_rate:.2f}%")
    print(f"Best Weights:  {best_weights}")
    print("="*50)

if __name__ == "__main__":
    main()