import logging
import pandas as pd
import numpy as np
from functools import partial
import time

from template.prelude_template import load_data, backtest_dynamic_dca

# Adjust the import path based on your actual file name. 
# Note: load_snp_data is no longer needed here as it is handled inside precompute_features now.
from example_erick.model_development_example_1 import precompute_features, compute_window_weights

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
    
    # S&P 500, FGI, Polymarket, and RSI are all handled natively inside this function now!
    features_df = precompute_features(btc_df) 
    
    print("Starting optimization grid search...\n")
    best_win_rate = 0.0
    best_weights = {}
    
    # List to store our results for CSV export
    results_log = []

    # 2. Iterate through weight combinations
    # With 6 variables, we need to constrain the ranges to avoid a combinatorial explosion.
    # MVRV is kept primary [40% - 60%]. Secondary signals are capped at 20%.
    for mvrv in [0.4, 0.5, 0.6]:
        for ma in [0.0, 0.1, 0.2]:
            for rsi in [0.0, 0.1, 0.2]:
                for fgi in [0.0, 0.1, 0.2]:
                    for snp in [0.0, 0.1, 0.2]:
                        
                        # Polymarket takes whatever is left over to equal 1.0
                        poly = 1.0 - (mvrv + ma + rsi + fgi + snp)
                        
                        # Ensure Polymarket is a valid positive weight and not absurdly high
                        # Since we are dealing with floating point math, round it first
                        poly = round(poly, 2)
                        
                        if 0.0 <= poly <= 0.3:
                            current_weights = {
                                'mvrv': round(mvrv, 2),
                                'ma': round(ma, 2),
                                'rsi': round(rsi, 2),
                                'fgi': round(fgi, 2),
                                'snp': round(snp, 2),
                                'poly': poly
                            }
                            
                            # Inject the specific weights into our wrapper
                            bound_wrapper = partial(
                                custom_wrapper, 
                                features_df=features_df, 
                                custom_weights=current_weights
                            )
                            
                            # Run JUST the backtest engine
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
                            
                            # Log to our results list
                            results_log.append({
                                'mvrv_weight': current_weights['mvrv'],
                                'ma_weight': current_weights['ma'],
                                'rsi_weight': current_weights['rsi'],
                                'fgi_weight': current_weights['fgi'],
                                'snp_weight': current_weights['snp'],
                                'poly_weight': current_weights['poly'],
                                'win_rate_percent': round(win_rate, 2)
                            })
                            
                            # Track the best performing weights
                            if win_rate > best_win_rate:
                                best_win_rate = win_rate
                                best_weights = current_weights

    # Save results to CSV
    csv_filename = "optimization_results.csv"
    results_df = pd.DataFrame(results_log)
    # Sort by highest win rate at the top for easy viewing
    results_df = results_df.sort_values(by="win_rate_percent", ascending=False)
    results_df.to_csv(csv_filename, index=False)

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best Win Rate: {best_win_rate:.2f}%")
    print(f"Best Weights:  {best_weights}")
    print(f"Results saved to: {csv_filename}")
    print("="*50)

if __name__ == "__main__":
    main()