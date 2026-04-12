import os
import csv
import json
import itertools
import logging
import pandas as pd
from pathlib import Path

# Import template components
from template.prelude_template import load_data, backtest_dynamic_dca

# Import your model logic and the training function
from example_LSTM_merged_2.model_development_example_2 import precompute_features, compute_window_weights
from example_LSTM_merged_2.run_backtest import train_lstm_model

# Globals to hold the state across 1024 iterations
_FEATURES_DF = None
_LSTM_MODEL = None
_CURRENT_WEIGHTS = {}

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """
    Wrapper that injects the currently selected grid search weights 
    into the model's weight computation function.
    """
    global _FEATURES_DF
    global _LSTM_MODEL
    global _CURRENT_WEIGHTS
    
    if _FEATURES_DF is None or _LSTM_MODEL is None:
        raise ValueError("Features or Model not initialized.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    
    return compute_window_weights(
        _lstm_model=_LSTM_MODEL, 
        features_df=_FEATURES_DF, 
        start_date=start_date, 
        end_date=end_date, 
        current_date=current_date, 
        weights=_CURRENT_WEIGHTS
    )

def main():
    global _FEATURES_DF, _LSTM_MODEL, _CURRENT_WEIGHTS
    
    # Setup base logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s", # Simplified format for cleaner scrolling
        datefmt="%H:%M:%S",
    )
    
    logging.info("Starting Grid Search Optimization for Signal Weights")
    
    # 1. Load Data
    btc_df = load_data()
    
    # 2. Train LSTM exactly once
    logging.info("Training LSTM base model...")
    _LSTM_MODEL = train_lstm_model(btc_df)
    
    # 3. Precompute Features
    logging.info("Precomputing market features...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 4. Setup Grid Search Parameters
    weight_options = [0.0, 0.5, 1.0, 2.0]
    signals = ['mvrv', 'fgi', 'poly', 'snp', 'ma']
    
    # Generate all 1024 combinations
    combinations = list(itertools.product(weight_options, repeat=len(signals)))
    
    best_score = -float('inf')
    best_weights = None
    best_metrics = {}
    
    # =========================================================================
    # CSV Setup
    # =========================================================================
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / "optimization_results.csv"
    
    # Initialize CSV with headers
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'MVRV', 'FGI', 'Poly', 'SNP', 'MA', 'Win_Rate', 'Exp_Decay', 'Final_Score'])

    logging.info(f"Total combinations to evaluate: {len(combinations)}")
    logging.info(f"Real-time results logging to: {csv_path}")
    logging.info("--------------------------------------------------")
    
    # Access the root logger to suppress output from the backtest engine during the loop
    logger = logging.getLogger()
    
    for idx, combo in enumerate(combinations):
        # Map current combination to the signal dictionary
        _CURRENT_WEIGHTS = dict(zip(signals, combo))
        
        # Suppress standard INFO logs to avoid printing the internal backtest summaries
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            # Run the backtest for this specific weight combination
            df_spd, exp_decay_percentile = backtest_dynamic_dca(
                btc_df,
                compute_weights_wrapper,
                features_df=_FEATURES_DF,
                strategy_label="Optimization_Run"
            )
            
            # Replicate the template's Final Score calculation
            win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
            score = 0.5 * win_rate + 0.5 * exp_decay_percentile
            
            # Re-enable logging momentarily to print the iteration result
            logger.setLevel(logging.INFO)
            logging.info(f"[{idx+1:04d}/{len(combinations)}] W: {combo} | WinRate: {win_rate:05.2f}% | ExpDecay: {exp_decay_percentile:05.2f}% | Score: {score:05.2f}%")
            
            # =========================================================================
            # Write to CSV immediately
            # =========================================================================
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx + 1,
                    _CURRENT_WEIGHTS['mvrv'],
                    _CURRENT_WEIGHTS['fgi'],
                    _CURRENT_WEIGHTS['poly'],
                    _CURRENT_WEIGHTS['snp'],
                    _CURRENT_WEIGHTS['ma'],
                    round(win_rate, 4),
                    round(exp_decay_percentile, 4),
                    round(score, 4)
                ])

            # Check for new high score
            if score > best_score:
                best_score = score
                best_weights = _CURRENT_WEIGHTS.copy()
                best_metrics = {
                    "score": score,
                    "win_rate": win_rate,
                    "exp_decay_percentile": exp_decay_percentile
                }
                logging.info(f"    --> 🏆 NEW BEST SCORE: {score:.2f}%")
                
        except Exception as e:
            logger.setLevel(logging.INFO)
            logging.error(f"Error evaluating combo {_CURRENT_WEIGHTS}: {e}")
            
        finally:
            # Restore logging state
            logger.setLevel(old_level)

    # =========================================================================
    # Output Final Results
    # =========================================================================
    logger.setLevel(logging.INFO) # Ensure logger is back on for final output
    logging.info("==================================================")
    logging.info("Optimization Complete!")
    logging.info(f"Absolute Best Score: {best_score:.2f}%")
    logging.info(f"Optimal Weights: {best_weights}")
    logging.info(f"Metrics: Win Rate = {best_metrics['win_rate']:.2f}%, Exp Decay = {best_metrics['exp_decay_percentile']:.2f}%")
    logging.info("==================================================")
    
    # Save the optimal combination to JSON for easy reference
    output_path = output_dir / "optimal_weights.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_score": best_score,
            "best_weights": best_weights,
            "metrics": best_metrics
        }, f, indent=4)
        
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()