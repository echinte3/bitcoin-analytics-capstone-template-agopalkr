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

# Globals to hold the state across iterations
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
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logging.info("Starting Extreme Sentiment Grid Search (FGI & Poly 1.0 -> 10.0)")
    
    # 1. Load Data
    btc_df = load_data()
    
    # 2. Train LSTM exactly once
    logging.info("Training LSTM base model...")
    _LSTM_MODEL = train_lstm_model(btc_df)
    
    # 3. Precompute Features
    logging.info("Precomputing market features...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 4. Setup Grid Search Parameters (Extreme Bounds)
    # Testing 1.0 through 10.0
    weight_options = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    signals_to_vary = ['fgi', 'poly']
    
    # Generate the 100 combinations (10^2)
    combinations = list(itertools.product(weight_options, repeat=len(signals_to_vary)))
    
    best_score = -float('inf')
    best_weights = None
    best_metrics = {}
    
    # =========================================================================
    # CSV Setup (Overwriting previous results)
    # =========================================================================
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / "optimization_results_2.csv"
    
    # Initialize CSV with headers
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'MVRV', 'FGI', 'Poly', 'SNP', 'MA', 'Win_Rate', 'Exp_Decay', 'Final_Score'])

    logging.info(f"Total combinations to evaluate: {len(combinations)}")
    logging.info(f"Real-time results logging to: {csv_path}")
    logging.info("--------------------------------------------------")
    
    logger = logging.getLogger()
    
    for idx, combo in enumerate(combinations):
        # Map current combination and lock MVRV=1.0, SNP=0.0, MA=0.0
        _CURRENT_WEIGHTS = {
            'mvrv': 1.0,       # Locked
            'fgi': combo[0],   # Varied
            'poly': combo[1],  # Varied
            'snp': 0.0,        # Locked
            'ma': 0.0          # Locked
        }
        
        # Suppress internal backtest logs
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            # Run the backtest
            df_spd, exp_decay_percentile = backtest_dynamic_dca(
                btc_df,
                compute_weights_wrapper,
                features_df=_FEATURES_DF,
                strategy_label="Extreme_Optimization_Run"
            )
            
            # Calculate final score
            win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
            score = 0.5 * win_rate + 0.5 * exp_decay_percentile
            
            # Log the iteration
            logger.setLevel(logging.INFO)
            logging.info(f"[{idx+1:03d}/{len(combinations)}] W: {combo} | WinRate: {win_rate:05.2f}% | ExpDecay: {exp_decay_percentile:05.2f}% | Score: {score:05.2f}%")
            
            # Write to CSV
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

            # Track high score
            if score > best_score:
                best_score = score
                best_weights = _CURRENT_WEIGHTS.copy()
                best_metrics = {
                    "score": score,
                    "win_rate": win_rate,
                    "exp_decay_percentile": exp_decay_percentile
                }
                logging.info(f"    --> 🚀 NEW BEST SCORE: {score:.2f}%")
                
        except Exception as e:
            logger.setLevel(logging.INFO)
            logging.error(f"Error evaluating combo {_CURRENT_WEIGHTS}: {e}")
            
        finally:
            logger.setLevel(old_level)

    # =========================================================================
    # Final Output
    # =========================================================================
    logger.setLevel(logging.INFO)
    logging.info("==================================================")
    logging.info("Extreme Optimization Complete!")
    logging.info(f"Absolute Best Score: {best_score:.2f}%")
    logging.info(f"Optimal Weights: {best_weights}")
    logging.info(f"Metrics: Win Rate = {best_metrics['win_rate']:.2f}%, Exp Decay = {best_metrics['exp_decay_percentile']:.2f}%")
    logging.info("==================================================")
    
    # Save optimal to JSON
    output_path = output_dir / "optimal_weights_2.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_score": best_score,
            "best_weights": best_weights,
            "metrics": best_metrics
        }, f, indent=4)
        
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()