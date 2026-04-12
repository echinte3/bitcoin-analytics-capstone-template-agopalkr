import itertools
import logging
import pandas as pd
from pathlib import Path

# Import core backtest engine
from template.prelude_template import load_data, backtest_dynamic_dca

# Import model specifics
from example_LSTM_merged.model_development_example_2 import precompute_features, compute_window_weights
from example_LSTM_merged.run_backtest import lstm

def main():
    # 1. Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Define CSV path early so we can write to it in the loop
    csv_path = output_dir / "optimization_results.csv"
    
    logging.info("Loading BTC Data...")
    btc_df = load_data()
    
    # 2. Train LSTM ONCE to save time
    logging.info("Training LSTM Timing Engine (This happens only once)...")
    trained_lstm = lstm(btc_df)
    
    # 3. Precompute Features ONCE
    logging.info("Precomputing features...")
    features_df = precompute_features(btc_df)
    
    # 4. Define Search Space
    weight_values = [0.0, 0.5, 1.0, 2.0]
    weight_keys = ['mvrv', 'ma', 'fgi', 'poly', 'snp']
    combinations = list(itertools.product(weight_values, repeat=len(weight_keys)))
    total_iters = len(combinations)
    
    logging.info(f"Starting Grid Search over {total_iters} combinations...")
    
    best_score = -1
    all_results = []
    
    # 5. Grid Search Loop
    for idx, combo in enumerate(combinations, 1):
        current_weights = dict(zip(weight_keys, combo))
        
        # Create a closure wrapper for this specific combination
        def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
            if df_window.empty:
                return pd.Series(dtype=float)
            
            start_date = df_window.index.min()
            end_date = df_window.index.max()
            current_date = end_date
            
            return compute_window_weights(
                trained_lstm, 
                features_df, 
                start_date, 
                end_date, 
                current_date, 
                weights=current_weights
            )
            
        try:
            # Run lightweight core backtest (skips all chart plotting/saving)
            df_spd, exp_decay_percentile = backtest_dynamic_dca(
                btc_df,
                compute_weights_wrapper,
                features_df=features_df,
                strategy_label="Optimizer"
            )
            
            # Calculate metrics
            win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
            score = 0.5 * win_rate + 0.5 * exp_decay_percentile
            
            # Store results
            result_row = {
                'score': score,
                'win_rate': win_rate,
                'exp_decay_percentile': exp_decay_percentile,
                **current_weights
            }
            all_results.append(result_row)
            
            # --- NEW: Detailed logging per iteration ---
            weight_str = f"[{combo[0]:.1f}, {combo[1]:.1f}, {combo[2]:.1f}, {combo[3]:.1f}, {combo[4]:.1f}]"
            logging.info(f"[{idx:4d}/{total_iters}] W: {weight_str} | Score: {score:5.2f}% | Win: {win_rate:5.2f}% | Decay: {exp_decay_percentile:5.2f}%")
            
            # Track the best visually in the console
            if score > best_score:
                best_score = score
                logging.info(f"   --> 🏆 NEW BEST SCORE!")
                
            # --- NEW: Save to CSV every iteration ---
            # Overwriting the CSV is extremely fast and ensures your file is always perfectly sorted
            pd.DataFrame(all_results).sort_values('score', ascending=False).to_csv(csv_path, index=False)
                
        except Exception as e:
            logging.error(f"Error testing combination {current_weights}: {e}")

    # 6. Output Summary
    logging.info("=" * 60)
    logging.info("OPTIMIZATION COMPLETE")
    logging.info(f"Full sorted results available in: {csv_path}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()