import os
import random
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import Strategy 1
from example_LSTM_merged_2.model_development_example_2 import precompute_features, compute_window_weights, create_sequences

# Globals
_FEATURES_DF = None
_lstm_model = None

def set_deterministic_seeds(seed=42):
    """Forces repeatable results across all random operations."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    global _FEATURES_DF
    global _lstm_model
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    
    # =========================================================================
    # OPTIMAL WEIGHTS (From Grid Search 1.0 -> 10.0)
    # Score: 75.48% | Win Rate: 96.87% | Exp Decay: 54.08%
    # =========================================================================
    optimal_weights = {
        'mvrv': 1.0,
        'fgi':  7.0,
        'poly': 6.0,
        'snp':  0.0,
        'ma':   0.0
    }
    
    return compute_window_weights(
        _lstm_model=_lstm_model, 
        features_df=_FEATURES_DF, 
        start_date=start_date, 
        end_date=end_date, 
        current_date=current_date, 
        weights=optimal_weights
    )

def train_lstm_model(df_inp):
    global _lstm_model
    set_deterministic_seeds(42)
    
    from scipy.signal import argrelextrema
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    logging.info("Training LSTM Model on 2018 market regimes...")
    df_btc_2018 = df_inp[df_inp.index.year == 2018]
    
    df_small = df_btc_2018[['PriceUSD_coinmetrics']].copy()
    df_small['Momentum'] = df_small['PriceUSD_coinmetrics'].diff()
    df_small['Acceleration'] = df_small['Momentum'].diff()
    df = df_small.copy()

    prices = df['PriceUSD_coinmetrics'].values
    max_idx = argrelextrema(prices, np.greater, order=5)[0]
    min_idx = argrelextrema(prices, np.less, order=5)[0]
    df['Target'] = 0
    df.iloc[max_idx, df.columns.get_loc('Target')] = 1
    df.iloc[min_idx, df.columns.get_loc('Target')] = 2

    cols = ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
    df['MA5'] = df['PriceUSD_coinmetrics'].rolling(5).mean()
    df['MA20'] = df['PriceUSD_coinmetrics'].rolling(20).mean()
    df['MomentumMA'] = df['Momentum'].rolling(10).mean()
    df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
    df['Volatility'] = df['PriceUSD_coinmetrics'].rolling(10).std()
    
    df_shifted = df.shift(1)[cols]
    df = df[['PriceUSD_coinmetrics']].join(df_shifted).dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, 20)
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    
    # Train the model silently to keep logs clean
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    _lstm_model = model
    logging.info("LSTM Model training complete.")
    return _lstm_model

def main():
    global _FEATURES_DF
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting DCA Strategy Analysis: LSTM Timing + Extreme Signal Sizing")
    
    # 1. Load Core Data
    btc_df = load_data()
    
    # 2. Train the LSTM Timing Engine
    train_lstm_model(btc_df)
    
    # 3. Precompute Features (MVRV, Polymarket, FGI, S&P)
    logging.info("Precomputing features and building signal overlays...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 4. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    
    # 5. Execute Run Analysis
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Final Capstone: LSTM + FGI(7.0) + Poly(6.0)",
    )

if __name__ == "__main__":
    main()