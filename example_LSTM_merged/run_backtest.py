import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import the newly updated Example 1 model
from example_LSTM_merged.model_development_example_2 import precompute_features, compute_window_weights, create_sequences

_FEATURES_DF = None
_lstm_model = None

# =============================================================================
# STRATEGY DIAL
# 0.0 = Signal completely turned OFF (Ignored)
# 1.0 = Normal effect
# 2.0 = Magnified by 2x (Squared)
# 0.5 = Dampened (Square Root)
# =============================================================================
MASTER_WEIGHTS = {
    'mvrv': 	0.0,    # Macro Valuation
    'ma': 	0.0,    # Long-term Trend
    'fgi': 	0.0,    # Retail Contrarian (Aggressively Magnified!)
    'poly': 	0.0,    # Polymarket Sentiment (Turned OFF for this test)
    'snp': 	0.0     # S&P 500 Risk Regime
}

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapts the specific model function to the template backtest engine."""
    global _FEATURES_DF
    global _lstm_model
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    
    return compute_window_weights(
        _lstm_model, 
        _FEATURES_DF, 
        start_date, 
        end_date, 
        current_date, 
        weights=MASTER_WEIGHTS # <-- Pass the control dials here
    )

def lstm(df_inp: pd.DataFrame):
    """Trains the global LSTM timing model on 2018 historical data."""
    global _lstm_model
    from scipy.signal import argrelextrema
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    logging.info("Training LSTM Timing Engine on 2018 data...")
    
    df_btc_2018 = df_inp[df_inp.index.year == 2018].copy()
    
    df_small = df_btc_2018[['PriceUSD_coinmetrics']].rename(columns={'PriceUSD_coinmetrics': 'PriceUSD'})
    df_small['Momentum'] = df_small['PriceUSD'].diff()
    df_small['Acceleration'] = df_small['Momentum'].diff()
    df = df_small.copy()

    # Label Extremes
    prices = df['PriceUSD'].values
    max_idx = argrelextrema(prices, np.greater, order=5)[0]
    min_idx = argrelextrema(prices, np.less, order=5)[0]
    df['Target'] = 0
    df.iloc[max_idx, df.columns.get_loc('Target')] = 1
    df.iloc[min_idx, df.columns.get_loc('Target')] = 2

    # Features
    cols= ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
    df['MA5'] = df['PriceUSD'].rolling(5).mean()
    df['MA20'] = df['PriceUSD'].rolling(20).mean()
    df['MomentumMA'] = df['Momentum'].rolling(10).mean()
    df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
    df['Volatility'] = df['PriceUSD'].rolling(10).std()
    
    df_shifted = df.shift(1)[cols]
    df = df[['PriceUSD']].join(df_shifted).dropna()

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
    
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    logging.info("LSTM Training Complete.")
    _lstm_model = model
    return _lstm_model

def main():
    global _FEATURES_DF
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy: LSTM Timing + Continuous Factor Sizing")
    
    btc_df = load_data()
    lstm(btc_df)
    
    logging.info("Precomputing Conviction signals (MVRV, FGI, S&P 500, Polymarket)...")
    _FEATURES_DF = precompute_features(btc_df)
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="V3 (Continuous Exponent Weighting)",
    )

if __name__ == "__main__":
    main()