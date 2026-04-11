import logging
import pandas as pd
from pathlib import Path
import numpy as np

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import Example 1 model
from example_LSTM.model_development_example_2 import precompute_features, compute_window_weights, load_snp_data

# Global variable to store precomputed features
_FEATURES_DF = None

_lstm_model = None

#Uses global lstm_model
def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Example 1 compute_window_weights.
    
    Adapts the specific Example 1 model function to the interface expected 
    by the template backtest engine.
    """
    global _FEATURES_DF
    global _lstm_model

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    #print('_lstm_model', _lstm_model.summary())
    print('_lstm_model====>', _lstm_model)
    print('LSTM Printing compute_weights_wrapper start_date=',start_date,'end_date=',end_date)
    
    # For backtesting, current_date = end_date (all dates are in the past)
    current_date = end_date
    
    return compute_window_weights(_lstm_model, _FEATURES_DF, start_date, end_date, current_date)

def create_sequences(data,  window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 0])
        #y.append(df.iloc[i+window]['PriceUSD'])
    return np.array(X), np.array(y)


#Build the lstm model
def lstm(df_inp):

    global _lstm_model
    import numpy as np
    import pandas as pd
    from scipy.signal import argrelextrema
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt

    df_btc_2018=df_inp[df_inp.index.year == 2018]
    window=30

    df_small = df_btc_2018[['PriceUSD']]
    df_small['Momentum'] = df_small['PriceUSD'].diff()
    df_small['Acceleration'] = df_small['Momentum'].diff()


    df = df_small.copy()

    # 2. Label Extremes
    prices = df['PriceUSD'].values
    max_idx = argrelextrema(prices, np.greater, order=5)[0]
    min_idx = argrelextrema(prices, np.less, order=5)[0]
    df['Target'] = 0
    df.iloc[max_idx, df.columns.get_loc('Target')] = 1
    df.iloc[min_idx, df.columns.get_loc('Target')] = 2

    # 3. Features
    cols= ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
    df['MA5'] = df['PriceUSD'].rolling(5).mean()
    df['MA20'] = df['PriceUSD'].rolling(20).mean()
    df['MomentumMA'] = df['Momentum'].rolling(10).mean()
    df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
    df['Volatility'] = df['PriceUSD'].rolling(10).std()
    df_shifted=df.shift(1)[cols]

    df = df[['PriceUSD']].join(df_shifted)
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, 20)
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(
        X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    model.summary()
    model.fit(X_train, y_train, epochs=30, batch_size=16)

    _lstm_model = model
    return _lstm_model


def main():
    global _FEATURES_DF
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy Analysis - Example 1 (Polymarket)")
    
    # 1. Load Data
    btc_df = load_data()

    #1a LSTM model. This sets the global model
    lstm(btc_df)
    
    # 2. Precompute Features (using Example 1 logic)
    logging.info("Precomputing features (including MVRV & Polymarket)...")
    _FEATURES_DF = precompute_features(btc_df)
    print('WHAT IS ', _FEATURES_DF.shape)


    # 2a. Load SP
    df_sp500 = load_snp_data()
    _FEATURES_DF=pd.merge(_FEATURES_DF, df_sp500,left_index=True, right_index=True, how='left')
    print('WHAT IS 2-->', _FEATURES_DF.shape)

    #FG in

    # 3. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    
    # 4. Run Analysis (reusing Template engine)
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        df_sp500=df_sp500,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Example 1 (Polymarket)",
    )

if __name__ == "__main__":
    main()
