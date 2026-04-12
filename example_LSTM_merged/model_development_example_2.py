"""Dynamic DCA weight computation using LSTM Timing + Independent Continuous Sizing.

Architecture:
1. Timing Engine (LSTM): Identifies local momentum reversals to trigger "Buy Points".
2. Sizing Engine: Uses continuous linear transformations mapped to an exponential 
   weighting system to scale allocation sizes smoothly and independently.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import base functionality from template
from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)

# =============================================================================
# Constants & Setup
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

MA_WINDOW = 200  
MVRV_GRADIENT_WINDOW = 30  
MVRV_ROLLING_WINDOW = 365  

# =============================================================================
# External Data Loading
# =============================================================================

def load_fgi_data() -> pd.DataFrame:
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "crypto_fear_and_greed_index_2019_2025.csv"
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["fgi_normalized"] = df["value"] / 100.0
        return df.set_index("date").sort_index()[["fgi_normalized"]]
    except Exception:
        return pd.DataFrame()

def load_snp_data() -> pd.DataFrame:
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "SP500.csv"
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.set_index('Date').sort_index()
        df['snp_ma'] = df['Close'].rolling(20, min_periods=10).mean()
        df['snp_vs_ma'] = (df['Close'] / df['snp_ma']) - 1.0
        return df[['snp_vs_ma']]
    except Exception:
        return pd.DataFrame()

def load_polymarket_btc_sentiment() -> pd.DataFrame:
    polymarket_data = load_polymarket_data()
    if "markets" not in polymarket_data:
        return pd.DataFrame()
    markets_df = polymarket_data["markets"]
    btc_markets = markets_df[markets_df["question"].str.contains("Bitcoin|BTC|btc", case=False, na=False)].copy()
    if btc_markets.empty:
        return pd.DataFrame()
    
    btc_markets["created_date"] = pd.to_datetime(btc_markets["created_at"]).dt.normalize()
    daily_stats = btc_markets.groupby("created_date").agg(
        daily_market_count=("market_id", "count"),
        daily_volume=("volume", "sum")
    ).reset_index().set_index("created_date").sort_index()
    
    daily_stats["market_count_pct"] = daily_stats["daily_market_count"].rolling(30, min_periods=1).apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    daily_stats["volume_pct"] = daily_stats["daily_volume"].rolling(30, min_periods=1).apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    daily_stats["polymarket_sentiment"] = (daily_stats["market_count_pct"] * 0.5 + daily_stats["volume_pct"] * 0.5).fillna(0.5)
    
    return daily_stats[["polymarket_sentiment"]]

# =============================================================================
# Feature Engineering
# =============================================================================

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 1. BTC Price vs MA
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    # 2. MVRV Z-Score
    mvrv_z = zscore(df[MVRV_COL].loc[price.index], MVRV_ROLLING_WINDOW).clip(-4, 4) if MVRV_COL in df.columns else pd.Series(0.0, index=price.index)

    # 3. Polymarket
    try:
        poly_df = load_polymarket_btc_sentiment()
        polymarket_sentiment = poly_df["polymarket_sentiment"].reindex(price.index).fillna(0.5) if not poly_df.empty else pd.Series(0.5, index=price.index)
    except Exception:
        polymarket_sentiment = pd.Series(0.5, index=price.index)

    # 4. FGI
    try:
        fgi_df = load_fgi_data()
        fgi_sentiment = fgi_df["fgi_normalized"].reindex(price.index).ffill().fillna(0.5) if not fgi_df.empty else pd.Series(0.5, index=price.index)
    except Exception:
        fgi_sentiment = pd.Series(0.5, index=price.index)

    # 5. S&P 500
    try:
        snp_df = load_snp_data()
        snp_vs_ma = snp_df["snp_vs_ma"].reindex(price.index).ffill().fillna(0.0) if not snp_df.empty else pd.Series(0.0, index=price.index)
    except Exception:
        snp_vs_ma = pd.Series(0.0, index=price.index)

    features = pd.DataFrame({
        PRICE_COL: price,
        "price_vs_ma": price_vs_ma,
        "mvrv_zscore": mvrv_z,
        "polymarket_sentiment": polymarket_sentiment,
        "fgi_sentiment": fgi_sentiment,
        "snp_vs_ma": snp_vs_ma,
    }, index=price.index)

    # Shift signals to prevent lookahead bias
    signal_cols = ["price_vs_ma", "mvrv_zscore", "polymarket_sentiment", "fgi_sentiment", "snp_vs_ma"]
    features[signal_cols] = features[signal_cols].shift(1).fillna(0)
    features["polymarket_sentiment"] = features["polymarket_sentiment"].replace(0, 0.5)
    features["fgi_sentiment"] = features["fgi_sentiment"].replace(0, 0.5)

    return features

# =============================================================================
# The Sizing Engine (Continuous & Exponentially Weighted)
# =============================================================================

def compute_conviction_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    polymarket_sentiment: np.ndarray,
    fgi_sentiment: np.ndarray,
    snp_vs_ma: np.ndarray,
    weights: dict,
) -> np.ndarray:
    """Computes final conviction using continuous transformations and power weights."""
    
    # 1. MVRV Base: Inverse relationship (Low Z-score = High Conviction)
    # Z-Score of -2 outputs ~1.5. Z-Score of 2 outputs ~0.5. Clipped to sane extremes.
    mvrv_base = np.clip(1.0 - (0.25 * mvrv_zscore), 0.2, 2.5)
    mvrv_final = mvrv_base ** weights.get('mvrv', 1.0)

    # 2. Long-term MA Base: Inverse relationship (Below MA = Undervalued)
    # If price is 40% below MA (-0.4), outputs 1.2
    ma_base = np.clip(1.0 - (0.5 * price_vs_ma), 0.5, 1.5)
    ma_final = ma_base ** weights.get('ma', 1.0)

    # 3. FGI Base: Contrarian (High Fear/Low FGI = High Conviction)
    # Fear of 0.1 outputs 1.4. Greed of 0.9 outputs 0.6.
    fgi_base = np.clip(1.0 + (0.5 - fgi_sentiment), 0.5, 1.5)
    fgi_final = fgi_base ** weights.get('fgi', 1.0)

    # 4. Polymarket Base: Follower (High Sentiment = High Conviction)
    # Sentiment of 0.9 outputs 1.4. Sentiment of 0.1 outputs 0.6.
    poly_base = np.clip(1.0 + (polymarket_sentiment - 0.5), 0.5, 1.5)
    poly_final = poly_base ** weights.get('poly', 1.0)

    # 5. S&P 500 Base: Risk On Regime (Above MA = High Conviction)
    # Amplified by 2x locally to make small macro moves matter more.
    snp_base = np.clip(1.0 + (snp_vs_ma * 2.0), 0.7, 1.3)
    snp_final = snp_base ** weights.get('snp', 1.0)

    # The Final Product
    conviction = mvrv_final * ma_final * fgi_final * poly_final * snp_final

    # Absolute Guardrails: Even with high exponents, never exceed 5x standard bet or drop below 0.1x
    return np.clip(conviction, 0.1, 5.0)

# =============================================================================
# The Timing & Allocation Engine (LSTM + Conviction)
# =============================================================================

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

def computeQtyLSTM(df: pd.DataFrame, buy_pts: list, conviction_series: np.ndarray) -> np.ndarray:
    lst = []
    cnt = 0
    prev_pt = -1
    map_buy_pts = set(buy_pts)
    AMT = 10000
    n = len(df)
    
    for row, conviction_val in zip(df.itertuples(index=True), conviction_series):
        qty_for_day = 1e-6
        is_last_day = (cnt == n - 1)
        
        if cnt in map_buy_pts or is_last_day:
            price = getattr(row, PRICE_COL)
            days_accumulated = cnt - prev_pt
            
            if days_accumulated > 0:
                baseline_cash = days_accumulated * (AMT / n)
                adjusted_cash = baseline_cash * conviction_val
                
                try:
                    qty_for_day = adjusted_cash / price
                except ZeroDivisionError:
                    qty_for_day = 1e-6
                    
            prev_pt = cnt
            
        lst.append(qty_for_day)
        cnt += 1

    lst = np.array(lst)
    total_sum = np.sum(lst)
    if total_sum > 0:
        lst = lst / total_sum
        
    return lst

def compute_weights_fast(
    _lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weights: dict,
) -> pd.Series:
    
    df_inp = features_df.loc[start_date:end_date].copy()
    if df_inp.empty:
        return pd.Series(dtype=float)

    # 1. Sizing Engine
    conviction_series = compute_conviction_multiplier(
        price_vs_ma=_clean_array(df_inp["price_vs_ma"].values),
        mvrv_zscore=_clean_array(df_inp["mvrv_zscore"].values),
        polymarket_sentiment=_clean_array(df_inp["polymarket_sentiment"].values),
        fgi_sentiment=_clean_array(df_inp["fgi_sentiment"].values),
        snp_vs_ma=_clean_array(df_inp["snp_vs_ma"].values),
        weights=weights
    )

    # 2. Timing Engine (LSTM)
    df_small = df_inp[[PRICE_COL]].copy()
    df_small['Momentum'] = df_small[PRICE_COL].diff()
    df_small['Acceleration'] = df_small['Momentum'].diff()
    df = df_small.copy()

    cols= ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
    df['MA5'] = df[PRICE_COL].rolling(5).mean()
    df['MA20'] = df[PRICE_COL].rolling(20).mean()
    df['MomentumMA'] = df['Momentum'].rolling(10).mean()
    df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
    df['Volatility'] = df[PRICE_COL].rolling(10).std()
    
    df_shifted = df.shift(1)[cols]
    df_lstm_ready = df_inp[[PRICE_COL]].join(df_shifted).dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm_ready)
    
    if len(scaled_data) > 20:
        X_test, _ = create_sequences(scaled_data, 20)
        pred = _lstm_model.predict(X_test, verbose=0)
        pred_inv = scaler.inverse_transform(np.c_[pred, np.zeros((len(pred), df_lstm_ready.shape[1] - 1))])[:, 0]

        upTrend, downTrend = False, False
        buy_pts = []
        for i in range(1, len(pred_inv)):
            if pred_inv[i] > pred_inv[i-1]:
                if downTrend:
                    offset = len(df_inp) - len(pred_inv)
                    buy_pts.append(i + offset)
                upTrend, downTrend = True, False
            elif pred_inv[i] < pred_inv[i-1]:
                downTrend, upTrend = True, False
    else:
        buy_pts = [] 

    weights_array = computeQtyLSTM(df_inp, buy_pts, conviction_series)
    return pd.Series(weights_array, index=df_inp.index)

def compute_window_weights(
    _lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    weights: dict,
) -> pd.Series:
    
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    missing = full_range.difference(features_df.index)
    
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        placeholder["fgi_sentiment"] = 0.5
        placeholder["polymarket_sentiment"] = 0.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    weights_series = compute_weights_fast(_lstm_model, features_df, start_date, end_date, weights)
    return weights_series.reindex(full_range, fill_value=0.0)