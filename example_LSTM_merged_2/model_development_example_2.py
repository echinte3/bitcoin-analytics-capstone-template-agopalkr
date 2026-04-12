"""Dynamic DCA weight computation using LSTM Timing + Signal Sizing.

This module integrates:
1. LSTM for detecting local extrema to identify optimal "buy points" (Timing).
2. Deterministic signals (MVRV, FGI, Polymarket, S&P 500, MA) for capital allocation (Sizing).
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
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

# Strategy parameters
MIN_W = 1e-6
MA_WINDOW = 200  
MVRV_GRADIENT_WINDOW = 30  
MVRV_ROLLING_WINDOW = 365  
MVRV_ACCEL_WINDOW = 14  
DYNAMIC_STRENGTH = 5.0  

# MVRV Zone thresholds
MVRV_ZONE_DEEP_VALUE = -2.0  
MVRV_ZONE_VALUE = -1.0  
MVRV_ZONE_CAUTION = 1.5  
MVRV_ZONE_DANGER = 2.5  

MVRV_VOLATILITY_WINDOW = 90  
MVRV_VOLATILITY_DAMPENING = 0.2  

# Global cache to prevent redundant LSTM inferences during grid search
_LSTM_TIMING_CACHE = {}

# =============================================================================
# Data Loading Functions (FGI, S&P 500, Polymarket)
# =============================================================================

def load_fgi_data() -> pd.DataFrame:
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "crypto_fear_and_greed_index_2019_2025.csv"
    
    if not file_path.exists():
        logging.warning("FGI data file not found. FGI signal will default to neutral.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["fgi_normalized"] = df["value"] / 100.0
        df = df.set_index("date").sort_index()
        return df[["value", "fgi_normalized", "value_classification"]]
    except Exception as e:
        logging.error(f"Failed to process FGI data: {e}")
        return pd.DataFrame()


def load_snp_data() -> pd.DataFrame:
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "SP500.csv"
    
    if not file_path.exists():
        logging.warning("S&P 500 data not found. Macro signal will default to neutral.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.set_index('Date').sort_index()
        df['snp_ma'] = df['Close'].rolling(20, min_periods=10).mean()
        df['snp_vs_ma'] = (df['Close'] / df['snp_ma']) - 1.0
        return df[['Close', 'snp_ma', 'snp_vs_ma']]
    except Exception as e:
        logging.error(f"Failed to process S&P 500 data: {e}")
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
    ).reset_index()
    
    daily_stats = daily_stats.set_index("created_date").sort_index()
    daily_stats["market_count_pct"] = daily_stats["daily_market_count"].rolling(30, min_periods=1).apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    daily_stats["volume_pct"] = daily_stats["daily_volume"].rolling(30, min_periods=1).apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    daily_stats["polymarket_sentiment"] = (daily_stats["market_count_pct"] * 0.5 + daily_stats["volume_pct"] * 0.5)
    daily_stats["polymarket_sentiment"] = daily_stats["polymarket_sentiment"].fillna(0.5)
    
    return daily_stats[["polymarket_sentiment"]]

# =============================================================================
# Helper Functions
# =============================================================================

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)

def classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
    return np.select(
        [mvrv_zscore < MVRV_ZONE_DEEP_VALUE, mvrv_zscore < MVRV_ZONE_VALUE, mvrv_zscore < MVRV_ZONE_CAUTION, mvrv_zscore < MVRV_ZONE_DANGER],
        [-2, -1, 0, 1],
        default=2,
    )

def compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)

def compute_signal_confidence(mvrv_zscore: np.ndarray, mvrv_gradient: np.ndarray, price_vs_ma: np.ndarray) -> np.ndarray:
    z_signal = -mvrv_zscore / 4  
    ma_signal = -price_vs_ma  
    gradient_alignment = np.where(z_signal < 0, np.where(mvrv_gradient > 0, 1.0, 0.5), np.where(mvrv_gradient < 0, 1.0, 0.5))
    signals = np.stack([z_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)
    agreement = 1.0 - np.clip(signal_std / 1.0, 0, 1)
    confidence = agreement * 0.7 + gradient_alignment * 0.3
    return np.clip(confidence, 0, 1)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

# =============================================================================
# Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found.")

    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)
        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(span=MVRV_GRADIENT_WINDOW, adjust=False).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)
        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean()
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3).fillna(0)
        mvrv_zone = pd.Series(classify_mvrv_zone(mvrv_z.values), index=mvrv_z.index)
        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)
        signal_confidence = pd.Series(0.5, index=price.index)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        signal_confidence = pd.Series(0.5, index=price.index)

    # Polymarket
    try:
        polymarket_df = load_polymarket_btc_sentiment()
        if not polymarket_df.empty:
            polymarket_sentiment = polymarket_df["polymarket_sentiment"].reindex(price.index, fill_value=0.5)
        else:
            polymarket_sentiment = pd.Series(0.5, index=price.index)
    except Exception:
        polymarket_sentiment = pd.Series(0.5, index=price.index)

    # FGI
    try:
        fgi_df = load_fgi_data()
        if not fgi_df.empty:
            fgi_sentiment = fgi_df["fgi_normalized"].reindex(price.index).ffill().fillna(0.5)
        else:
            fgi_sentiment = pd.Series(0.5, index=price.index)
    except Exception:
        fgi_sentiment = pd.Series(0.5, index=price.index)

    # S&P 500
    try:
        snp_df = load_snp_data()
        if not snp_df.empty:
            snp_vs_ma = snp_df["snp_vs_ma"].reindex(price.index).ffill().fillna(0.0)
        else:
            snp_vs_ma = pd.Series(0.0, index=price.index)
    except Exception:
        snp_vs_ma = pd.Series(0.0, index=price.index)

    features = pd.DataFrame({
        PRICE_COL: price,
        "price_ma": ma,
        "price_vs_ma": price_vs_ma,
        "mvrv_zscore": mvrv_z,
        "mvrv_gradient": mvrv_gradient,
        "mvrv_acceleration": mvrv_acceleration,
        "mvrv_zone": mvrv_zone,
        "mvrv_volatility": mvrv_volatility,
        "signal_confidence": signal_confidence,
        "polymarket_sentiment": polymarket_sentiment,
        "fgi_sentiment": fgi_sentiment,
        "snp_vs_ma": snp_vs_ma,
    }, index=price.index)

    signal_cols = ["price_vs_ma", "mvrv_zscore", "mvrv_gradient", "mvrv_acceleration", "mvrv_zone", "mvrv_volatility", "polymarket_sentiment", "fgi_sentiment", "snp_vs_ma"]
    features[signal_cols] = features[signal_cols].shift(1)

    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["polymarket_sentiment"] = features["polymarket_sentiment"].fillna(0.5)
    features["fgi_sentiment"] = features["fgi_sentiment"].fillna(0.5)
    features = features.fillna(0)

    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    return features

# =============================================================================
# Dynamic Sizing & LSTM Allocation
# =============================================================================

def compute_signal_multipliers(df_row, weights: dict = None) -> float:
    """Computes final conviction multiplier based on interpolated signals."""
    if weights is None:
        weights = {'mvrv': 0.0, 'fgi': 0.0, 'poly': 0.0, 'snp': 0.0, 'ma': 0.0}

    mvrv_zscore = getattr(df_row, 'mvrv_zscore', 0.0)
    fgi = getattr(df_row, 'fgi_sentiment', 0.5)
    poly = getattr(df_row, 'polymarket_sentiment', 0.5)
    snp = getattr(df_row, 'snp_vs_ma', 0.0)
    btc_ma = getattr(df_row, 'price_vs_ma', 0.0)

    # Bounds Interpolation
    m_raw_mvrv = np.interp(mvrv_zscore, [-2.0, 2.5], [1.5, 0.5])
    m_raw_fgi = np.interp(fgi, [0.0, 1.0], [1.5, 0.5])
    m_raw_poly = np.interp(poly, [0.0, 1.0], [0.8, 1.2])
    m_raw_snp = np.interp(snp, [-0.05, 0.05], [0.8, 1.2])
    m_raw_ma = np.interp(btc_ma, [-0.1, 0.1], [0.8, 1.2])

    # Weights Application
    m_final_mvrv = 1.0 + weights.get('mvrv', 0.0) * (m_raw_mvrv - 1.0)
    m_final_fgi  = 1.0 + weights.get('fgi', 0.0)  * (m_raw_fgi - 1.0)
    m_final_poly = 1.0 + weights.get('poly', 0.0) * (m_raw_poly - 1.0)
    m_final_snp  = 1.0 + weights.get('snp', 0.0)  * (m_raw_snp - 1.0)
    m_final_ma   = 1.0 + weights.get('ma', 0.0)   * (m_raw_ma - 1.0)

    conviction = m_final_mvrv * m_final_fgi * m_final_poly * m_final_snp * m_final_ma
    return max(0.01, conviction)


def computeQtyLSTM_Dynamic(df, buy_pts, weights=None):
    """Sizes LSTM buy points dynamically based on signal conviction."""
    lst = []
    cnt = 0
    prev_pt = -1
    map_buy_pts = set(buy_pts)
    AMT = 10000
    n = len(df)
    
    for row in df.itertuples(index=True):
        qty_for_day = 1e-6
        is_last_day = (cnt == n - 1)
        
        if cnt in map_buy_pts or is_last_day:
            price = getattr(row, 'PriceUSD_coinmetrics', 1e-6)
            days_accumulated = cnt - prev_pt
            
            try:
                if days_accumulated > 0:
                    base_cash = days_accumulated * (AMT / n)
                    conviction = compute_signal_multipliers(row, weights)
                    final_cash = base_cash * conviction
                    qty_for_day = final_cash / price
            except ZeroDivisionError:
                qty_for_day = 1e-6
                
            prev_pt = cnt
            
        cnt += 1
        lst.append(qty_for_day)

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
        n_past: int | None = None,
        locked_weights: np.ndarray | None = None,
        weights: dict | None = None,
) -> pd.Series:
    
    global _LSTM_TIMING_CACHE
    
    df_inp = features_df.loc[start_date:end_date].copy()
    if df_inp.empty:
        return pd.Series(dtype=float)

    # 1. Check if we already computed the LSTM timing for this specific date window
    cache_key = (start_date, end_date)
    
    if cache_key not in _LSTM_TIMING_CACHE:
        # Prepare features for LSTM
        df_small = df_inp[['PriceUSD_coinmetrics']].copy()
        df_small['Momentum'] = df_small['PriceUSD_coinmetrics'].diff().copy()
        df_small['Acceleration'] = df_small['Momentum'].diff().copy()
        df = df_small.copy()

        cols= ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
        df['MA5'] = df['PriceUSD_coinmetrics'].rolling(5).mean()
        df['MA20'] = df['PriceUSD_coinmetrics'].rolling(20).mean()
        df['MomentumMA'] = df['Momentum'].rolling(10).mean()
        df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
        df['Volatility'] = df['PriceUSD_coinmetrics'].rolling(10).std()
        df_shifted = df.shift(1)[cols]

        df = df_inp[['PriceUSD_coinmetrics']].join(df_shifted).dropna()

        # Predict Extrema (This is the heavy calculation)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        X_test, y_test = create_sequences(scaled_data, 20)
        
        pred = _lstm_model.predict(X_test, verbose=0)
        pred_inv = scaler.inverse_transform(np.c_[pred, np.zeros((len(pred), df.shape[1] - 1))])[:, 0]

        upTrend = False
        downTrend = False
        buy_pts = []
        
        for i in range(len(pred_inv)):
            if i == 0: continue
            if pred_inv[i] > pred_inv[i-1]:
                if downTrend == True:
                    buy_pts.append(i)
                upTrend = True
                downTrend = False
            elif pred_inv[i] < pred_inv[i-1]:
                downTrend = True
                upTrend = False
                
        # Save the result to the cache so we never compute this window again
        _LSTM_TIMING_CACHE[cache_key] = buy_pts

    # 2. Retrieve the pre-computed timing points
    buy_pts = _LSTM_TIMING_CACHE[cache_key]

    # 3. Apply the Dynamic Sizing (This is fast, purely Pandas math)
    l = computeQtyLSTM_Dynamic(df_inp, buy_pts, weights=weights)
    return pd.Series(l, index=df_inp.index)


def compute_window_weights(
    _lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    weights: dict | None = None,
) -> pd.Series:
    
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        if "mvrv_zone" in placeholder.columns: placeholder["mvrv_zone"] = 0
        if "mvrv_volatility" in placeholder.columns: placeholder["mvrv_volatility"] = 0.5
        if "signal_confidence" in placeholder.columns: placeholder["signal_confidence"] = 0.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0

    weights_out = compute_weights_fast(_lstm_model, features_df, start_date, end_date, n_past, locked_weights, weights=weights)
    return weights_out.reindex(full_range, fill_value=0.0)