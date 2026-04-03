"""Dynamic DCA weight computation using MVRV + 200-day Max Drawdown + Polymarket strategy.

This module extends the template model with Polymarket sentiment integration
and a rolling maximum drawdown metric to aggressively accumulate during capitulation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
MA_WINDOW = 200  # 200-day rolling window (now used for Moving Maximum)
MVRV_GRADIENT_WINDOW = 30  # Window for MVRV trend detection
MVRV_ROLLING_WINDOW = 365  # Window for MVRV Z-score normalization
MVRV_ACCEL_WINDOW = 14  # Window for acceleration calculation
DYNAMIC_STRENGTH = 5.0  # Multiplier for weight adjustments

# MVRV Zone thresholds (based on historical distribution)
MVRV_ZONE_DEEP_VALUE = -2.0  # Z-score threshold for deep value
MVRV_ZONE_VALUE = -1.0  # Z-score threshold for value
MVRV_ZONE_CAUTION = 1.5  # Z-score threshold for caution
MVRV_ZONE_DANGER = 2.5  # Z-score threshold for danger

# Volatility adjustment parameters
MVRV_VOLATILITY_WINDOW = 90  # Window for volatility calculation
MVRV_VOLATILITY_DAMPENING = (
    0.2  # How much to dampen signals in high volatility (reduced)
)

# Feature column names (for compatibility)
FEATS = [
    "price_vs_max",  # Updated from price_vs_ma
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
    "polymarket_sentiment",
]

# =============================================================================
# FGI Data Loading
# =============================================================================

def load_fgi_data() -> pd.DataFrame:
    """Load the Crypto Fear and Greed Index (FGI) data."""
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "crypto_fear_and_greed_index_2019_2025.csv"
    
    if not file_path.exists():
        logging.warning(
            f"FGI data file not found at {file_path}. "
            "FGI signal will default to neutral."
        )
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

# =============================================================================
# S&P Data Loading
# =============================================================================

def load_snp_data() -> pd.DataFrame:
    """Load S&P 500 data and compute 20-day MA distance."""
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

# =============================================================================
# Model-Specific Data Loading
# =============================================================================

def load_polymarket_btc_sentiment() -> pd.DataFrame:
    """Load Polymarket BTC-related markets and compute daily sentiment."""
    polymarket_data = load_polymarket_data()
    
    if "markets" not in polymarket_data:
        return pd.DataFrame()
    
    markets_df = polymarket_data["markets"]
    btc_markets = markets_df[
        markets_df["question"].str.contains("Bitcoin|BTC|btc", case=False, na=False)
    ].copy()
    
    if btc_markets.empty:
        return pd.DataFrame()
    
    btc_markets["created_date"] = pd.to_datetime(btc_markets["created_at"]).dt.normalize()
    daily_stats = btc_markets.groupby("created_date").agg(
        daily_market_count=("market_id", "count"),
        daily_volume=("volume", "sum")
    ).reset_index()
    
    daily_stats = daily_stats.set_index("created_date").sort_index()
    daily_stats["market_count_pct"] = (
        daily_stats["daily_market_count"]
        .rolling(30, min_periods=1)
        .apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    )
    
    daily_stats["volume_pct"] = (
        daily_stats["daily_volume"]
        .rolling(30, min_periods=1)
        .apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
    )
    
    daily_stats["polymarket_sentiment"] = (
        daily_stats["market_count_pct"] * 0.5 + daily_stats["volume_pct"] * 0.5
    )
    daily_stats["polymarket_sentiment"] = daily_stats["polymarket_sentiment"].fillna(0.5)
    
    return daily_stats[["polymarket_sentiment"]]

# =============================================================================
# Helper Functions
# =============================================================================

def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)

def classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Classify MVRV into discrete zones for regime detection."""
    return np.select(
        [
            mvrv_zscore < MVRV_ZONE_DEEP_VALUE,
            mvrv_zscore < MVRV_ZONE_VALUE,
            mvrv_zscore < MVRV_ZONE_CAUTION,
            mvrv_zscore < MVRV_ZONE_DANGER,
        ],
        [-2, -1, 0, 1],
        default=2,
    )

def compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility of MVRV Z-score."""
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)
        if len(x) > 1
        else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)

def compute_signal_confidence(
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    price_vs_max: np.ndarray,
) -> np.ndarray:
    """Compute confidence score based on signal agreement."""
    # Normalize all signals to [-1, 1] where negative = buy signal
    z_signal = -mvrv_zscore / 4  
    drawdown_signal = -price_vs_max  # Deeper drawdown = stronger buy signal

    gradient_alignment = np.where(
        z_signal < 0,  
        np.where(mvrv_gradient > 0, 1.0, 0.5),  
        np.where(mvrv_gradient < 0, 1.0, 0.5),  
    )

    signals = np.stack([z_signal, drawdown_signal], axis=0)
    signal_std = signals.std(axis=0)

    max_std = 1.0  
    agreement = 1.0 - np.clip(signal_std / max_std, 0, 1)
    confidence = agreement * 0.7 + gradient_alignment * 0.3

    return np.clip(confidence, 0, 1)

def compute_mean_reversion_pressure(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute mean reversion pressure based on distance from equilibrium."""
    pressure = np.tanh(mvrv_zscore * 0.5)
    extreme_pressure = np.where(
        np.abs(mvrv_zscore) > 2,
        np.sign(mvrv_zscore) * 0.3 * (np.abs(mvrv_zscore) - 2),
        0,
    )
    return np.clip(pressure + extreme_pressure, -1, 1)

# =============================================================================
# Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MVRV and Max Drawdown features for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_max: Negative percentage distance from 200-day max [-1, 0]
    - mvrv_zscore: MVRV Z-score (365-day window), clipped to [-4, 4]
    - mvrv_gradient: Smoothed MVRV trend direction in [-1, 1]
    - mvrv_acceleration: Second derivative of MVRV gradient (momentum)
    - mvrv_zone: Discrete zone classification [-2, -1, 0, 1, 2]
    - polymarket_sentiment: Normalized sentiment from BTC market activity [0, 1]
    - fgi_sentiment: Normalized Crypto Fear & Greed Index [0, 1] 
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 200-day Moving Maximum and Drawdown
    rolling_max = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        # Returns a negative decimal (e.g., -0.45 for a 45% crash). Capped at 0.
        price_vs_max = ((price / rolling_max) - 1.0).clip(upper=0).fillna(0)

    # MVRV features
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)

        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(
            span=MVRV_GRADIENT_WINDOW, adjust=False
        ).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)

        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean()
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3).fillna(0)

        mvrv_zone = pd.Series(
            classify_mvrv_zone(mvrv_z.values),
            index=mvrv_z.index,
        )

        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)
        signal_confidence = pd.Series(0.5, index=price.index)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        signal_confidence = pd.Series(0.5, index=price.index)

    # Load External Sentiments
    try:
        polymarket_df = load_polymarket_btc_sentiment()
        if not polymarket_df.empty:
            polymarket_sentiment = polymarket_df["polymarket_sentiment"].reindex(
                price.index, fill_value=0.5
            )
        else:
            polymarket_sentiment = pd.Series(0.5, index=price.index)
    except Exception as e:
        polymarket_sentiment = pd.Series(0.5, index=price.index)

    try:
        fgi_df = load_fgi_data()  
        if not fgi_df.empty:
            fgi_sentiment = fgi_df["fgi_normalized"].reindex(price.index).ffill().fillna(0.5)
        else:
            fgi_sentiment = pd.Series(0.5, index=price.index)
    except Exception as e:
        fgi_sentiment = pd.Series(0.5, index=price.index)

    try:
        snp_df = load_snp_data()
        if not snp_df.empty:
            snp_vs_ma = snp_df["snp_vs_ma"].reindex(price.index).ffill().fillna(0.0)
        else:
            snp_vs_ma = pd.Series(0.0, index=price.index)
    except Exception as e:
        snp_vs_ma = pd.Series(0.0, index=price.index)

    # Build and lag features
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_max": rolling_max,
            "price_vs_max": price_vs_max,
            "mvrv_zscore": mvrv_z,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_zone": mvrv_zone,
            "mvrv_volatility": mvrv_volatility,
            "signal_confidence": signal_confidence,
            "polymarket_sentiment": polymarket_sentiment,
            "fgi_sentiment": fgi_sentiment,  
            "snp_vs_ma": snp_vs_ma, 
        },
        index=price.index,
    )

    # Lag signals by 1 day
    signal_cols = [
        "price_vs_max",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
        "polymarket_sentiment",
        "fgi_sentiment",  
        "snp_vs_ma",  
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["polymarket_sentiment"] = features["polymarket_sentiment"].fillna(0.5)
    features["fgi_sentiment"] = features["fgi_sentiment"].fillna(0.5) 
    features = features.fillna(0)

    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_max"].values,
    )

    return features

# =============================================================================
# Dynamic Multiplier
# =============================================================================

def compute_asymmetric_extreme_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute asymmetric boost for extreme MVRV values."""
    boost = np.zeros_like(mvrv_zscore)

    deep_value_mask = mvrv_zscore < MVRV_ZONE_DEEP_VALUE
    boost = np.where(
        deep_value_mask,
        0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5,
        boost,
    )

    value_mask = (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE)
    boost = np.where(
        value_mask,
        -0.5 * mvrv_zscore,  
        boost,
    )

    caution_mask = (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER)
    boost = np.where(
        caution_mask,
        -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION),
        boost,
    )

    danger_mask = mvrv_zscore >= MVRV_ZONE_DANGER
    boost = np.where(
        danger_mask,
        -0.5 * (mvrv_zscore - MVRV_ZONE_DANGER) ** 2 - 0.3,
        boost,
    )

    return boost

def compute_acceleration_modifier(
    mvrv_acceleration: np.ndarray,
    mvrv_gradient: np.ndarray,
) -> np.ndarray:
    """Compute modifier based on MVRV acceleration (momentum)."""
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0

    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),  
        1.0 - 0.2 * np.abs(mvrv_acceleration),  
    )

    return np.clip(modifier, 0.5, 1.5)

def compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray,
    mvrv_zscore: np.ndarray,
) -> np.ndarray:
    """Compute trend modifier with adaptive thresholds."""
    threshold = np.where(
        mvrv_zscore < -1,
        0.1,  
        np.where(mvrv_zscore > 1.5, 0.4, 0.2),  
    )

    modifier = np.where(
        mvrv_gradient > threshold,
        1.0 + 0.5 * np.minimum(mvrv_gradient, 1.0),  
        np.where(
            mvrv_gradient < -threshold,
            0.3 + 0.2 * (1 + mvrv_gradient),  
            1.0,  
        ),
    )

    return np.clip(modifier, 0.3, 1.5)

def compute_drawdown_signal(
    price_vs_max: np.ndarray, 
    mvrv_gradient: np.ndarray, 
    mvrv_zscore: np.ndarray
) -> np.ndarray:
    """Convert drawdown into an accumulation signal.
    
    Drawdown is a negative percentage. We invert it to create a positive 
    buy signal. The deeper the crash, the stronger the signal.
    """
    # A -40% (-0.4) drawdown becomes a +0.4 buy signal multiplier
    base_signal = -price_vs_max 
    
    # Apply trend modifier so we buy even harder when momentum shifts 
    # upwards while still in a deep drawdown.
    trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    
    return base_signal * trend_modifier

def compute_dynamic_multiplier(
    price_vs_max: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
    polymarket_sentiment: np.ndarray | None = None,
    fgi_sentiment: np.ndarray | None = None,  
    snp_vs_ma: np.ndarray | None = None, 
    weights: dict | None = None,  
) -> np.ndarray:
    """Compute weight multiplier using MVRV, Drawdown, and Sentiment signals.
    
    Current Configured Weights:
    - MVRV (64%)
    - Drawdown (16%) 
    - Polymarket (20%)
    - FGI (0%)
    - S&P 500 Macro (0%)
    """
    # Update default weights to requested ratios
    if weights is None:
        weights = {'mvrv': 0.64, 'drawdown': 0.16, 'fgi': 0.0, 'snp': 0.0, 'poly': 0.20}

    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros_like(mvrv_zscore)
    if mvrv_volatility is None:
        mvrv_volatility = np.full_like(mvrv_zscore, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full_like(mvrv_zscore, 0.5)
    if polymarket_sentiment is None:
        polymarket_sentiment = np.full_like(mvrv_zscore, 0.5)
    if fgi_sentiment is None:
        fgi_sentiment = np.full_like(mvrv_zscore, 0.5)
    if snp_vs_ma is None:
        snp_vs_ma = np.zeros_like(mvrv_zscore)
    
    # 1. MVRV value signal + Extreme boost
    value_signal = -mvrv_zscore
    extreme_boost = compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = value_signal + extreme_boost

    # 2. Drawdown Signal (replaces MA signal)
    drawdown_signal = compute_drawdown_signal(price_vs_max, mvrv_gradient, mvrv_zscore)

    # 3. Acceleration modifier
    accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

    # 4. Polymarket sentiment signal 
    polymarket_signal = (polymarket_sentiment - 0.5) * 0.2  

    # 5. FGI sentiment signal (Maintained but zeroed via weight)
    fgi_signal = (0.5 - fgi_sentiment) * 0.2  

    # 6. S&P 500 Macro Signal (Maintained but zeroed via weight)
    macro_signal = -np.clip(snp_vs_ma, -0.1, 0.1)

    # Combine signals strictly matching the requested 64/16/20 distribution
    combined = (
        value_signal * weights['mvrv'] + 
        drawdown_signal * weights['drawdown'] + 
        fgi_signal * weights['fgi'] +
        macro_signal * weights['snp'] + 
        polymarket_signal * weights['poly'] 
    )

    accel_modifier_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    accel_modifier_subtle = np.clip(accel_modifier_subtle, 0.85, 1.15)
    combined = combined * accel_modifier_subtle

    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,  
        1.0,  
    )
    combined = combined * confidence_boost

    volatility_dampening = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,  
    )
    combined = combined * volatility_dampening

    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

# =============================================================================
# Weight Computation API
# =============================================================================

def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
    weights: dict | None = None, 
) -> pd.Series:
    """Compute weights for a date window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    price_vs_max = _clean_array(df["price_vs_max"].values)
    mvrv_zscore = _clean_array(df["mvrv_zscore"].values)
    mvrv_gradient = _clean_array(df["mvrv_gradient"].values)

    if "mvrv_acceleration" in df.columns:
        mvrv_acceleration = _clean_array(df["mvrv_acceleration"].values)
    else:
        mvrv_acceleration = None

    if "mvrv_volatility" in df.columns:
        mvrv_volatility = _clean_array(df["mvrv_volatility"].values)
        mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)
    else:
        mvrv_volatility = None

    if "signal_confidence" in df.columns:
        signal_confidence = _clean_array(df["signal_confidence"].values)
        signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence)
    else:
        signal_confidence = None

    if "polymarket_sentiment" in df.columns:
        polymarket_sentiment = _clean_array(df["polymarket_sentiment"].values)
        polymarket_sentiment = np.where(polymarket_sentiment == 0, 0.5, polymarket_sentiment)
    else:
        polymarket_sentiment = None

    if "fgi_sentiment" in df.columns:
        fgi_sentiment = _clean_array(df["fgi_sentiment"].values)
        fgi_sentiment = np.where(fgi_sentiment == 0, 0.5, fgi_sentiment)
    else:
        fgi_sentiment = None

    if "snp_vs_ma" in df.columns:
        snp_vs_ma = _clean_array(df["snp_vs_ma"].values)
    else:
        snp_vs_ma = None

    dyn = compute_dynamic_multiplier(
        price_vs_max,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
        polymarket_sentiment,
        fgi_sentiment, 
        snp_vs_ma, 
        weights=weights, 
    )
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)

def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    weights: dict | None = None,  
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        if "mvrv_zone" in placeholder.columns:
            placeholder["mvrv_zone"] = 0
        if "mvrv_volatility" in placeholder.columns:
            placeholder["mvrv_volatility"] = 0.5
        if "signal_confidence" in placeholder.columns:
            placeholder["signal_confidence"] = 0.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights, weights=weights
    )
    return weights.reindex(full_range, fill_value=0.0)