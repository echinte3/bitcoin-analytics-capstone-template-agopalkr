"""Dynamic DCA weight computation using MVRV + 200-day MA + Polymarket strategy.

This module extends the template model with Polymarket sentiment integration.
It demonstrates how to:
1. Import from template modules
2. Add model-specific data loading functions
3. Integrate external data sources (Polymarket) into the feature set
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
#MA_WINDOW = 200  # 200-day simple moving average
MA_WINDOW = 50
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
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
    "polymarket_sentiment",
]


# =============================================================================
# Model-Specific Data Loading
# =============================================================================


def load_polymarket_btc_sentiment() -> pd.DataFrame:
    """Load Polymarket BTC-related markets and compute daily sentiment.
    
    This is a model-specific function that processes raw Polymarket data
    to extract BTC sentiment signals. It uses the general load_polymarket_data()
    function from template/prelude_template.py.
    
    Aggregates BTC-related prediction markets by creation date to compute:
    - daily_market_count: number of new BTC markets created each day
    - daily_volume: total volume of BTC markets created each day
    - polymarket_sentiment: normalized sentiment score [0, 1]
    
    Returns:
        DataFrame indexed by date with sentiment features.
        Returns empty DataFrame if Polymarket data not found.
    """
    # Load raw Polymarket data using the general function
    polymarket_data = load_polymarket_data()
    
    if "markets" not in polymarket_data:
        logging.warning(
            "Polymarket markets data not found. "
            "Polymarket sentiment will be neutral (0.0) for all dates."
        )
        return pd.DataFrame()
    
    markets_df = polymarket_data["markets"]
    
    # Filter to BTC-related markets
    btc_markets = markets_df[
        markets_df["question"].str.contains("Bitcoin|BTC|btc", case=False, na=False)
    ].copy()
    
    logging.info(f"Found {len(btc_markets)} BTC-related markets in Polymarket data")
    
    if btc_markets.empty:
        logging.warning("No BTC-related markets found in Polymarket data")
        return pd.DataFrame()
    
    # Extract creation date (normalize to date only)
    btc_markets["created_date"] = pd.to_datetime(btc_markets["created_at"]).dt.normalize()
    
    # Aggregate by creation date
    daily_stats = btc_markets.groupby("created_date").agg(
        daily_market_count=("market_id", "count"),
        daily_volume=("volume", "sum")
    ).reset_index()
    
    # Compute normalized sentiment score
    # High market creation activity = high sentiment
    # Use rolling 30-day percentile to normalize
    daily_stats = daily_stats.set_index("created_date").sort_index()
    
    # Compute rolling percentiles (30-day window)
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
    
    # Combine into single sentiment score (average of percentiles)
    daily_stats["polymarket_sentiment"] = (
        daily_stats["market_count_pct"] * 0.5 + daily_stats["volume_pct"] * 0.5
    )
    
    # Fill NaN with neutral (0.5)
    daily_stats["polymarket_sentiment"] = daily_stats["polymarket_sentiment"].fillna(0.5)
    
    logging.info(
        f"Polymarket sentiment computed: {len(daily_stats)} days, "
        f"{daily_stats.index.min().date()} to {daily_stats.index.max().date()}"
    )
    
    return daily_stats[["polymarket_sentiment"]]


# =============================================================================
# Helper Functions
# =============================================================================


# Note: softmax is not used in this model, removed to avoid duplication


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)


def classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Classify MVRV into discrete zones for regime detection.

    Zones:
    - -2 (deep_value): Z < -2.0 (historically rare, extreme buying opportunity)
    - -1 (value): -2.0 <= Z < -1.0 (undervalued, increase buying)
    -  0 (neutral): -1.0 <= Z < 1.5 (fair value, normal DCA)
    - +1 (caution): 1.5 <= Z < 2.5 (overvalued, reduce buying)
    - +2 (danger): Z >= 2.5 (extreme overvaluation, minimize buying)

    Args:
        mvrv_zscore: Array of MVRV Z-scores

    Returns:
        Array of zone classifications in [-2, -1, 0, 1, 2]
    """
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
    """Compute rolling volatility of MVRV Z-score.

    High volatility periods suggest uncertainty - signals should be dampened.
    Low volatility periods suggest conviction - signals can be amplified.

    Args:
        mvrv_zscore: MVRV Z-score series
        window: Rolling window for volatility calculation

    Returns:
        Normalized volatility in [0, 1] where 1 = high volatility
    """
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    # Normalize to [0, 1] using historical quantiles
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
    price_vs_ma: np.ndarray,
) -> np.ndarray:
    """Compute confidence score based on signal agreement.

    When multiple signals agree, confidence is high:
    - Low Z-score + Rising gradient = High confidence buy
    - High Z-score + Falling gradient = High confidence reduce

    Args:
        mvrv_zscore: MVRV Z-score in [-4, 4]
        mvrv_gradient: Trend direction in [-1, 1]
        price_vs_ma: Price vs MA in [-1, 1]

    Returns:
        Confidence score in [0, 1] where 1 = all signals strongly agree
    """
    # Normalize all signals to [-1, 1] where negative = buy signal
    z_signal = -mvrv_zscore / 4  # Normalize to [-1, 1]
    ma_signal = -price_vs_ma  # Below MA = buy signal

    # Gradient indicates momentum direction
    # Positive gradient with buy signals = confirmation
    # Negative gradient with buy signals = divergence (lower confidence)
    gradient_alignment = np.where(
        z_signal < 0,  # Buy signal from Z-score
        np.where(mvrv_gradient > 0, 1.0, 0.5),  # Rising = confirmation
        np.where(mvrv_gradient < 0, 1.0, 0.5),  # Falling = confirmation for sell
    )

    # Calculate agreement: how many signals point the same direction?
    signals = np.stack([z_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)

    # Low std = high agreement, high std = disagreement
    # Transform std to confidence: confidence = 1 - normalized_std
    max_std = 1.0  # Maximum possible std when signals fully disagree
    agreement = 1.0 - np.clip(signal_std / max_std, 0, 1)

    # Combine agreement with gradient alignment
    confidence = agreement * 0.7 + gradient_alignment * 0.3

    return np.clip(confidence, 0, 1)


def compute_mean_reversion_pressure(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute mean reversion pressure based on distance from equilibrium.

    MVRV tends to revert to the mean. The further from equilibrium,
    the stronger the expected reversion pressure.

    Uses a sigmoid-like function to model increasing pressure at extremes.

    Args:
        mvrv_zscore: MVRV Z-score in [-4, 4]

    Returns:
        Reversion pressure in [-1, 1] where:
        - Positive = pressure to decrease (price likely to fall)
        - Negative = pressure to increase (price likely to rise)
    """
    # Sigmoid-like pressure: increases non-linearly at extremes
    pressure = np.tanh(mvrv_zscore * 0.5)

    # Extra pressure at extremes (beyond ±2 std)
    extreme_pressure = np.where(
        np.abs(mvrv_zscore) > 2,
        np.sign(mvrv_zscore) * 0.3 * (np.abs(mvrv_zscore) - 2),
        0,
    )

    return np.clip(pressure + extreme_pressure, -1, 1)


# =============================================================================
# Feature Engineering
# =============================================================================


def load_snp_data():
    #sp_file = Path.cwd() / "data/SP500.csv"
    sp_file = "C:/monu/georgiatech/practicum/trilemma/bitcoin/bitcoin-analytics-capstone-template/data2/SP500.csv"
    df_sp500 = pd.read_csv(sp_file)
    df_sp500['Date']=pd.to_datetime(df_sp500['Date'])
    df_sp500 = df_sp500.set_index('Date')
    df_sp500.index = df_sp500.index.normalize()
    roll_mean=df_sp500['Close'].rolling(20).mean().dropna().to_frame()
    df_sp500=pd.merge(df_sp500, roll_mean,left_index=True, right_index=True)
    df_sp500=df_sp500.rename(columns={"Close_x": "Close", "Close_y": "MA"})
    print(df_sp500.head(2))
    return df_sp500

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MVRV and MA features for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_ma: Normalized distance from 200-day MA, clipped to [-1, 1]
    - mvrv_zscore: MVRV Z-score (365-day window), clipped to [-4, 4]
    - mvrv_gradient: Smoothed MVRV trend direction in [-1, 1]
    - mvrv_acceleration: Second derivative of MVRV gradient (momentum)
    - mvrv_zone: Discrete zone classification [-2, -1, 0, 1, 2]
    - polymarket_sentiment: Normalized sentiment from BTC market activity [0, 1]

    Args:
        df: DataFrame with price and MVRV columns

    Returns:
        DataFrame with price and computed features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 200-day MA and distance
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    # MVRV features
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]

        # Core Z-score (365-day window)
        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)

        # Smoothed gradient using EMA
        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(
            span=MVRV_GRADIENT_WINDOW, adjust=False
        ).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)

        # MVRV acceleration (second derivative - momentum detection)
        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean()
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3).fillna(0)

        # Zone classification
        mvrv_zone = pd.Series(
            classify_mvrv_zone(mvrv_z.values),
            index=mvrv_z.index,
        )

        # MVRV volatility (for signal dampening in uncertain periods)
        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)

        # Signal confidence (computed after lag, using lagged values)
        # Will be computed after lag is applied
        signal_confidence = pd.Series(0.5, index=price.index)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        signal_confidence = pd.Series(0.5, index=price.index)

    # Load Polymarket sentiment (if available)
    try:
        polymarket_df = load_polymarket_btc_sentiment()
        if not polymarket_df.empty:
            # Merge with price index, fill missing dates with neutral (0.5)
            polymarket_sentiment = polymarket_df["polymarket_sentiment"].reindex(
                price.index, fill_value=0.5
            )
        else:
            polymarket_sentiment = pd.Series(0.5, index=price.index)
    except (ImportError, FileNotFoundError, Exception) as e:
        # If Polymarket data not available, use neutral sentiment
        logging.warning(f"Polymarket sentiment not available: {e}")
        polymarket_sentiment = pd.Series(0.5, index=price.index)

    # Build and lag features
    features = pd.DataFrame(
        {
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
        },
        index=price.index,
    )

    # Lag signals by 1 day to prevent look-ahead bias
    signal_cols = [
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
        "polymarket_sentiment",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # Fill NaN values with appropriate defaults
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["polymarket_sentiment"] = features["polymarket_sentiment"].fillna(0.5)
    features = features.fillna(0)

    # Compute signal confidence using lagged values (no look-ahead)
    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    return features


# =============================================================================
# Weight Allocation
# =============================================================================


# Note: _compute_stable_signal and allocate_sequential_stable are imported from template.model_development_template


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_asymmetric_extreme_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute asymmetric boost for extreme MVRV values.

    Key insight: Bitcoin's MVRV is asymmetric - extreme lows are rare buying
    opportunities, while extreme highs often precede corrections.

    Behavior:
    - Z < -2: Strong positive boost (aggressive accumulation)
    - Z < -1: Moderate positive boost (value buying)
    - -1 <= Z <= 1.5: No boost (neutral zone)
    - Z > 1.5: Negative boost (reduce buying)
    - Z > 2.5: Strong negative boost (minimize buying)

    Args:
        mvrv_zscore: Array of MVRV Z-scores in [-4, 4]

    Returns:
        Boost values (positive = increase buying, negative = reduce buying)
    """
    boost = np.zeros_like(mvrv_zscore)

    # Deep undervaluation: strong positive boost (quadratic increase)
    deep_value_mask = mvrv_zscore < MVRV_ZONE_DEEP_VALUE
    boost = np.where(
        deep_value_mask,
        0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5,
        boost,
    )

    # Moderate undervaluation: linear positive boost
    value_mask = (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE)
    boost = np.where(
        value_mask,
        -0.5 * mvrv_zscore,  # Linear boost proportional to undervaluation
        boost,
    )

    # Caution zone: moderate negative boost
    caution_mask = (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER)
    boost = np.where(
        caution_mask,
        -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION),
        boost,
    )

    # Danger zone: strong negative boost (quadratic decrease)
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
    """Compute modifier based on MVRV acceleration (momentum).

    Acceleration helps identify:
    - Momentum building in current direction
    - Potential trend reversals

    Args:
        mvrv_acceleration: Second derivative of MVRV gradient
        mvrv_gradient: First derivative (trend direction)

    Returns:
        Modifier in [0.5, 1.5] to scale other signals
    """
    # Same-direction acceleration: momentum building
    # Opposite-direction acceleration: potential reversal
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0

    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),  # Amplify if momentum building
        1.0 - 0.2 * np.abs(mvrv_acceleration),  # Dampen if potential reversal
    )

    return np.clip(modifier, 0.5, 1.5)


def compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray,
    mvrv_zscore: np.ndarray,
) -> np.ndarray:
    """Compute trend modifier with adaptive thresholds.

    Instead of fixed 0.2/-0.2 thresholds, adapts based on current MVRV level:
    - In deep value: be more aggressive with dip buying
    - In overvalued territory: be more conservative

    Args:
        mvrv_gradient: MVRV trend direction in [-1, 1]
        mvrv_zscore: Current MVRV Z-score

    Returns:
        Trend modifier for MA signal
    """
    # Adaptive thresholds based on MVRV level
    # Lower threshold in value zone (more sensitive to reversals)
    # Higher threshold in danger zone (require stronger confirmation)
    threshold = np.where(
        mvrv_zscore < -1,
        0.1,  # Low threshold in value zone
        np.where(mvrv_zscore > 1.5, 0.4, 0.2),  # High threshold in danger zone
    )

    # Compute modifier
    modifier = np.where(
        mvrv_gradient > threshold,
        1.0 + 0.5 * np.minimum(mvrv_gradient, 1.0),  # Bull: up to 1.5x
        np.where(
            mvrv_gradient < -threshold,
            0.3 + 0.2 * (1 + mvrv_gradient),  # Bear: down to 0.3x
            1.0,  # Neutral
        ),
    )

    return np.clip(modifier, 0.3, 1.5)


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
    polymarket_sentiment: np.ndarray | None = None,
    sp500_price: np.ndarray | None = None,
    ma500_price: np.ndarray | None = None,

) -> np.ndarray:
    """Compute weight multiplier from MVRV and MA signals.

    Enhanced strategy with multiple MVRV signals:
    - Primary (64%): MVRV value signal with asymmetric extreme boost
    - Secondary (16%): MA signal with adaptive trend modulation
    - Tertiary (20%): Polymarket sentiment modifier

    Modulated by:
    - Signal confidence: Amplify when signals agree
    - Volatility: Dampen in high uncertainty periods

    Args:
        price_vs_ma: Distance from 200-day MA in [-1, 1]
        mvrv_zscore: MVRV Z-score in [-4, 4]
        mvrv_gradient: MVRV trend direction in [-1, 1]
        mvrv_acceleration: Optional MVRV acceleration [-1, 1]
        mvrv_volatility: Optional volatility percentile [0, 1]
        signal_confidence: Optional confidence score [0, 1]
        polymarket_sentiment: Optional Polymarket sentiment [0, 1]

    Returns:
        Multipliers centered around 1.0
    """
    # Default to neutral if not provided
    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros_like(mvrv_zscore)
    if mvrv_volatility is None:
        mvrv_volatility = np.full_like(mvrv_zscore, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full_like(mvrv_zscore, 0.5)
    if polymarket_sentiment is None:
        polymarket_sentiment = np.full_like(mvrv_zscore, 0.5)

    # 1. MVRV value signal: low MVRV = buy more
    value_signal = -mvrv_zscore

    # 2. Asymmetric extreme boost (corrected sign logic)
    extreme_boost = compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = value_signal + extreme_boost

    # 3. MA signal: buy when below MA, with adaptive trend modulation
    ma_signal = -price_vs_ma
    trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = ma_signal * trend_modifier

    # 4. Acceleration modifier: momentum detection
    accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

    # 5. Polymarket sentiment signal: high sentiment = slight bullish modifier
    # Normalize from [0, 1] to [-0.1, 0.1] for subtle effect
    polymarket_signal = (polymarket_sentiment - 0.5) * 0.2  # Range: [-0.1, 0.1]

    # Combine signals with weights
    # Primary: MVRV value (64%), Secondary: MA (16%), Tertiary: Polymarket (20%)
    # Focus on core MVRV signal with asymmetric boost
    # combined = value_signal * 0.64 + ma_signal * 0.16 + polymarket_signal * 0.20
    arr = ma500_price/sp500_price
    arr[np.isnan(arr)] = 1
    # 60%
    #combined = value_signal * 0.30 + ma_signal * 0.1 + polymarket_signal * 0.10 + arr * 0.6
    # also arund 60%
    #combined = value_signal * 0.40 + ma_signal * 0.05 + polymarket_signal * 0.05 + arr * 0.5

    #Not ok
    #combined = value_signal * 0.15 + ma_signal * 0.08 + polymarket_signal * 0.02 + arr * 0.75

    #Not much diff
    combined = (arr**0.5)*(value_signal * 0.80 + ma_signal * 0.20) + (1-arr*0.5)*polymarket_signal * 0.20

    # Apply acceleration modifier (subtle: range [0.85, 1.15])
    accel_modifier_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    accel_modifier_subtle = np.clip(accel_modifier_subtle, 0.85, 1.15)
    combined = combined * accel_modifier_subtle

    # Confidence boost only when very high (> 0.7), otherwise neutral
    # This prevents dampening when signals disagree
    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,  # Up to 1.15x
        1.0,  # Neutral otherwise
    )
    combined = combined * confidence_boost

    # Volatility dampening only in extreme volatility (top 20%)
    # This prevents over-dampening in normal conditions
    volatility_dampening = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,  # No dampening for normal volatility
    )
    combined = combined * volatility_dampening
    arr = ma500_price/sp500_price
    arr[np.isnan(arr)] = 0

    #combined = combined * (arr)**0.8
    #print('RATIO-->',arr)
    #combined = combined * (sp500_price/ma500_price)**1

    # Scale and clip
    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)

    multiplier = np.exp(adjustment)

    multiplier = multiplier * arr

    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


# Note: _clean_array is imported from template.model_development_template



import math
def computeQty(df):
    qty=0
    #up_days=30
    #down_days=30
    up_days=30
    down_days=15
    traded=False
    upTrend=False
    downTrend=False
    i=0
    n=len(df)
    print('n',n)
    cnt=0
    lst=[]
    AMT=10000
    buy_dates=[]
    for row in df.itertuples(index=True): # index=False excludes the index from the tuple
        #print(row.PriceUSD, row.MA_PriceUSD)
        qty_for_day = 1e-6
        if math.isnan(row.MA_PriceUSD):
            #print('NAN')
            lst.append(qty_for_day)
            continue
        if traded==False:
            if row.PriceUSD_coinmetrics > row.MA_PriceUSD:
                qty_for_day = (up_days*(AMT/n))/row.PriceUSD_coinmetrics
                traded=True
                upTrend=True
                downTrend=False
                #print(row.Index)
                buy_dates.append(row.Index)
            elif row.PriceUSD_coinmetrics < row.MA_PriceUSD:
                traded=True
                downTrend=True
                upTrend=False
        else:
            if upTrend == True:
                cnt+=1
            elif downTrend == True:
                cnt+=1

            if cnt >= up_days and upTrend:
                upTrend = False
                traded = False
                cnt=0
            if cnt >= down_days and downTrend:
                qty_for_day = (down_days *(AMT/n))/row.PriceUSD_coinmetrics
                downTrend=False
                traded=False
                cnt=0
                buy_dates.append(row.Index)
        lst.append(qty_for_day)
    lst = lst/np.sum(lst)
    return (lst,buy_dates)

def computeQtyLSTM(df, buy_pts):
    import sys
    lst=[]
    #for row in df.itertuples(index=True): # index=False excludes the index from the tuple
    #    #print(row.PriceUSD, row.MA_PriceUSD)
    #    qty_for_day = 1e-6
    #    #print('row',row)
    #    if math.isnan(row.MA_PriceUSD):
    #        #print('NAN')
    #        lst.append(qty_for_day)
    #        continue
    #    lst.append(qty_for_day)
    cnt=0
    prev_pt = 0

    map_buy_pts = dict()
    for pt in buy_pts:
        map_buy_pts[pt] = pt

    AMT=10000
    n=len(df)
    dates=[]
    #print(map_buy_pts)
    for row in df.itertuples(index=True): # index=False excludes the index from the tuple
        #print(row.PriceUSD, row.MA_PriceUSD)
        qty_for_day = 1e-6
        if cnt in map_buy_pts:
            price = row.PriceUSD_coinmetrics
            try:
                qty_for_day = (cnt-prev_pt)*(AMT/n)/row.PriceUSD_coinmetrics
            except ZeroDivisionError:
                print('Exception', row, row.Index)
                qty_for_day = 1e-6
                #dates.append(row.Index)
                ##sys.exit()
                #continue
            prev_pt = cnt
        dates.append(row.Index)

        cnt=cnt+1
        lst.append(qty_for_day)

    lst = lst/np.sum(lst)
    #print('dates=',dates)
    #sys.exit()
    return lst


def computeQty2(df):
    qty=0
    #up_days=30
    #down_days=30
    up_days=30
    down_days=15
    traded=False
    upTrend=False
    downTrend=False
    i=0
    n=len(df)
    print('n',n)
    cnt=0
    lst=[]
    AMT=10000
    buy_dates=[]
    for row in df.itertuples(index=True): # index=False excludes the index from the tuple
        #print(row.PriceUSD, row.MA_PriceUSD)
        qty_for_day = 1e-6
        if math.isnan(row.MA_PriceUSD):
            #print('NAN')
            lst.append(qty_for_day)
            continue
        if traded==False:
            if row.MA_PriceUSD2 > row.MA_PriceUSD:
                qty_for_day = (up_days*(AMT/n))/row.PriceUSD_coinmetrics
                traded=True
                upTrend=True
                downTrend=False
                #print(row.Index)
                buy_dates.append(row.Index)
            elif row.MA_PriceUSD2 < row.MA_PriceUSD:
                traded=True
                downTrend=True
                upTrend=False
        else:
            if upTrend == True:
                cnt+=1
            elif downTrend == True:
                cnt+=1

            if cnt >= up_days and upTrend:
                upTrend = False
                traded = False
                cnt=0
            if cnt >= down_days and downTrend:
                qty_for_day = (down_days *(AMT/n))/row.PriceUSD_coinmetrics
                downTrend=False
                traded=False
                cnt=0
                buy_dates.append(row.Index)
        lst.append(qty_for_day)
    lst = lst/np.sum(lst)
    return (lst,buy_dates)



def compute_weights_fast_1(
        _lstm_model,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_past: int | None = None,
        locked_weights: np.ndarray | None = None,

) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """

    #global _lstm_model

    print('In compute_weights_fast', _lstm_model)

    df = features_df.loc[start_date:end_date].copy()
    #window=30
    # Very good performance
    window=15
    #df_small = df_btc_2025[['PriceUSD']]
    #ma1 = df['PriceUSD'].shift().rolling(window, min_periods=window // 2).mean().to_frame()


    # 1. Shift the data by one day (or row)
    # This aligns the previous day's value to the current row.
    #shifted_value = df['PriceUSD_coinmetrics'].shift(1)
    shifted_value = df['PriceUSD_coinmetrics'].shift(3)

    df['MA_PriceUSD'] = shifted_value.rolling(window=window, min_periods=window//2).mean()
    df['MA_PriceUSD2'] = shifted_value.rolling(window=window//2, min_periods=window//4).mean()

    print('CHANGE HERE df-->', df.shape)

    if df.empty:
        return pd.Series(dtype=float)

    #l = computeQty(df)
    l = computeQty2(df)
    print('Lengh of l',len(l[0]))
    return pd.Series(l[0], index=df.index)

def create_sequences(data,  window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 0])
        #y.append(df.iloc[i+window]['PriceUSD'])
    return np.array(X), np.array(y)


def compute_weights_fast(
        _lstm_model,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_past: int | None = None,
        locked_weights: np.ndarray | None = None,

) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """

    #global _lstm_model

    #print('In compute_weights_fast', _lstm_model)

    df_inp = features_df.loc[start_date:end_date].copy()

    if df_inp.empty:
        return pd.Series(dtype=float)


    df_small = df_inp[['PriceUSD_coinmetrics']].copy()
    df_small['Momentum'] = df_small['PriceUSD_coinmetrics'].diff().copy()
    df_small['Acceleration'] = df_small['Momentum'].diff().copy()
    df = df_small.copy()
    #prices = df['PriceUSD_coinmetrics'].values

    # 3. Features
    cols= ['MA5','MA20','Momentum','MomentumMA','Acceleration','Volatility']
    df['MA5'] = df['PriceUSD_coinmetrics'].rolling(5).mean()
    df['MA20'] = df['PriceUSD_coinmetrics'].rolling(20).mean()
    df['MomentumMA'] = df['Momentum'].rolling(10).mean()
    df['AccelerationMA'] = df['Acceleration'].rolling(10).mean()
    df['Volatility'] = df['PriceUSD_coinmetrics'].rolling(10).std()
    df_shifted=df.shift(1)[cols]

    df = df_inp[['PriceUSD_coinmetrics']].join(df_shifted)
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X_test, y_test = create_sequences(scaled_data, 20)
    pred = _lstm_model.predict(X_test)

    pred_inv = scaler.inverse_transform(
        np.c_[pred, np.zeros((len(pred), df.shape[1] - 1))]
    )[:, 0]

    upTrend=False
    downTrend=False
    buy_pts=[]
    for i in range(len(pred_inv)):
        #print(i)
        if i == 0:
            continue
        if pred_inv[i] > pred_inv[i-1]:
            if downTrend == True:
                buy_pts.append(i)
            upTrend=True
            downTrend=False
        elif pred_inv[i] < pred_inv[i-1]:
            downTrend=True
            upTrend=False


    #print('buy_pts:',buy_pts)
    #l = computeQty(df)
    l = computeQtyLSTM(df_inp, buy_pts)
    print('Length of l',len(l))
    return pd.Series(l, index=df_inp.index)



def compute_weights_fast_forward_guessing(
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_past: int | None = None,
        locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """
    df = features_df.loc[start_date:end_date].copy()
    #window=30
    # Very good performance
    window=15
    #df_small = df_btc_2025[['PriceUSD']]
    #ma1 = df['PriceUSD'].shift().rolling(window, min_periods=window // 2).mean().to_frame()


    # 1. Shift the data by one day (or row)
    # This aligns the previous day's value to the current row.
    #shifted_value = df['PriceUSD_coinmetrics'].shift(1)
    shifted_value = df['PriceUSD_coinmetrics'].shift(3)

    df['MA_PriceUSD'] = shifted_value.rolling(window=window, min_periods=window//2).mean()
    df['MA_PriceUSD2'] = shifted_value.rolling(window=window//2, min_periods=window//4).mean()

    print('CHANGE HERE df-->', df.shape)

    if df.empty:
        return pd.Series(dtype=float)

    #l = computeQty(df)
    l = computeQty2(df)
    print('Lengh of l',len(l[0]))
    return pd.Series(l[0], index=df.index)



# NEED TO CHANGE THIS
# CHANGE THIS TO RETURN NEW LOGIC
def compute_weights_fast_Orig(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """
    df = features_df.loc[start_date:end_date]
    print('CHANGE HERE df-->', df.shape)

    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract and clean features
    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    mvrv_zscore = _clean_array(df["mvrv_zscore"].values)
    mvrv_gradient = _clean_array(df["mvrv_gradient"].values)

    # Extract new features if available
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

    sp500_price = _clean_array(df["Close"].values)
    ma500_price = _clean_array(df["MA"].values)

    # Compute dynamic weights with enhanced features
    dyn = compute_dynamic_multiplier(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
        polymarket_sentiment,
        sp500_price,
        ma500_price,
    )
    raw = base * dyn


    #print('MY METHOD 2 ---->', raw.shape)

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    print('CHANGE HERE WEIGHTS-->', weights.head(2))

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    _lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Two modes:
    1. BACKTEST (locked_weights=None): Signal-based allocation
    2. PRODUCTION (locked_weights provided): DB-backed stability

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Investment window start
        end_date: Investment window end
        current_date: Current date (past/future boundary)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights summing to 1.0
    """

    #global _lstm_model

    print('start',start_date,'end=',end_date,'shape=',features_df.shape)
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        # Set appropriate defaults for new features
        if "mvrv_zone" in placeholder.columns:
            placeholder["mvrv_zone"] = 0
        if "mvrv_volatility" in placeholder.columns:
            placeholder["mvrv_volatility"] = 0.5
        if "signal_confidence" in placeholder.columns:
            placeholder["signal_confidence"] = 0.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    #GOPANANT
    weights = compute_weights_fast(
        _lstm_model, features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
