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
MA_WINDOW = 200  # 200-day simple moving average
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
# FGI Data Loading
# =============================================================================

def load_fgi_data() -> pd.DataFrame:
    """Load the Crypto Fear and Greed Index (FGI) data.
    
    Reads the FGI CSV from the data directory, converts dates to datetime index,
    and normalizes the 0-100 score to a 0.0-1.0 scale for easier signal blending.
    
    Returns:
        DataFrame indexed by date containing 'value', 'value_classification', 
        and 'fgi_normalized'. Returns an empty DataFrame if file is not found.
    """
    # Resolve the path exactly like the polymarket function
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / "data" / "crypto_fear_and_greed_index_2019_2025.csv"
    
    if not file_path.exists():
        logging.warning(
            f"FGI data file not found at {file_path}. "
            "FGI signal will default to neutral."
        )
        return pd.DataFrame()
        
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Convert the 'date' column to proper datetime objects and remove any time data
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        
        # Ensure 'value' is numeric (coercing any weird string errors to NaN)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Normalize the 0-100 score to 0.0-1.0 for the multiplier math
        df["fgi_normalized"] = df["value"] / 100.0
        
        # Set the date as the index and sort chronologically
        df = df.set_index("date").sort_index()
        
        logging.info(
            f"FGI data loaded: {len(df)} days, "
            f"from {df.index.min().date()} to {df.index.max().date()}"
        )
        
        # Return only the relevant columns
        return df[["value", "fgi_normalized", "value_classification"]]
        
    except Exception as e:
        logging.error(f"Failed to process FGI data: {e}")
        return pd.DataFrame()

        
# =============================================================================
<<<<<<< Updated upstream
=======
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
        
        # Calculate 20-day MA
        df['snp_ma'] = df['Close'].rolling(20, min_periods=10).mean()
        
        # Calculate distance from MA (Positive = Uptrend, Negative = Downtrend)
        df['snp_vs_ma'] = (df['Close'] / df['snp_ma']) - 1.0
        
        return df[['Close', 'snp_ma', 'snp_vs_ma']]
        
    except Exception as e:
        logging.error(f"Failed to process S&P 500 data: {e}")
        return pd.DataFrame()

# =============================================================================
>>>>>>> Stashed changes
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


import logging
import pandas as pd
import numpy as np
# Assuming your zscore, classify_mvrv_zone, compute_mvrv_volatility, 
# compute_signal_confidence, load_polymarket_btc_sentiment, and load_fgi_data 
# are imported here.

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MVRV and MA features for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_ma: Normalized distance from 200-day MA, clipped to [-1, 1]
    - mvrv_zscore: MVRV Z-score (365-day window), clipped to [-4, 4]
    - mvrv_gradient: Smoothed MVRV trend direction in [-1, 1]
    - mvrv_acceleration: Second derivative of MVRV gradient (momentum)
    - mvrv_zone: Discrete zone classification [-2, -1, 0, 1, 2]
    - polymarket_sentiment: Normalized sentiment from BTC market activity [0, 1]
    - fgi_sentiment: Normalized Crypto Fear & Greed Index [0, 1]  <-- [NEW] Added to docstring

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

    # =========================================================================
    # [NEW] Load FGI Sentiment (if available)
    # =========================================================================
    try:
        fgi_df = load_fgi_data()  # Make sure this function is imported/available
        if not fgi_df.empty:
            # Reindex to match the price timeline. 
            # We use ffill() to carry forward the previous day's FGI if a day is missing,
            # then fill any remaining (like early historical data) with neutral 0.5.
            fgi_sentiment = fgi_df["fgi_normalized"].reindex(price.index).ffill().fillna(0.5)
        else:
            fgi_sentiment = pd.Series(0.5, index=price.index)
    except Exception as e:
        logging.warning(f"FGI sentiment not available: {e}")
        fgi_sentiment = pd.Series(0.5, index=price.index)

<<<<<<< Updated upstream
=======
    # =========================================================================
    # [NEW] Load S&P 500 Macro Environment
    # =========================================================================
    try:
        snp_df = load_snp_data()
        if not snp_df.empty:
            # Reindex to price, forward fill weekends/holidays, fill early history with 0.0 (neutral)
            snp_vs_ma = snp_df["snp_vs_ma"].reindex(price.index).ffill().fillna(0.0)
        else:
            snp_vs_ma = pd.Series(0.0, index=price.index)
    except Exception as e:
        logging.warning(f"S&P 500 data not available: {e}")
        snp_vs_ma = pd.Series(0.0, index=price.index)


>>>>>>> Stashed changes
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
            "fgi_sentiment": fgi_sentiment,  # <-- [NEW] Added to the main dataframe
<<<<<<< Updated upstream
=======
            "snp_vs_ma": snp_vs_ma, # <-- [NEW] Added here
>>>>>>> Stashed changes
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
        "fgi_sentiment",  # <-- [NEW] Ensures FGI is shifted by 1 day!
<<<<<<< Updated upstream
=======
        "snp_vs_ma",  # <-- [NEW] Stock market data is now safely time-shifted
>>>>>>> Stashed changes
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # Fill NaN values with appropriate defaults
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["polymarket_sentiment"] = features["polymarket_sentiment"].fillna(0.5)
    features["fgi_sentiment"] = features["fgi_sentiment"].fillna(0.5) # <-- [NEW] Clean up NaNs from the shift
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
    fgi_sentiment: np.ndarray | None = None,  # <-- [NEW] Added FGI parameter
<<<<<<< Updated upstream
=======
    snp_vs_ma: np.ndarray | None = None, # <-- [NEW] Added S&P parameter
    weights: dict | None = None,  # <-- [NEW] Add weights parameter
>>>>>>> Stashed changes
) -> np.ndarray:
    """Compute weight multiplier from MVRV, MA, and Sentiment signals.

    Enhanced strategy with Retail Contrarian weighting (Option 2):
    - Primary (60%): MVRV value signal with asymmetric extreme boost
    - Secondary (15%): MA signal with adaptive trend modulation
    - Tertiary (5%): Polymarket sentiment modifier (Smart Money)
    - Quaternary (20%): FGI sentiment modifier (Retail Contrarian) <-- [NEW]

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
        fgi_sentiment: Optional Fear & Greed Index [0, 1] <-- [NEW]

    Returns:
        Multipliers centered around 1.0
    """
<<<<<<< Updated upstream
=======
    # [NEW] Default weights if none are provided
    if weights is None:
        weights = {'mvrv': 0.50, 'ma': 0.15, 'fgi': 0.15, 'snp': 0.10, 'poly': 0.10}

>>>>>>> Stashed changes
    # Default to neutral if not provided
    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros_like(mvrv_zscore)
    if mvrv_volatility is None:
        mvrv_volatility = np.full_like(mvrv_zscore, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full_like(mvrv_zscore, 0.5)
    if polymarket_sentiment is None:
        polymarket_sentiment = np.full_like(mvrv_zscore, 0.5)
    
    # [NEW] Default FGI to neutral 0.5 (neither fear nor greed)
    if fgi_sentiment is None:
        fgi_sentiment = np.full_like(mvrv_zscore, 0.5)
<<<<<<< Updated upstream

=======
    if snp_vs_ma is None:
        snp_vs_ma = np.zeros_like(mvrv_zscore)
    
>>>>>>> Stashed changes
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

    # 5. Polymarket sentiment signal (Smart Money Follower)
    # High sentiment = slight bullish modifier. Range: [-0.1, 0.1]
    polymarket_signal = (polymarket_sentiment - 0.5) * 0.2  

    # =========================================================================
    # [NEW] 6. FGI sentiment signal (Retail Contrarian)
    # Notice the math is reversed: (0.5 - FGI) instead of (FGI - 0.5)
    # If FGI is 0.0 (Extreme Fear)  -> (0.5 - 0.0) * 0.2 = +0.1 (Buy More)
    # If FGI is 1.0 (Extreme Greed) -> (0.5 - 1.0) * 0.2 = -0.1 (Buy Less)
    # =========================================================================
    fgi_signal = (0.5 - fgi_sentiment) * 0.2  

<<<<<<< Updated upstream
    # =========================================================================
    # [UPDATED] Combine signals with Option 2 Weights
    # 60% MVRV | 15% MA | 5% Polymarket | 20% FGI
    # =========================================================================
    combined = (
        value_signal * 0.60 + 
        ma_signal * 0.15 + 
        polymarket_signal * 0.05 + 
        fgi_signal * 0.20
=======

    # 7. S&P 500 Macro Signal
    # If S&P drops below MA, snp_vs_ma is negative. Inverting it makes it a buy signal.
    # We clip it between [-0.1, 0.1] so extreme stock market crashes don't break the bot.
    macro_signal = -np.clip(snp_vs_ma, -0.1, 0.1)

    # =========================================================================
    # Combine signals ensuring weights sum perfectly to 1.0 (100%)
    # 50% + 15% + 15% + 10% + 10% = 1.0
    # =========================================================================
    # [UPDATED] Use the dynamic weights dictionary instead of hardcoded numbers
    combined = (
        value_signal * weights['mvrv'] + 
        ma_signal * weights['ma'] + 
        fgi_signal * weights['fgi'] +
        macro_signal * weights['snp'] + 
        polymarket_signal * weights['poly'] 
>>>>>>> Stashed changes
    )

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

    # Scale and clip
    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


# Note: _clean_array is imported from template.model_development_template


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
<<<<<<< Updated upstream
=======
    weights: dict | None = None,  # <-- [NEW] Catch it here
>>>>>>> Stashed changes
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

    # ==========================================
    # [NEW] Extract FGI Sentiment
    # ==========================================
    if "fgi_sentiment" in df.columns:
        fgi_sentiment = _clean_array(df["fgi_sentiment"].values)
        fgi_sentiment = np.where(fgi_sentiment == 0, 0.5, fgi_sentiment)
    else:
        fgi_sentiment = None

<<<<<<< Updated upstream
=======

    if "snp_vs_ma" in df.columns:
        snp_vs_ma = _clean_array(df["snp_vs_ma"].values)
    else:
        snp_vs_ma = None


>>>>>>> Stashed changes
    # Compute dynamic weights with enhanced features
    dyn = compute_dynamic_multiplier(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
        polymarket_sentiment,
        fgi_sentiment,  # <-- [NEW] Pass it into the multiplier here!
<<<<<<< Updated upstream
=======
        snp_vs_ma, # <-- [NEW] Hand it to the brain
	weights=weights, # <-- Pass it here
>>>>>>> Stashed changes
    )
    raw = base * dyn

    # Allocate with stability
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
<<<<<<< Updated upstream
=======
    weights: dict | None = None,  # <-- [NEW] 1. Add the parameter here
>>>>>>> Stashed changes
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

    weights = compute_weights_fast(
<<<<<<< Updated upstream
        features_df, start_date, end_date, n_past, locked_weights
=======
        features_df, start_date, end_date, n_past, locked_weights, weights=weights
>>>>>>> Stashed changes
    )
    return weights.reindex(full_range, fill_value=0.0)
