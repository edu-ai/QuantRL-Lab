# Observation Strategies

This directory contains strategies for constructing the agent's observation space from market data and portfolio state.

## FeatureAwareObservationStrategy

The primary strategy recommended for use is `FeatureAwareObservationStrategy`.

### Design Philosophy

Standard observation strategies often apply a blanket normalization (e.g., dividing by the first price in the window) to all features. This is destructive for **stationary features** (like RSI, Oscillators, or Sentiment scores) because it erases their absolute meaning (e.g., an RSI of 70 becomes indistinguishable from an RSI of 30 if they both started at 50 relative to the window).

The `FeatureAwareObservationStrategy` solves this by classifying features into two categories and applying "Smart Normalization":

### 1. Price-like Features
**Examples:** Open, High, Low, Close, SMA, EMA, Bollinger Bands (Upper/Lower).

*   **Behavior:** These are non-stationary and unbounded.
*   **Normalization:** Relative to the first step in the lookback window.
*   **Formula:** `value_t / value_0` (The agent sees percentage changes over the window).

### 2. Stationary Features
**Examples:** RSI, STOCH, MFI, ADX, CCI, Sentiment Scores, Cyclical Time Features (`day_sin`, `month_cos`).

*   **Behavior:** These oscillate within a fixed range (e.g., 0-100 or -1 to 1).
*   **Normalization:**
    *   **Raw:** Passed through as-is (preserving the signal "RSI is Overbought").
    *   **Scaled:** Optionally scaled (e.g., divided by 100) to fit a 0-1 range for neural network stability.
*   **Formula:** `value_t` (or `value_t / 100`).

### Auto-Detection

The strategy automatically detects feature types by checking column names against a list of known keywords:
*   `RSI`, `STOCH`, `MFI`, `ADX`, `WILLR`, `CCI`
*   `sentiment`, `grade`, `rating`
*   `day_sin`, `day_cos`, `month_sin`, `month_cos` (Time Features)
*   `BB_bandwidth`, `%B`

### Observation Structure

The final observation vector is a flat concatenation of:
1.  **Market Window**: `[window_size * num_features]` (Normalized as described above)
2.  **Portfolio State**: `[9 features]`
    *   Balance Ratio
    *   Position Size Ratio
    *   Unrealized P/L %
    *   Price Position in recent range (0-1)
    *   Recent Volatility
    *   Trend Strength
    *   Risk/Reward metrics
