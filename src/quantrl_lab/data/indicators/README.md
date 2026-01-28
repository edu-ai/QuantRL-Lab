# Technical Indicators for RL Agents

This directory contains a suite of technical indicators designed to enrich the observation space of Reinforcement Learning (RL) agents. By transforming raw OHLCV (Open, High, Low, Close, Volume) data into structured features, we provide the agent with a "richer" state representation, allowing it to detect patterns, trends, and market regimes more effectively than from raw price action alone.

## Why Indicators Matter for RL

In Financial Reinforcement Learning, the **Markov Property** implies that the current state should contain all necessary information to make an optimal decision. Raw price history alone is often noisy and non-stationary. Technical indicators help by:

1.  **Stationarity**: Many indicators (RSI, CCI, ADX) are bounded or mean-reverting, making them easier for neural networks to normalize and learn from compared to unbounded raw prices.
2.  **Feature Extraction**: They pre-calculate complex relationships (momentum, volatility expansion) that a dense layer might struggle to infer from raw sequence data immediately.
3.  **Regime Detection**: Indicators like ADX or ATR help the agent distinguish between trending vs. ranging markets or high vs. low volatility environments, enabling dynamic strategy adaptation.

---

## Available Indicators

### 1. Trend Indicators
*Helping the agent identify the direction and strength of the market.*

*   **SMA (Simple Moving Average)** & **EMA (Exponential Moving Average)**
    *   **What**: Average price over a window. EMA weights recent prices more heavily.
    *   **RL Value**: Provides a baseline to measure price deviation. The relationship between Price, SMA, and EMA (e.g., crossovers) signals trend direction.
*   **MACD (Moving Average Convergence Divergence)**
    *   **What**: The difference between two EMAs (fast and slow), plus a signal line.
    *   **RL Value**: A powerful momentum and trend-follower. It helps the agent anticipate trend reversals and accelerations.
*   **ADX (Average Directional Index)**
    *   **What**: Measures the **strength** of a trend, regardless of direction.
    *   **RL Value**: Critical for "meta-decisions". A high ADX (>25) signals a strong trend (good for trend-following policies), while a low ADX suggests a ranging market (good for mean-reversion).
*   **CCI (Commodity Channel Index)**
    *   **What**: Measures the difference between the current price and its historical average deviation.
    *   **RL Value**: Identifies cyclical trends. High values indicate the price is statistically far above the mean, potentially signaling a reversion or a strong breakout.

### 2. Momentum & Oscillators
*Helping the agent identify overbought or oversold conditions.*

*   **RSI (Relative Strength Index)**
    *   **What**: Measures the speed and change of price movements on a scale of 0-100.
    *   **RL Value**: The classic mean-reversion feature. Low values (<30) suggest potential buy opportunities (oversold), while high values (>70) suggest selling pressure.
*   **Stochastic Oscillator (STOCH)**
    *   **What**: Compares a closing price to its price range over a given period.
    *   **RL Value**: Very sensitive to short-term momentum shifts. Useful for timing precise entries within a larger trend.
*   **Williams %R (WILLR)**
    *   **What**: Similar to Stochastic but on a scale of -100 to 0.
    *   **RL Value**: effectively highlights extreme market conditions. Often used to spot entry points during pullbacks in a strong trend.

### 3. Volatility Indicators
*Helping the agent understand risk and market activity levels.*

*   **ATR (Average True Range)**
    *   **What**: Decomposed measure of market volatility (absolute movement).
    *   **RL Value**: Essential for risk management. Agents can learn to adjust position sizes inversely to ATR (lower size in high volatility) or set dynamic stop-loss targets.
*   **Bollinger Bands (BB)**
    *   **What**: A set of lines plotted two standard deviations (positively and negatively) away from an SMA.
    *   **RL Value**: Measures "relative" volatility. Price touching the bands can signal a continuation (walking the band) or a reversal (rejection), depending on the regime. Bandwidth (width of bands) signals an impending breakout (squeeze).

### 4. Volume Indicators
*Helping the agent confirm price moves with liquidity.*

*   **OBV (On-Balance Volume)**
    *   **What**: Cumulative total of volume added on up days and subtracted on down days.
    *   **RL Value**: "Smart money" tracking. Divergence between Price and OBV (e.g., price rising but OBV flat) often precedes a reversal, giving the agent a leading signal.
*   **MFI (Money Flow Index)**
    *   **What**: A volume-weighted RSI.
    *   **RL Value**: Combines price and volume to identify overbought/oversold conditions more reliably than price alone.

---

## Usage Example

To add these indicators to your dataframe using the `IndicatorRegistry`:

```python
from quantrl_lab.data.indicators import IndicatorRegistry

# Apply specific indicators
df = IndicatorRegistry.apply("SMA", df, window=20)
df = IndicatorRegistry.apply("RSI", df, window=14)
df = IndicatorRegistry.apply("ADX", df, window=14)

# Or use within a VectorizedStrategy
```
