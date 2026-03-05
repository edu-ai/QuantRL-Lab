# Generalization & Overfitting in Financial RL

One of the biggest challenges in applying Reinforcement Learning (RL) to finance is **overfitting**. An agent might achieve a high Sharpe ratio on historical training data but fail completely (0% return or losses) on unseen test data.

This guide explains why this happens and how to improve generalization.

## The Problem: Memorization vs. Learning

Financial time series are **non-stationary** (statistical properties change over time) and have a **low signal-to-noise ratio**.

If you train an agent on a single stock (e.g., AAPL) for a short period (e.g., 2022-2023):
1.  The agent "memorizes" the specific price path of AAPL during that period.
2.  It learns simple heuristics like "always buy" (if the market was bullish) or "buy when price is $150" (absolute price dependence).
3.  When presented with test data (e.g., 2024), the price levels and dynamics are different. The agent's memorized rules fail, and it often defaults to doing nothing (holding cash) to avoid losing money, especially if the policy entropy has collapsed.

## Strategies for Improvement

### 1. Training on Multiple Assets
Instead of training on a single stock, train on a **universe** of stocks (e.g., S&P 500 constituents or a tech sector basket).
*   **Why?** The agent cannot memorize 50 different price paths simultaneously. It is forced to learn *general* patterns (e.g., "buy when RSI < 30 and trend is up") that apply across different assets.
*   **Implementation:** Use a Vectorized Environment (`VecEnv`) where each environment instance runs a different stock.

### 2. Feature Engineering & Stationarity
Avoid feeding raw prices (`Open`, `Close`) directly to the neural network.
*   **Problem:** Raw prices are unbounded. A price of $150 seen during training might never appear in testing if the stock splits or rallies to $300.
*   **Solution:** Use stationary features:
    *   **Returns:** `df['Close'].pct_change()`
    *   **Log Returns:** `np.log(df['Close'] / df['Close'].shift(1))`
    *   **Technical Indicators:** RSI (0-100), MACD Histogram (oscillates around 0), Bollinger Band %B.
    *   **Normalized Prices:** Divide current window prices by the *first* price in the window (relative scaling).

*QuantRL-Lab's `FeatureAwareObservationStrategy` implements this normalization automatically.*

### 3. Action Space Design
*   **Symmetric Scaling:** Ensure that an uninitialized network (outputting 0.0) maps to a neutral or exploratory action, not "Hold". (Already implemented in `StandardActionStrategy`).
*   **Curriculum Learning:** Start with a simple action space (Buy/Sell/Hold) and strictly limited position sizes. Once the agent shows promise, introduce complex order types (Limit, TIF) and variable sizing.

### 4. Reward Shaping
The standard `PortfolioValueChange` reward is sparse and noisy.
*   **Risk-Adjusted Reward:** Reward the **Sharpe Ratio** or **Sortino Ratio** over a rolling window, rather than raw returns. This penalizes volatility.
*   **Entropy Regularization:** Keep `ent_coef` high (e.g., 0.01 - 0.05) during training to force the agent to keep exploring different actions.

### 5. Walk-Forward Validation
Don't just do a single Train/Test split. Use **Walk-Forward Validation**:
1.  Train on Jan-Mar. Test on Apr.
2.  Train on Jan-Apr. Test on May.
3.  ...
This mimics how a strategy would actually be deployed and retrained.

## Debugging Checklist
If your agent gets 0% return on Test:
- [ ] **Check Exploration:** Is the agent taking actions? (Use `inspect_agent_behavior.py`).
- [ ] **Check Obs Scale:** Are observation values extremely large or small? (Normalize inputs).
- [ ] **Check Overfitting:** Is Training performance suspiciously good (Sharpe > 3.0)? If so, you need more data or regularization.
