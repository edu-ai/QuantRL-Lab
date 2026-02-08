from __future__ import annotations

from typing import TYPE_CHECKING, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol

from quantrl_lab.environments.core.interfaces import BaseObservationStrategy
from quantrl_lab.environments.utils import calc_trend


class FeatureAwareObservationStrategy(BaseObservationStrategy):
    """
    Feature-aware observation strategy with smart normalization.

    Unlike the standard strategy which normalizes everything relative to the
    window start, this strategy discriminates between feature types:
    1. Price-like (Open, High, Low, Close, SMA, EMA, BB): Normalized relative to the first step in the window.
    2. Stationary (RSI, STOCH, MFI, ADX, Time Features): Passed through raw or scaled
       independently, preserving their absolute values (e.g., Overbought/Oversold levels).
    """

    NUM_PORTFOLIO_FEATURES = 9

    def __init__(
        self,
        volatility_lookback: int = 10,
        trend_lookback: int = 10,
        normalize_stationary: bool = True,
    ):
        """
        Args:
            volatility_lookback: Steps to calculate recent volatility.
            trend_lookback: Steps to calculate trend.
            normalize_stationary: If True, attempts to scale known 0-100 indicators to 0-1.
        """
        super().__init__()
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.normalize_stationary = normalize_stationary

        # Keywords to identify stationary features that shouldn't be relatively normalized
        self.stationary_keywords = {
            "RSI",
            "STOCH",
            "MFI",
            "ADX",
            "WILLR",
            "CCI",
            "BOP",
            "CMO",
            "AROON",
            "ULTOSC",
            "sentiment",
            "grade",
            "rating",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "time_features",
            "BB_bandwidth",
            "BB_percent",
            "%B",
        }

        # Cache for column indices
        self._price_cols_idx: List[int] = []
        self._stationary_cols_idx: List[int] = []
        self._initialized_indices = False

    def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Box:
        obs_market_shape = env.window_size * env.num_features
        total_obs_dim = obs_market_shape + self.NUM_PORTFOLIO_FEATURES
        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def _identify_columns(self, env: TradingEnvProtocol):
        """Identify which columns are stationary vs price-like based on
        names."""
        if self._initialized_indices:
            return

        # Default: All price-like if no names available (fallback to old behavior)
        if not hasattr(env, "original_columns") or env.original_columns is None:
            self._price_cols_idx = list(range(env.num_features))
            self._stationary_cols_idx = []
            self._initialized_indices = True
            return

        self._price_cols_idx = []
        self._stationary_cols_idx = []

        for i, col_name in enumerate(env.original_columns):
            is_stationary = False
            col_upper = col_name.upper()

            # Check against keywords
            for kw in self.stationary_keywords:
                if kw.upper() in col_upper:
                    is_stationary = True
                    break

            # Special case: BB bands themselves are price-like, but bandwidth/%B are stationary
            # The keyword check above handles bandwidth/%B, but let's be explicit about exclusion
            # if needed. For now, the keyword set covers the stationary ones.

            if is_stationary:
                self._stationary_cols_idx.append(i)
            else:
                self._price_cols_idx.append(i)

        self._initialized_indices = True

    def build_observation(self, env: TradingEnvProtocol) -> np.ndarray:
        # Lazy init of indices
        self._identify_columns(env)

        # === 1. Market Window Extraction ===
        start_idx = max(0, env.current_step - env.window_size + 1)
        end_idx = env.current_step + 1

        # Get raw window
        raw_window = env.data[start_idx:end_idx, :]

        # Padding if needed (at start of episode)
        actual_len = raw_window.shape[0]
        if actual_len < env.window_size:
            if actual_len > 0:
                padding = np.repeat(raw_window[0, :][np.newaxis, :], env.window_size - actual_len, axis=0)
            else:
                padding = np.zeros((env.window_size - actual_len, env.num_features), dtype=env.data.dtype)
            raw_window = np.concatenate((padding, raw_window), axis=0)

        # === 2. Smart Normalization ===
        normalized_window = raw_window.copy()

        # A. Normalize Price-like columns (Relative to first step in window)
        # Formula: value / first_value
        if self._price_cols_idx:
            price_subset = raw_window[:, self._price_cols_idx]
            first_step_prices = price_subset[0, :]

            # Avoid division by zero
            denominator = np.where(np.abs(first_step_prices) < 1e-9, 1.0, first_step_prices)

            norm_prices = price_subset / denominator
            # Zero out where denominator was effectively zero (to avoid massive explosions)
            norm_prices[:, np.abs(first_step_prices) < 1e-9] = 0.0

            normalized_window[:, self._price_cols_idx] = norm_prices

        # B. Normalize Stationary columns
        # Formula: Raw value (or scaled / 100 for indicators)
        if self._stationary_cols_idx:
            stationary_subset = raw_window[:, self._stationary_cols_idx]

            if self.normalize_stationary:
                # Heuristic scaling for 0-100 indicators
                if hasattr(env, "original_columns"):
                    for local_idx, global_idx in enumerate(self._stationary_cols_idx):
                        col_name = env.original_columns[global_idx].upper()
                        # Scale 0-100 indicators to 0-1
                        if any(x in col_name for x in ["RSI", "STOCH", "MFI", "ADX", "WILLR", "CCI"]):
                            # WILLR is -100 to 0, others 0 to 100. CCI is unbounded.
                            # Simple division by 100 helps keep magnitude similar to normalized prices (~1.0)
                            stationary_subset[:, local_idx] = stationary_subset[:, local_idx] / 100.0

            normalized_window[:, self._stationary_cols_idx] = stationary_subset

        # === 3. Portfolio State (Standard Logic) ===
        current_price = env._get_current_price()
        total_shares = env.portfolio.total_shares

        position_size_ratio, unrealized_pl_pct, risk_reward_ratio, dist_stop, dist_target = 0.0, 0.0, 0.0, 0.0, 0.0

        if total_shares > 0:
            portfolio_value = env.portfolio.get_value(current_price)
            if portfolio_value > 1e-9:
                position_size_ratio = (total_shares * current_price) / portfolio_value

            entry_prices = [
                o["price"]
                for o in env.portfolio.executed_orders_history
                if o["type"] in ["market_buy", "limit_buy_executed"]
            ]
            avg_entry = np.mean(entry_prices) if entry_prices else current_price

            if avg_entry > 1e-9:
                unrealized_pl_pct = (current_price - avg_entry) / avg_entry

            # Risk metrics
            sl_orders = env.portfolio.stop_loss_orders
            tp_orders = env.portfolio.take_profit_orders
            if sl_orders and tp_orders:
                avg_sl = np.mean([o.price for o in sl_orders])
                avg_tp = np.mean([o.price for o in tp_orders])
                if abs(current_price - avg_sl) > 1e-9:
                    risk_reward_ratio = (avg_tp - current_price) / (current_price - avg_sl)
                if current_price > 1e-9:
                    dist_stop = (current_price - avg_sl) / current_price
                    dist_target = (avg_tp - current_price) / current_price

        # === 4. Feature Engineering (Volatility & Trend) ===
        recent_slice = env.data[max(0, env.current_step - self.volatility_lookback + 1) : end_idx]
        price_col = env.price_column_index

        recent_high = np.max(recent_slice[:, price_col]) if len(recent_slice) > 0 else current_price
        recent_low = np.min(recent_slice[:, price_col]) if len(recent_slice) > 0 else current_price

        # Price position within recent range (0.0 to 1.0)
        price_range = recent_high - recent_low
        price_pos = (current_price - recent_low) / price_range if price_range > 1e-9 else 0.5

        volatility = 0.0
        if len(recent_slice) > 1:
            rets = np.diff(recent_slice[:, price_col]) / recent_slice[:-1, price_col]
            volatility = np.std(rets)

        trend = calc_trend(env, self.trend_lookback)

        portfolio_features = np.array(
            [
                env.portfolio.balance / env.portfolio.initial_balance,
                position_size_ratio,
                unrealized_pl_pct,
                price_pos,
                volatility,
                trend,
                risk_reward_ratio,
                dist_stop,
                dist_target,
            ],
            dtype=np.float32,
        )

        return np.concatenate((normalized_window.flatten(), portfolio_features))
