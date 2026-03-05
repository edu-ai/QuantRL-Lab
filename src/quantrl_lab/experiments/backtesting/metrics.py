from typing import Any, Dict, List

import numpy as np
import pandas as pd


class MetricsCalculator:
    """Calculates comprehensive financial and RL performance metrics
    from backtest episodes."""

    def calculate(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregated metrics across all episodes.

        Args:
            episodes: List of episode result dictionaries from evaluation.

        Returns:
            Dictionary of calculated metrics.
        """
        valid_episodes = [ep for ep in episodes if "error" not in ep]
        if not valid_episodes:
            return {}

        # 1. RL Metrics
        rewards = [ep["total_reward"] for ep in valid_episodes]
        lengths = [ep["steps"] for ep in valid_episodes]

        # 2. Financial Metrics (Portfolio Returns)
        # Calculate returns per episode
        returns_pct = []
        equity_curves = []

        for ep in valid_episodes:
            initial = ep.get("initial_value", 0)
            final = ep.get("final_value", 0)
            if initial != 0:
                returns_pct.append((final - initial) / initial)

            # Reconstruct equity curve if detailed actions are available
            if "detailed_actions" in ep:
                curve = [a.get("portfolio_value", 0) for a in ep["detailed_actions"]]
                if curve:
                    equity_curves.append(pd.Series(curve))

        # aggregated stats
        metrics = {
            "total_episodes": len(valid_episodes),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_episode_length": float(np.mean(lengths)),
            "avg_return_pct": float(np.mean(returns_pct)) * 100 if returns_pct else 0.0,
            "std_return_pct": float(np.std(returns_pct)) * 100 if returns_pct else 0.0,
            "win_rate": float(np.mean([r > 0 for r in returns_pct])) if returns_pct else 0.0,
        }

        # 3. Advanced Financial Metrics (Sharpe, Sortino, Drawdown)
        # We need a continuous equity curve or daily returns series to do this properly.
        # Since episodes might be disjoint, we can calculate these per episode and average them,
        # or concatenate them if they represent a continuous sequence (which they usually don't in evaluation).
        # Best approach for evaluation episodes: Average the metrics per episode.

        sharpes = []
        sortinos = []
        max_drawdowns = []

        for curve in equity_curves:
            if len(curve) < 2:
                continue

            # Calculate returns series for this episode
            rets = curve.pct_change().fillna(0)

            # Annualization factor (assuming daily data for standard envs)
            ann_factor = 252

            # Sharpe
            volatility = rets.std() * np.sqrt(ann_factor)
            ann_return = rets.mean() * ann_factor
            risk_free = 0.02

            sharpe = (ann_return - risk_free) / volatility if volatility > 0 else 0
            sharpes.append(sharpe)

            # Sortino
            downside = rets[rets < 0]
            downside_std = downside.std() * np.sqrt(ann_factor)
            sortino = (ann_return - risk_free) / downside_std if downside_std > 0 else 0
            sortinos.append(sortino)

            # Max Drawdown
            cum_max = curve.cummax()
            drawdown = (curve / cum_max) - 1
            max_dd = drawdown.min()
            max_drawdowns.append(max_dd)

        if sharpes:
            metrics["avg_sharpe_ratio"] = float(np.mean(sharpes))
            metrics["avg_sortino_ratio"] = float(np.mean(sortinos))
            metrics["avg_max_drawdown"] = float(np.mean(max_drawdowns))

        return metrics
