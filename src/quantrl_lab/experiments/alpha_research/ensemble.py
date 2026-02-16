from typing import List

import numpy as np
import pandas as pd

from .core import AlphaResult


class AlphaEnsemble:
    """Combines multiple alpha strategies into a single portfolio
    signal."""

    def __init__(self, results: List[AlphaResult]):
        """
        Initialize with a list of alpha results.

        Only completed jobs with valid equity curves are used.
        """
        self.results = [r for r in results if r.status == "completed" and r.equity_curve is not None]
        if not self.results:
            raise ValueError("No valid alpha results provided for ensembling.")

    def combine(self, method: str = "equal_weight") -> pd.Series:
        """
        Combine signals using the specified method.

        Args:
            method (str): "equal_weight", "inverse_volatility", "ic_weighted"

        Returns:
            pd.Series: Combined equity curve (normalized to start at 1.0)
        """
        if method == "equal_weight":
            return self._equal_weight()
        elif method == "inverse_volatility":
            return self._inverse_volatility()
        elif method == "ic_weighted":
            return self._ic_weighted()
        elif method == "sharpe_weighted":
            return self._sharpe_weighted()
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def _get_returns_matrix(self) -> pd.DataFrame:
        """Helper to get a DataFrame of daily returns for all
        strategies."""
        returns_dict = {}
        for i, r in enumerate(self.results):
            # Use index as key to ensure column order matches self.results
            returns_dict[i] = r.equity_curve.pct_change().fillna(0)

        return pd.DataFrame(returns_dict)

    def _equal_weight(self) -> pd.Series:
        """Combine strategies with equal weights."""
        returns_df = self._get_returns_matrix()
        # Average daily returns across all strategies (rebalanced daily)
        combined_returns = returns_df.mean(axis=1)
        # Reconstruct equity curve starting at 1.0
        equity = (1 + combined_returns).cumprod()
        return equity / equity.iloc[0]

    def _inverse_volatility(self) -> pd.Series:
        """
        Combine strategies inversely proportional to their volatility
        (Risk Parity).

        Less volatile strategies get higher weight.
        """
        returns_df = self._get_returns_matrix()
        vols = returns_df.std()

        # Avoid division by zero if volatility is 0
        vols = vols.replace(0, np.inf)
        inv_vols = 1 / vols

        total_inv_vol = inv_vols.sum()
        if total_inv_vol == 0:
            # Fallback to equal weight if all volatilities are infinite (or effectively 0 contribution)
            return self._equal_weight()

        # Normalize weights to sum to 1
        weights = inv_vols / total_inv_vol

        # Calculate weighted average returns
        combined_returns = returns_df.dot(weights)

        equity = (1 + combined_returns).cumprod()
        return equity / equity.iloc[0]

    def _ic_weighted(self) -> pd.Series:
        """
        Combine strategies based on their Information Coefficient (IC).

        Strategies with higher predictive power get higher weight.
        """
        returns_df = self._get_returns_matrix()

        ics = []
        for r in self.results:
            ic = r.metrics.get("ic", 0)
            if pd.isna(ic):
                ic = 0
            # Clip negative ICs to 0 (we only want positive predictive power)
            ics.append(max(0, ic))

        weights = np.array(ics)

        if weights.sum() == 0:
            # Fallback to equal weight if all ICs are <= 0
            return self._equal_weight()

        weights = weights / weights.sum()

        # Calculate weighted average returns
        combined_returns = returns_df.dot(weights)

        equity = (1 + combined_returns).cumprod()
        return equity / equity.iloc[0]

    def _sharpe_weighted(self) -> pd.Series:
        """Combine strategies based on their historical Sharpe Ratio."""
        returns_df = self._get_returns_matrix()

        sharpes = []
        for r in self.results:
            s = r.metrics.get("sharpe_ratio", 0)
            if pd.isna(s):
                s = 0
            # Clip negative Sharpe to 0
            sharpes.append(max(0, s))

        weights = np.array(sharpes)

        if weights.sum() == 0:
            return self._equal_weight()

        weights = weights / weights.sum()

        combined_returns = returns_df.dot(weights)
        equity = (1 + combined_returns).cumprod()
        return equity / equity.iloc[0]
