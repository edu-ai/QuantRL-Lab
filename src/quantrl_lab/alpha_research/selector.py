import uuid
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console

from quantrl_lab.alpha_research.converters import results_to_pipeline_config
from quantrl_lab.alpha_research.models import AlphaJob
from quantrl_lab.alpha_research.runner import AlphaRunner

console = Console()


class AlphaSelector:
    """Selects the best alpha factors (indicators) for a given
    dataset."""

    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        self.data = data
        self.verbose = verbose
        self.runner = AlphaRunner(verbose=False)  # Run quiet, we control output

    def suggest_indicators(
        self,
        candidates: Optional[List[Dict]] = None,
        metric: str = "sharpe_ratio",
        threshold: float = 0.0,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Test a set of candidate indicators and return the best
        performing ones.

        Args:
            candidates: List of indicator configs to test. If None, uses a default grid.
                        Format: [{"name": "RSI", "params": {"window": 14}}]
            metric: Metric to optimize (sharpe_ratio, ic, annual_return).
            threshold: Minimum value for the metric.
            top_k: Number of top indicators to return.

        Returns:
            List of indicator configurations ready for DataProcessor.
        """
        if candidates is None:
            candidates = self._get_default_candidates()

        if self.verbose:
            console.print(f"[cyan]Evaluating {len(candidates)} candidate indicators...[/cyan]")

        jobs = []
        for cand in candidates:
            # Map indicator to strategy
            strategy_config = self._map_indicator_to_strategy(cand)
            if not strategy_config:
                if self.verbose:
                    console.print(f"[yellow]Skipping {cand['name']}: No strategy mapping found.[/yellow]")
                continue

            job = AlphaJob(
                data=self.data,
                indicator_name=cand["name"],
                indicator_params=cand.get("params", {}),
                strategy_name=strategy_config["name"],
                strategy_params=strategy_config["params"],
                id=f"job_{cand['name']}_{uuid.uuid4().hex[:4]}",
            )
            jobs.append(job)

        results = self.runner.run_batch(jobs)

        # Filter by threshold before passing to converter
        above_threshold = [r for r in results if r.metrics.get(metric, -999) >= threshold]

        if self.verbose:
            console.print(f"[green]Found {len(above_threshold)} indicators above threshold {threshold}[/green]")
            for r in sorted(above_threshold, key=lambda x: x.metrics.get(metric, -999), reverse=True)[:top_k]:
                console.print(
                    f"  - {r.job.indicator_name} ({r.job.indicator_params}): {metric}={r.metrics.get(metric, 0):.4f}"
                )

        return results_to_pipeline_config(above_threshold, top_n=top_k, metric=metric)

    def _get_default_candidates(self) -> List[Dict]:
        """Return a default grid covering all registered strategies."""
        candidates = []

        # RSI — mean reversion
        for w in [7, 14, 21]:
            candidates.append({"name": "RSI", "params": {"window": w}})

        # SMA — trend following
        for w in [20, 50, 200]:
            candidates.append({"name": "SMA", "params": {"window": w}})

        # EMA — trend following (faster response than SMA)
        for w in [12, 26, 50]:
            candidates.append({"name": "EMA", "params": {"window": w}})

        # MACD — crossover
        candidates.append({"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}})
        candidates.append({"name": "MACD", "params": {"fast": 5, "slow": 35, "signal": 5}})

        # Bollinger Bands — mean reversion at bands
        candidates.append({"name": "BB", "params": {"window": 20, "num_std": 2.0}})

        # ATR — volatility breakout
        candidates.append({"name": "ATR", "params": {"window": 14}})

        # CCI — mean reversion
        candidates.append({"name": "CCI", "params": {"window": 20}})

        # Stochastic — mean reversion
        candidates.append({"name": "STOCH", "params": {"k_window": 14, "d_window": 3}})

        # OBV — volume trend
        candidates.append({"name": "OBV", "params": {}})

        # ADX — trend strength + direction
        candidates.append({"name": "ADX", "params": {"window": 14}})

        return candidates

    def _map_indicator_to_strategy(self, indicator: Dict) -> Optional[Dict]:
        """Map an indicator config to a suitable testing strategy."""
        name = indicator.get("name", "").upper()

        if name == "RSI":
            return {"name": "mean_reversion", "params": {"oversold": 30, "overbought": 70}}
        elif name in ("SMA", "EMA"):
            return {"name": "trend_following", "params": {}}
        elif name == "MACD":
            return {"name": "macd_crossover", "params": {}}
        elif name == "BB":
            return {"name": "bollinger_bands", "params": {}}
        elif name == "ATR":
            return {"name": "volatility_breakout", "params": {}}
        elif name == "CCI":
            return {"name": "cci_reversal", "params": {}}
        elif name == "STOCH":
            return {"name": "stochastic", "params": {}}
        elif name == "OBV":
            return {"name": "obv_trend", "params": {}}
        elif name == "ADX":
            return {"name": "adx_trend", "params": {}}

        return None
