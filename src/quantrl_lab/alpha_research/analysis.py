import copy
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

from quantrl_lab.alpha_research.models import AlphaJob, AlphaResult
from quantrl_lab.alpha_research.runner import AlphaRunner

console = Console()


class RobustnessTester:
    """
    Tools for testing the robustness of alpha strategies.

    Includes Parameter Sensitivity Analysis and Sub-period Analysis.
    """

    def __init__(self, runner: Optional[AlphaRunner] = None):
        self.runner = runner or AlphaRunner(verbose=False)

    def parameter_sensitivity(
        self,
        base_job: AlphaJob,
        param_grid: Dict[str, List[Any]] = None,
        indicator_param_grid: Dict[str, List[Any]] = None,
        strategy_param_grid: Dict[str, List[Any]] = None,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Run a grid search over indicator and/or strategy parameters.

        Parameters can be supplied in two ways:

        * **Explicit split** (recommended): pass ``indicator_param_grid``
          and/or ``strategy_param_grid`` so each key is routed to the
          correct ``AlphaJob`` dict without ambiguity.

        * **Legacy flat dict** (``param_grid``): for backward compatibility.
          Keys that already exist in ``base_job.indicator_params`` are routed
          there; all other keys go to ``strategy_params``.  This can
          mis-route *new* indicator params that are not yet in the base job.

        Args:
            base_job (AlphaJob): The base job configuration (not mutated).
            param_grid (Dict[str, List[Any]]): Legacy flat parameter grid.
            indicator_param_grid (Dict[str, List[Any]]): Grid for indicator
                params. Keys are always written to ``AlphaJob.indicator_params``.
            strategy_param_grid (Dict[str, List[Any]]): Grid for strategy
                params. Keys are always written to ``AlphaJob.strategy_params``.
            n_jobs (int): Number of parallel jobs (-1 = all cores).

        Returns:
            pd.DataFrame: Parameter combinations with resulting metrics.
        """
        import itertools

        # Build a unified param dict split by destination
        ind_grid = dict(indicator_param_grid or {})
        strat_grid = dict(strategy_param_grid or {})

        if param_grid:
            # Legacy routing: existing indicator keys → indicator_params, rest → strategy_params
            for k, v in param_grid.items():
                if k in base_job.indicator_params:
                    ind_grid[k] = v
                else:
                    strat_grid[k] = v

        combined_keys = list(ind_grid.keys()) + list(strat_grid.keys())
        combined_values = list(ind_grid.values()) + list(strat_grid.values())
        combinations = list(itertools.product(*combined_values))
        n_ind_keys = len(ind_grid)  # first N keys belong to indicator_params

        console.print(f"[cyan]Running sensitivity analysis on {len(combinations)} combinations...[/cyan]")

        jobs = []
        for i, combo in enumerate(combinations):
            # deepcopy avoids mutating the original job across iterations
            new_job = copy.deepcopy(base_job)
            new_job.id = f"{base_job.id}_sens_{i}"

            # Route each value to the correct param dict
            for j, (k, v) in enumerate(zip(combined_keys, combo)):
                if j < n_ind_keys:
                    new_job.indicator_params[k] = v
                else:
                    new_job.strategy_params[k] = v
                new_job.tags[k] = str(v)

            jobs.append(new_job)

        # Run batch
        results = self.runner.run_batch(jobs, n_jobs=n_jobs)

        # Collect results into a DataFrame
        data = []
        for r, combo in zip(results, combinations):
            if r.status == "completed":
                row = dict(zip(combined_keys, combo))
                # Flatten metrics dict
                row.update(r.metrics)
                data.append(row)
            else:
                console.print(f"[red]Job failed for combo {combo}: {r.error}[/red]")

        return pd.DataFrame(data)

    def sub_period_analysis(
        self, result: AlphaResult, period: str = "Y"  # Y=Yearly, Q=Quarterly, M=Monthly
    ) -> pd.DataFrame:
        """
        Analyze strategy performance across different sub-periods.

        Args:
            result (AlphaResult): Completed alpha result.
            period (str): Pandas frequency string (Y, Q, M).

        Returns:
            pd.DataFrame: Metrics for each sub-period.
        """
        if result.status != "completed" or result.equity_curve is None:
            raise ValueError("Result is not completed or has no equity curve.")

        equity = result.equity_curve
        if not isinstance(equity.index, pd.DatetimeIndex):
            # Try to convert to DatetimeIndex
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception as e:
                raise ValueError(f"Equity curve index is not datetime and cannot be converted: {e}")

        returns = equity.pct_change().dropna()

        # Use resample instead of groupby for time series
        # We want to calculate metrics for each period
        period_metrics = []

        # Iterate over periods
        # resample('Y') gives the last day of each year
        # We can iterate through groups
        # Use 'QE' instead of 'Q' as 'Q' is deprecated in recent pandas versions
        if period == "Q":
            freq = "QE"
        elif period == "M":
            freq = "ME"
        elif period == "Y":
            freq = "YE"
        else:
            freq = period

        grouper = returns.groupby(pd.Grouper(freq=freq))

        for name, group in grouper:
            if len(group) < 10:  # Skip very short periods
                continue

            # Calculate metrics for this chunk
            total_ret = (1 + group).prod() - 1
            ann_factor = 252  # Assuming daily data
            vol = group.std() * np.sqrt(ann_factor)

            sharpe = (group.mean() * ann_factor) / vol if vol > 0 else 0

            # Drawdown for this period
            period_cum_ret = (1 + group).cumprod()
            max_dd = (period_cum_ret / period_cum_ret.cummax() - 1).min()

            period_metrics.append(
                {"Period": name, "Return": total_ret, "Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": max_dd}
            )

        df = pd.DataFrame(period_metrics)
        if not df.empty:
            df.set_index("Period", inplace=True)
        return df

    def plot_parameter_heatmap(
        self,
        sensitivity_df: pd.DataFrame,
        param_x: str,
        param_y: str,
        metric: str = "sharpe_ratio",
        title: str = "Parameter Sensitivity Heatmap",
    ) -> plt.Figure:
        """Plot a heatmap of a metric for two parameters."""
        try:
            # Pivot table to grid format
            pivot_table = sensitivity_df.pivot_table(index=param_y, columns=param_x, values=metric)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
            ax.set_title(f"{title} ({metric})")
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            return fig
        except Exception as e:
            console.print(f"[red]Failed to plot heatmap: {e}[/red]")
            return plt.figure()

    def plot_sub_period_stability(
        self,
        period_df: pd.DataFrame,
        metrics: List[str] = ["Sharpe", "Return"],
        title: str = "Sub-period Stability Analysis",
    ) -> plt.Figure:
        """Plot stability of metrics across sub-periods."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize index to string for better plotting if it's datetime
        if isinstance(period_df.index, pd.DatetimeIndex):
            period_df.index = period_df.index.strftime("%Y-%m-%d")

        period_df[metrics].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig
