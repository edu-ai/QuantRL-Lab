import copy
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

from quantrl_lab.experiments.alpha_research.core import AlphaJob, AlphaResult
from quantrl_lab.experiments.alpha_research.runner import AlphaRunner

console = Console()


class RobustnessTester:
    """
    Tools for testing the robustness of alpha strategies.

    Includes Parameter Sensitivity Analysis and Sub-period Analysis.
    """

    def __init__(self, runner: Optional[AlphaRunner] = None):
        self.runner = runner or AlphaRunner(verbose=False)

    def parameter_sensitivity(
        self, base_job: AlphaJob, param_grid: Dict[str, List[Any]], n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Run a grid search over strategy parameters to analyze
        sensitivity.

        Args:
            base_job (AlphaJob): The base job configuration.
            param_grid (Dict[str, List[Any]]): Dictionary of parameters to vary.
                                               e.g. {"window": [10, 20, 30], "threshold": [1, 2]}
            n_jobs (int): Number of parallel jobs.

        Returns:
            pd.DataFrame: DataFrame with parameter combinations and resulting metrics.
        """
        import itertools

        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        console.print(f"[cyan]Running sensitivity analysis on {len(combinations)} combinations...[/cyan]")

        jobs = []
        for i, combo in enumerate(combinations):
            # Create a copy of the base job
            # We must use deepcopy to avoid mutating the original
            new_job = copy.deepcopy(base_job)
            new_job.id = f"{base_job.id}_sens_{i}"

            # Map values to keys
            current_params = dict(zip(keys, combo))

            # Update params in the job
            # Priority: Strategy params first, then indicator params if key exists there
            # (or check explicitly where it belongs)

            for k, v in current_params.items():
                if k in new_job.indicator_params:
                    new_job.indicator_params[k] = v
                else:
                    new_job.strategy_params[k] = v

            # Add tags for tracking
            for k, v in current_params.items():
                new_job.tags[k] = str(v)

            jobs.append(new_job)

        # Run batch
        results = self.runner.run_batch(jobs, n_jobs=n_jobs)

        # Collect results into a DataFrame
        data = []
        for r, combo in zip(results, combinations):
            if r.status == "completed":
                row = dict(zip(keys, combo))
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
            period_df.index = period_df.index.strftime('%Y-%m-%d')

        period_df[metrics].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig
