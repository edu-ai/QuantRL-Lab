"""
Single-stock hyperparameter tuning script.

Validates the full workflow end-to-end on a single symbol with Optuna:
  1. Fetch OHLCV data
  2. Alpha Research — suggest indicators
  3. Fetch optional enrichment data (FMP, Alpaca)
  4. Build and execute the DataPipeline (via shared utilities)
  5. Train/eval/test split (3-way)
  6. Tune PPO hyperparameters using Optuna (optimizing on the eval set)
  7. Report best hyperparameters
"""

import os
import sys

# Allow ``python examples/end_to_end/single_asset/tune_single_symbol.py``
# to resolve the shared package regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402

load_dotenv()
console = Console()

# ── Shared utilities ──────────────────────────────────────────────────────────
from shared.data_utils import (  # noqa: E402
    get_date_range,
    init_data_sources,
    process_symbol,
    select_alpha_indicators,
    train_eval_test_split_by_date,
)

SYMBOL = "AAPL"
PERIOD_YEARS = 5  # Increased to give a meaningful test set
WINDOW_SIZE = 40
TUNE_TIMESTEPS = 50_000  # Per Optuna trial — keep fast for broad search
REFIT_TIMESTEPS = 200_000  # Final refit on train+eval — thorough convergence
N_TRIALS = 20  # Minimum ~20 trials for meaningful Optuna search


# ── Data ──────────────────────────────────────────────────────────────────────


def fetch_sync_enrichment(sources, symbol: str, start_date, end_date) -> dict:
    """
    Synchronously fetch optional FMP + Alpaca enrichment for one symbol.

    Returns a dict compatible with ``process_symbol(enrichment=...)``:
    ``{ratings_df, sector_perf_df, industry_perf_df, news_df}``
    """
    enrichment = {
        "ratings_df": None,
        "sector_perf_df": None,
        "industry_perf_df": None,
        "news_df": None,
    }

    if sources.fmp:
        fmp = sources.fmp
        try:
            enrichment["ratings_df"] = fmp.get_historical_rating(symbol, limit=500)
            console.print(f"[green]Analyst ratings:[/green] {enrichment['ratings_df'].shape}")
        except Exception as exc:
            console.print(f"[yellow]Analyst ratings failed: {exc}[/yellow]")
        try:
            profile = fmp.get_company_profile(symbol)
            if not profile.empty:
                sector = profile.iloc[0].get("sector")
                industry = profile.iloc[0].get("industry")
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                if sector:
                    enrichment["sector_perf_df"] = fmp.get_historical_sector_performance(
                        sector, start=start_str, end=end_str
                    )
                    console.print(f"[green]Sector perf ({sector}):[/green] {enrichment['sector_perf_df'].shape}")
                if industry:
                    enrichment["industry_perf_df"] = fmp.get_historical_industry_performance(
                        industry, start=start_str, end=end_str
                    )
                    console.print(f"[green]Industry perf ({industry}):[/green] {enrichment['industry_perf_df'].shape}")
        except Exception as exc:
            console.print(f"[yellow]Market context failed: {exc}[/yellow]")

    if sources.alpaca:
        try:
            enrichment["news_df"] = sources.alpaca.get_news_data(symbols=[symbol], start=start_date, end=end_date)
            console.print(f"[green]News data:[/green] {enrichment['news_df'].shape}")
        except Exception as exc:
            console.print(f"[yellow]News data failed: {exc}[/yellow]")

    return enrichment


def build_processed_data(symbol: str) -> "pd.DataFrame":  # noqa: F821
    """Fetch, enrich, and process data for a single symbol."""
    # 1. Initialise sources and compute date range
    sources = init_data_sources()
    start_date, end_date = get_date_range(PERIOD_YEARS)

    # 2. OHLCV
    raw_df = sources.loader.get_historical_ohlcv_data(symbols=[symbol], start=start_date, end=end_date, timeframe="1d")
    if "Date" in raw_df.columns:
        raw_df = raw_df.set_index("Date")
    console.print(f"[cyan]Raw OHLCV:[/cyan] {raw_df.shape}")

    # 3. Alpha Research — suggest indicators
    indicators = select_alpha_indicators({symbol: raw_df}, metric="ic", threshold=0.02, top_k=4, verbose=True)

    # 4. Optional enrichment (sync)
    enrichment = fetch_sync_enrichment(sources, symbol, start_date, end_date)

    # 5. Build pipeline, execute, drop NaNs
    processed_df = process_symbol(symbol, raw_df, indicators, enrichment, verbose=True)

    # Drop non-numeric columns that the trading env cannot cast to float32
    processed_df = processed_df.select_dtypes(include="number")
    console.print(f"[green]Final shape:[/green] {processed_df.shape}")
    return processed_df


# ── Tuning ────────────────────────────────────────────────────────────────────


def main():
    console.rule(f"[bold blue]Single-Stock Tuning — {SYMBOL}[/bold blue]")

    # --- Phase 1: Data ---
    console.rule("[dim]Phase 1: Data[/dim]")
    processed_df = build_processed_data(SYMBOL)
    console.print(f"[green]Date range:[/green] {processed_df.index.min().date()} → {processed_df.index.max().date()}")
    console.print(f"[green]Features:[/green] {list(processed_df.columns)}")

    # --- Phase 2: Train/Eval/Test split ---
    console.rule("[dim]Phase 2: Split[/dim]")
    train_df, eval_df, test_df = train_eval_test_split_by_date(processed_df, train_ratio=0.7, eval_ratio=0.15)

    if len(train_df) <= WINDOW_SIZE:
        console.print(f"[red]Train set too small ({len(train_df)} rows <= window_size={WINDOW_SIZE})[/red]")
        return
    if len(eval_df) <= WINDOW_SIZE:
        console.print(f"[red]Eval set too small ({len(eval_df)} rows <= window_size={WINDOW_SIZE})[/red]")
        return
    if len(test_df) <= WINDOW_SIZE:
        console.print(f"[red]Test set too small ({len(test_df)} rows <= window_size={WINDOW_SIZE})[/red]")
        return

    # --- Phase 3: Setup Environment using Builder ---
    console.rule("[dim]Phase 3: Environment[/dim]")
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
    from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
    from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

    # Simplified reward: portfolio return is the primary signal.
    # Heavy penalties (drawdown, sortino) previously caused the agent to learn
    # "do nothing" as the optimal strategy. A tiny turnover penalty is kept to
    # discourage churning without suppressing exploration.
    reward_strat = CompositeReward(
        strategies=[
            PortfolioValueChangeReward(),
            TurnoverPenaltyReward(penalty_factor=0.1),
        ],
        weights=[1.0, 0.05],
        auto_scale=False,  # Disable auto-scale: raw % return is already well-scaled
    )

    builder = BacktestEnvironmentBuilder()
    builder.with_data(train_data=train_df, eval_data=eval_df, test_data=test_df)
    builder.with_env_params(
        initial_balance=100_000.0,
        transaction_cost_pct=0.001,
        window_size=WINDOW_SIZE,
    )
    builder.with_strategies(
        action=StandardActionStrategy(),
        reward=reward_strat,
        observation=FeatureAwareObservationStrategy(normalize_stationary=True),
    )
    env_config = builder.build()

    # --- Phase 4: Run Tuning Experiment ---
    console.rule("[dim]Phase 4: Hyperparameter Tuning[/dim]")
    from stable_baselines3 import PPO

    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner
    from quantrl_lab.experiments.tuning.optuna_runner import OptunaRunner, create_ppo_search_space

    # Define the parameter search space
    search_space = create_ppo_search_space()
    # Raise entropy floor to prevent premature collapse to "do nothing" policy.
    # 1e-4 is too low — PPO can still suppress exploration entirely.
    search_space["ent_coef"]["low"] = 0.01

    # Runners
    base_runner = BacktestRunner(verbose=False)
    tuner = OptunaRunner(runner=base_runner, storage_url="sqlite:///optuna_studies.db")

    study = tuner.optimize_hyperparameters(
        algo_class=PPO,
        env_config=env_config,
        search_space=search_space,
        study_name=f"ppo_tuning_{SYMBOL.lower()}",
        n_trials=N_TRIALS,
        total_timesteps=TUNE_TIMESTEPS,
        num_eval_episodes=3,  # More episodes for a stable eval signal
        optimization_metric="eval_avg_return_pct",  # Optimize based on evaluation set!
        direction="maximize",
    )

    console.rule("[bold green]Tuning Complete![/bold green]")
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        console.print("[bold red]No successful trials — cannot refit.[/bold red]")
        return

    console.print("[bold cyan]Best Trial Params:[/bold cyan]")
    for key, value in study.best_params.items():
        console.print(f"  [yellow]{key}[/yellow]: {value}")

    # --- Phase 5: Refit on train+eval, evaluate on test ---
    console.rule("[dim]Phase 5: Refit & Final Evaluation[/dim]")

    import pandas as pd

    refit_df = pd.concat([train_df, eval_df]).sort_index()
    console.print(f"[cyan]Refit data:[/cyan] {len(refit_df)} rows (train + eval)")

    refit_builder = BacktestEnvironmentBuilder()
    refit_builder.with_data(train_data=refit_df, test_data=test_df)
    refit_builder.with_env_params(
        initial_balance=100_000.0,
        transaction_cost_pct=0.001,
        window_size=WINDOW_SIZE,
    )
    refit_builder.with_strategies(
        action=StandardActionStrategy(),
        reward=reward_strat,
        observation=FeatureAwareObservationStrategy(normalize_stationary=True),
    )
    refit_env_config = refit_builder.build()

    refit_job = ExperimentJob(
        algorithm_class=PPO,
        env_config=refit_env_config,
        algorithm_config=dict(policy="MlpPolicy", **study.best_params),
        total_timesteps=REFIT_TIMESTEPS,  # More timesteps for thorough final training
        n_envs=1,
        tags={"stage": "refit"},
    )

    final_runner = BacktestRunner(verbose=True)
    final_result = final_runner.run_job(refit_job)
    final_runner.inspect_result(final_result)


if __name__ == "__main__":
    main()
