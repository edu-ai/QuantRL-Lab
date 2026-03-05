"""
Single-stock training with A2C (Advantage Actor-Critic).

A2C is a synchronous, on-policy actor-critic algorithm.  Key differences
vs PPO and SAC:

- On-policy like PPO, but with no clipping — simpler update rule.
- Synchronous: each ``n_envs`` rollout is collected before a single gradient
  step, so more envs directly increase throughput without async coordination.
- Lower memory footprint than PPO (no large rollout buffer needed).
- Typically converges faster early on but may be less stable than PPO on
  longer horizons.

Workflow:
  1. Fetch OHLCV data
  2. Alpha Research — suggest indicators
  3. Fetch optional enrichment data (FMP, Alpaca)
  4. Build and execute the DataPipeline (via shared utilities)
  5. Train/test split
  6. Train an A2C agent
  7. Evaluate on train and test sets

Shared data-loading logic lives in ``examples/end_to_end/shared/data_utils.py``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402

load_dotenv()
console = Console()

from shared.data_utils import (  # noqa: E402
    get_date_range,
    init_data_sources,
    process_symbol,
    select_alpha_indicators,
    train_test_split_by_date,
)

SYMBOL = "AAPL"
PERIOD_YEARS = 2
WINDOW_SIZE = 40
TOTAL_TIMESTEPS = 250_000


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
    """
    Fetch, enrich, and process data for a single symbol.

    Uses shared utilities for data-source init, date range, alpha
    selection, and pipeline execution.
    """
    sources = init_data_sources()
    start_date, end_date = get_date_range(PERIOD_YEARS)

    raw_df = sources.loader.get_historical_ohlcv_data(symbols=[symbol], start=start_date, end=end_date, timeframe="1d")
    if "Date" in raw_df.columns:
        raw_df = raw_df.set_index("Date")
    console.print(f"[cyan]Raw OHLCV:[/cyan] {raw_df.shape}")

    indicators = select_alpha_indicators({symbol: raw_df}, metric="ic", threshold=0.02, top_k=4, verbose=True)
    enrichment = fetch_sync_enrichment(sources, symbol, start_date, end_date)
    processed_df = process_symbol(symbol, raw_df, indicators, enrichment, verbose=True)
    processed_df = processed_df.select_dtypes(include="number")
    console.print(f"[green]Final shape:[/green] {processed_df.shape}")
    return processed_df


# ── Training & Evaluation ─────────────────────────────────────────────────────


def main():
    console.rule(f"[bold blue]Single-Stock A2C Training — {SYMBOL}[/bold blue]")

    # --- Phase 1: Data ---
    console.rule("[dim]Phase 1: Data[/dim]")
    processed_df = build_processed_data(SYMBOL)
    console.print(f"[green]Date range:[/green] {processed_df.index.min().date()} → {processed_df.index.max().date()}")
    console.print(f"[green]Features:[/green] {list(processed_df.columns)}")

    # --- Phase 2: Train/Test split ---
    console.rule("[dim]Phase 2: Split[/dim]")
    train_df, test_df = train_test_split_by_date(processed_df, split_ratio=0.8)

    if len(train_df) <= WINDOW_SIZE:
        console.print(f"[red]Train set too small ({len(train_df)} rows ≤ window_size={WINDOW_SIZE})[/red]")
        return
    if len(test_df) <= WINDOW_SIZE:
        console.print(f"[red]Test set too small ({len(test_df)} rows ≤ window_size={WINDOW_SIZE})[/red]")
        return

    # --- Phase 3: Environment ---
    console.rule("[dim]Phase 3: Environment[/dim]")
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
    from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

    # Pure portfolio return reward to encourage aggressive trading
    reward_strat = CompositeReward(
        strategies=[
            PortfolioValueChangeReward(),
        ],
        weights=[1.0],
        auto_scale=False,
    )

    builder = BacktestEnvironmentBuilder()
    builder.with_data(train_data=train_df, test_data=test_df)
    builder.with_env_params(
        initial_balance=100_000.0,
        transaction_cost_pct=0.0,  # set to 0.0 to remove all friction and encourage aggressive trading
        window_size=WINDOW_SIZE,
    )
    builder.with_strategies(
        action=StandardActionStrategy(),
        reward=reward_strat,
        observation=FeatureAwareObservationStrategy(normalize_stationary=True),
    )
    env_config = builder.build()

    # --- Phase 4: Training ---
    # A2C notes:
    #   - ``n_steps=5`` is the classic A2C setting (short rollout, frequent updates).
    #   - ``n_envs=4`` runs four parallel envs to stabilise gradient estimates.
    #   - ``ent_coef=0.01`` encourages exploration; tune up if the agent collapses
    #     to "hold" early in training.
    #   - ``vf_coef=0.5`` balances actor and critic loss contributions.
    #   - ``max_grad_norm=0.5`` clips gradients to prevent instability.
    console.rule("[dim]Phase 4: Training (A2C)[/dim]")
    from stable_baselines3 import A2C

    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    job = ExperimentJob(
        algorithm_class=A2C,
        env_config=env_config,
        algorithm_config=dict(
            policy="MlpPolicy",
            learning_rate=7e-4,
            n_steps=5,  # short rollout — A2C's classic setting
            gamma=0.99,
            gae_lambda=1.0,  # no GAE bias correction (pure advantage)
            ent_coef=0.1,  # increased to force more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            verbose=0,
        ),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=4,
    )

    runner = BacktestRunner(verbose=True)
    result = runner.run_job(job)
    runner.inspect_result(result)


if __name__ == "__main__":
    main()
