from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from quantrl_lab.alpha_research.converters import results_to_pipeline_config
from quantrl_lab.alpha_research.models import AlphaResult
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import TechnicalIndicatorStep
from quantrl_lab.environments.core.interfaces import (
    BaseActionStrategy,
    BaseObservationStrategy,
    BaseRewardStrategy,
)
from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig
from quantrl_lab.environments.stock.single import SingleStockTradingEnv
from quantrl_lab.experiments.backtesting.config.environment_config import (
    BacktestEnvironmentConfig,
)


class BacktestEnvironmentBuilder:
    """
    A fluent builder for creating BacktestEnvironmentConfig objects.

    Integrates:
    - Data Processing (via DataPipeline)
    - Alpha Research Signals (via AlphaResult)
    - Strategy Configuration
    """

    def __init__(self):
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._eval_data: Optional[pd.DataFrame] = None

        # Strategies
        self._action_strategy: Optional[BaseActionStrategy] = None
        self._reward_strategy: Optional[BaseRewardStrategy] = None
        self._observation_strategy: Optional[BaseObservationStrategy] = None

        # Environment Config
        self._env_params: Dict[str, Any] = {
            "initial_balance": 100000.0,
            "transaction_cost_pct": 0.001,
            "slippage": 0.0005,
            "window_size": 20,
        }

        # Pipeline
        self._alpha_results: List[AlphaResult] = []
        self._alpha_top_n: int = 5
        self._alpha_metric: str = "ic"
        self._custom_pipeline_steps: List[Any] = []

    def with_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
    ) -> "BacktestEnvironmentBuilder":
        """Set the data for the environment."""
        self._train_data = train_data
        self._test_data = test_data
        self._eval_data = eval_data
        return self

    def with_strategies(
        self,
        action: BaseActionStrategy,
        reward: BaseRewardStrategy,
        observation: BaseObservationStrategy,
    ) -> "BacktestEnvironmentBuilder":
        """Set the strategies for the environment."""
        self._action_strategy = action
        self._reward_strategy = reward
        self._observation_strategy = observation
        return self

    def with_env_params(self, **kwargs) -> "BacktestEnvironmentBuilder":
        """Update environment parameters (e.g. initial_balance,
        window_size)."""
        self._env_params.update(kwargs)
        return self

    def with_alpha_signals(
        self,
        results: List[AlphaResult],
        top_n: int = 5,
        metric: str = "ic",
    ) -> "BacktestEnvironmentBuilder":
        """
        Integrate validated alpha signals into the data pipeline.

        Args:
            results: List of AlphaResult objects from AlphaRunner.
            top_n: Number of top performing indicators to select.
            metric: Metric to use for selection (e.g. "ic", "sharpe_ratio").
        """
        self._alpha_results = results
        self._alpha_top_n = top_n
        self._alpha_metric = metric
        return self

    def add_pipeline_step(self, step: Any) -> "BacktestEnvironmentBuilder":
        """Add a custom step to the DataPipeline."""
        self._custom_pipeline_steps.append(step)
        return self

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pipeline transformations to data."""
        if df is None or df.empty:
            return df

        pipeline = DataPipeline()

        # 1. Add Alpha Signals if configured
        if self._alpha_results:
            indicator_config = results_to_pipeline_config(
                self._alpha_results, top_n=self._alpha_top_n, metric=self._alpha_metric
            )
            if indicator_config:
                logger.info(f"Injecting {len(indicator_config)} alpha signals into environment data.")
                pipeline.add_step(TechnicalIndicatorStep(indicators=indicator_config))

        # 2. Add custom steps
        for step in self._custom_pipeline_steps:
            pipeline.add_step(step)

        if len(pipeline) > 0:
            processed, _ = pipeline.execute(df)

            # Clean up NaNs (critical for RL stability)
            initial_len = len(processed)
            processed = processed.dropna()
            dropped = initial_len - len(processed)

            if dropped > 0:
                logger.info(f"Dropped {dropped} rows containing NaNs from processed data.")

            return processed

        return df

    def build(self) -> BacktestEnvironmentConfig:
        """Build the BacktestEnvironmentConfig."""
        if self._train_data is None or self._test_data is None:
            raise ValueError("Training and testing data must be provided.")

        if not all([self._action_strategy, self._reward_strategy, self._observation_strategy]):
            raise ValueError("All strategies (action, reward, observation) must be provided.")

        # Process Data (Inject Alphas)
        train_processed = self._process_data(self._train_data)
        test_processed = self._process_data(self._test_data)
        eval_processed = self._process_data(self._eval_data) if self._eval_data is not None else None

        # Create Config
        env_config_obj = SingleStockEnvConfig(**self._env_params)

        # Create Factories
        def create_env(data: pd.DataFrame):
            if "Symbol" in data.columns and len(data["Symbol"].unique()) > 1:
                # Multi-stock panel data: return a list of factories
                factories = []
                for sym in data["Symbol"].unique():
                    sym_data = data[data["Symbol"] == sym].copy()
                    factories.append(
                        lambda d=sym_data: SingleStockTradingEnv(
                            data=d,
                            config=env_config_obj,
                            action_strategy=self._action_strategy,
                            reward_strategy=self._reward_strategy,
                            observation_strategy=self._observation_strategy,
                        )
                    )
                return factories
            else:
                # Single stock data: return a single factory
                return lambda: SingleStockTradingEnv(
                    data=data,
                    config=env_config_obj,
                    action_strategy=self._action_strategy,
                    reward_strategy=self._reward_strategy,
                    observation_strategy=self._observation_strategy,
                )

        return BacktestEnvironmentConfig(
            train_env_factory=create_env(train_processed),
            test_env_factory=create_env(test_processed),
            eval_env_factory=create_env(eval_processed) if eval_processed is not None else None,
            parameters={
                **self._env_params,
                "action_strategy": self._action_strategy.__class__.__name__,
                "reward_strategy": self._reward_strategy.__class__.__name__,
                "observation_strategy": self._observation_strategy.__class__.__name__,
                "alpha_signals_count": len(self._alpha_results) if self._alpha_results else 0,
            },
            description=f"Auto-configured Stock Env with {len(self._alpha_results) if self._alpha_results else 0} alphas",  # noqa: E501
        )
