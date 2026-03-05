from typing import Callable, List, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule  # noqa: F401
from rich.table import Table
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from quantrl_lab.environments.core.interfaces import (
    BaseActionStrategy,
    BaseObservationStrategy,
    BaseRewardStrategy,
)

from .config.environment_config import BacktestEnvironmentConfig
from .core import ExperimentJob, ExperimentResult
from .evaluation import evaluate_model
from .metrics import MetricsCalculator
from .training import train_model

console = Console()


class BacktestRunner:
    """
    Orchestrates complete backtesting workflows by chaining training and
    evaluation.

    This class provides high-level interfaces for running comprehensive
    experiments that train multiple algorithms on different environment
    configurations and evaluate their performance.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metrics_calculator = MetricsCalculator()

    def run_job(self, job: ExperimentJob) -> ExperimentResult:
        """
        Executes a single experiment job using the new Job/Batch
        architecture.

        Args:
            job (ExperimentJob): The job description containing all parameters.

        Returns:
            ExperimentResult: The result of the experiment.
        """
        import time

        start_time = time.time()

        if self.verbose:
            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            console.print(f"[bold blue]RUNNING JOB: {job.id}[/bold blue]")
            console.print(f"[cyan]Algo: {job.algorithm_class.__name__} | Env: {job.env_config.name}[/cyan]")
            if job.algorithm_config:
                console.print(f"[dim]Config: {job.algorithm_config}[/dim]")

        try:
            # 1. Training Phase
            if self.verbose:
                console.print("[bold green]🔄 Training...[/bold green]")

            # Create vectorized environment for training
            if isinstance(job.env_config.train_env_factory, list):
                # Multi-stock vectorized training
                train_vec_env = SubprocVecEnv(job.env_config.train_env_factory)
            else:
                # Single-stock parallel rollout
                train_vec_env = make_vec_env(job.env_config.train_env_factory, n_envs=job.n_envs)

            model = train_model(
                algo_class=job.algorithm_class,
                env=train_vec_env,
                config=job.algorithm_config,
                total_timesteps=job.total_timesteps,
                verbose=1 if self.verbose else 0,
            )

            # 2. Evaluation Phase
            if self.verbose:
                console.print("[bold blue]📊 Evaluating...[/bold blue]")

            def _evaluate_factories(factories) -> tuple:
                if not isinstance(factories, list):
                    factories = [factories]
                all_rewards = []
                all_episodes = []
                for factory in factories:
                    env = factory()
                    rew, eps = evaluate_model(model=model, env=env, num_episodes=job.num_eval_episodes, verbose=False)
                    all_rewards.extend(rew)
                    all_episodes.extend(eps)
                    env.close()
                return all_rewards, all_episodes

            # Evaluate on Train
            train_rewards, train_episodes = _evaluate_factories(job.env_config.train_env_factory)

            # Evaluate on Test
            test_rewards, test_episodes = _evaluate_factories(job.env_config.test_env_factory)

            # 3. Metrics Calculation
            train_metrics = self.metrics_calculator.calculate(train_episodes)
            test_metrics = self.metrics_calculator.calculate(test_episodes)

            # Flattened metrics for the result object
            # Prefix with dataset name for clarity
            metrics = {}
            for k, v in train_metrics.items():
                metrics[f"train_{k}"] = v
            for k, v in test_metrics.items():
                metrics[f"test_{k}"] = v

            # 4. Feature Importance
            top_features = {}
            explanation_method = "Correlation"
            if self.verbose:
                console.print("[bold yellow]🧠 Analyzing Feature Importance...[/bold yellow]")
            try:
                from quantrl_lab.experiments.backtesting.explainer import AgentExplainer

                exp_factory = job.env_config.test_env_factory
                env_for_explainer = exp_factory[0]() if isinstance(exp_factory, list) else exp_factory()

                explainer = AgentExplainer(model, env_for_explainer)
                top_features = explainer.analyze_feature_importance(top_k=5)
                explanation_method = getattr(explainer, "last_method_used", "Correlation")
                env_for_explainer.close()
            except Exception as e:
                explanation_method = "Correlation"
                if self.verbose:
                    console.print(f"[yellow]Feature importance analysis skipped/failed: {e}[/yellow]")

            execution_time = time.time() - start_time

            result = ExperimentResult(
                job=job,
                metrics=metrics,
                model=model,
                train_episodes=train_episodes,
                test_episodes=test_episodes,
                top_features=top_features,
                explanation_method=explanation_method,
                status="completed",
                execution_time=execution_time,
            )

            if self.verbose:
                train_return = metrics.get("train_avg_return_pct", 0.0)
                test_return = metrics.get("test_avg_return_pct", 0.0)
                train_sharpe = metrics.get("train_avg_sharpe_ratio", 0.0)
                test_sharpe = metrics.get("test_avg_sharpe_ratio", 0.0)

                train_color = "green" if train_return > 0 else "red"
                test_color = "green" if test_return > 0 else "red"

                console.print("[bold]Result:[/bold]")
                console.print(
                    f"  Train: [{train_color}]{train_return:.2f}%[/{train_color}] (Sharpe: {train_sharpe:.2f})"
                )
                console.print(f"  Test:  [{test_color}]{test_return:.2f}%[/{test_color}] (Sharpe: {test_sharpe:.2f})")

            return result

        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]❌ Job Failed: {str(e)}[/bold red]")
                import traceback

                console.print(traceback.format_exc())

            return ExperimentResult(
                job=job, metrics={}, status="failed", error=e, execution_time=time.time() - start_time
            )

    def run_batch(self, jobs: List[ExperimentJob]) -> List[ExperimentResult]:
        """
        Executes a batch of jobs sequentially (can be upgraded to
        parallel later).

        Args:
            jobs (List[ExperimentJob]): List of jobs to run.

        Returns:
            List[ExperimentResult]: Results for each job.
        """
        results = []
        if self.verbose:
            console.print(f"\n[bold magenta]Starting Batch Execution: {len(jobs)} jobs[/bold magenta]")

        for i, job in enumerate(jobs):
            if self.verbose:
                console.print(f"\n[dim]--- Job {i+1}/{len(jobs)} ---[/dim]")
            results.append(self.run_job(job))

        if self.verbose:
            success_count = sum(1 for r in results if r.status == "completed")
            console.print(f"\n[bold magenta]Batch Completed: {success_count}/{len(jobs)} successful[/bold magenta]")

        return results

    @staticmethod
    def inspect_result(result: ExperimentResult) -> None:
        """
        Inspect and display the results of a single experiment job.

        Args:
            result (ExperimentResult): The result object to inspect.
        """
        job = result.job
        metrics = result.metrics

        # --- Main Summary Panel ---
        algo_name = job.algorithm_class.__name__
        config_id = job.tags.get("config_id", "default")
        train_return = metrics.get("train_avg_return_pct", 0.0)
        test_return = metrics.get("test_avg_return_pct", 0.0)

        train_return_color = "green" if train_return >= 0 else "red"
        test_return_color = "green" if test_return >= 0 else "red"

        summary_text = (
            f"Job ID: [bold]{job.id}[/bold]\n"
            f"Algorithm: [bold cyan]{algo_name}[/bold cyan]\n"
            f"Env: [yellow]{job.env_config.name}[/yellow]\n"
            f"Config ID: [yellow]{config_id}[/yellow]\n"
            f"Status: {result.status}\n"
            f"Train Avg Return: [{train_return_color}]{train_return:.2f}%[/{train_return_color}]\n"
            f"Test Avg Return:  [{test_return_color}]{test_return:.2f}%[/{test_return_color}]\n"
        )

        # Add advanced metrics if available
        if "test_avg_sharpe_ratio" in metrics:
            summary_text += f"Test Sharpe: {metrics['test_avg_sharpe_ratio']:.2f}\n"
        if "test_avg_max_drawdown" in metrics:
            summary_text += f"Test Max DD: {metrics['test_avg_max_drawdown']*100:.2f}%\n"

        if result.top_features:
            summary_text += f"\n[bold]Top Learned Features ({result.explanation_method}):[/bold]\n"
            for feat, score in result.top_features.items():
                summary_text += f"  - {feat}: {score:+.2f}\n"

        if result.error:
            summary_text += f"\n[red]Error: {str(result.error)}[/red]"

        console.print(Panel(summary_text, title="[bold]Experiment Result[/bold]", expand=False))

        if result.status == "failed":
            return

        # --- Episode Details Table ---
        episode_table = Table(title="Episode Performance Details", show_header=True, header_style="bold magenta")
        episode_table.add_column("Dataset", style="cyan")
        episode_table.add_column("Episode", justify="center")
        episode_table.add_column("Return %", justify="right")
        episode_table.add_column("Reward", justify="right")
        episode_table.add_column("Final Value", justify="right")
        episode_table.add_column("Total Steps", justify="right")

        # Function to add rows for a dataset (train/test)
        def add_episode_rows(dataset_name, episodes):
            if not episodes:
                return
            for i, ep in enumerate(episodes):
                if "error" in ep:
                    continue
                initial = ep.get("initial_value", 0)
                final = ep.get("final_value", 0)
                reward = ep.get("total_reward", 0)

                ret = ((final - initial) / initial) * 100 if initial != 0 else 0
                ret_color = "green" if ret >= 0 else "red"
                reward_color = "green" if reward >= 0 else "red"

                episode_table.add_row(
                    dataset_name,
                    str(i + 1),
                    f"[{ret_color}]{ret:.2f}%[/{ret_color}]",
                    f"[{reward_color}]{reward:.2f}[/{reward_color}]",
                    f"${final:,.2f}",
                    str(ep.get("steps", "N/A")),
                )

        add_episode_rows("Train", result.train_episodes)
        add_episode_rows("Test", result.test_episodes)

        if result.train_episodes or result.test_episodes:
            console.print(episode_table)
        else:
            console.print("[yellow]No episode data available.[/yellow]")

        # --- Action Distribution Table ---
        all_episodes = result.train_episodes + result.test_episodes
        all_actions: dict = {}
        total_steps = 0
        for ep in all_episodes:
            if "error" not in ep:
                total_steps += ep.get("steps", 0)
                for action_type, count in ep.get("actions_taken", {}).items():
                    all_actions[action_type] = all_actions.get(action_type, 0) + count

        if all_actions and total_steps > 0:
            action_table = Table(title="Action Distribution (all episodes)", show_header=True, header_style="bold cyan")
            action_table.add_column("Action", style="cyan")
            action_table.add_column("Count", justify="right")
            action_table.add_column("% of Steps", justify="right", style="yellow")
            for action_type, count in sorted(all_actions.items()):
                action_table.add_row(action_type, str(count), f"{count / total_steps * 100:.1f}%")
            console.print(action_table)

    @staticmethod
    def inspect_batch(results: List[ExperimentResult]) -> None:
        """
        Inspect and display a summary of a batch of experiments.

        Args:
            results (List[ExperimentResult]): List of experiment results.
        """
        console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
        console.print("[bold magenta]BATCH EXPERIMENT SUMMARY[/bold magenta]")
        console.print(f"[bold magenta]{'='*80}[/bold magenta]")
        # Preset column removed
        table = Table(title="Batch Results", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Algo", style="cyan")
        table.add_column("Env", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Train Ret %", justify="right")
        table.add_column("Test Ret %", justify="right")
        table.add_column("Test Sharpe", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Top Feature", style="dim")

        for res in results:
            job = res.job
            metrics = res.metrics

            status_style = "green" if res.status == "completed" else "red"
            status_str = f"[{status_style}]{res.status}[/{status_style}]"

            if res.status == "completed":
                train_ret = metrics.get("train_avg_return_pct", 0.0)
                test_ret = metrics.get("test_avg_return_pct", 0.0)
                test_sharpe = metrics.get("test_avg_sharpe_ratio", 0.0)

                train_color = "green" if train_ret >= 0 else "red"
                test_color = "green" if test_ret >= 0 else "red"

                train_str = f"[{train_color}]{train_ret:.2f}%[/{train_color}]"
                test_str = f"[{test_color}]{test_ret:.2f}%[/{test_color}]"
                sharpe_str = f"{test_sharpe:.2f}"
            else:
                train_str = "-"
                test_str = "-"
                sharpe_str = "-"

            top_feat_str = "-"
            if res.top_features:
                # Get the highest correlated feature
                top_feat_name, top_feat_corr = list(res.top_features.items())[0]
                top_feat_str = f"{top_feat_name} ({top_feat_corr:+.2f})"

            # Add row to table
            table.add_row(
                job.id,
                job.algorithm_class.__name__,
                job.env_config.name,
                status_str,
                train_str,
                test_str,
                sharpe_str,
                f"{res.execution_time:.1f}",
                top_feat_str,
            )

        console.print(table)

    @staticmethod
    def create_env_config(train_env_factory: Callable, test_env_factory: Callable) -> BacktestEnvironmentConfig:
        """
        Helper method to create env_config from individual factory
        functions.

        Args:
            train_env_factory (Callable): Function that creates training environment
            test_env_factory (Callable): Function that creates test environment

        Returns:
            BacktestEnvironmentConfig: Environment configuration object
        """
        return BacktestEnvironmentConfig(train_env_factory=train_env_factory, test_env_factory=test_env_factory)

    @staticmethod
    def create_env_config_factory(
        train_data: "pd.DataFrame",
        test_data: "pd.DataFrame",
        action_strategy: "BaseActionStrategy",
        reward_strategy: "BaseRewardStrategy",
        observation_strategy: "BaseObservationStrategy",
        eval_data: Optional["pd.DataFrame"] = None,
        initial_balance: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        window_size: int = 20,
        order_expiration_steps: int = 5,
    ) -> BacktestEnvironmentConfig:
        """
        Creates a configuration object with environment factories.
        DEPRECATED: Use BacktestEnvironmentBuilder instead.

        Args:
            train_data (pd.DataFrame): DataFrame for the training environment.
            test_data (pd.DataFrame): DataFrame for the test environment.
            action_strategy (BaseActionStrategy): The action strategy to use.
            reward_strategy (BaseRewardStrategy): The reward strategy to use.
            observation_strategy (BaseObservationStrategy): The observation strategy.
            eval_data (Optional[pd.DataFrame], optional): DataFrame for evaluation.
            initial_balance (float, optional): Initial portfolio balance.
            transaction_cost_pct (float, optional): Transaction cost percentage.
            window_size (int, optional): The size of the observation window.

        Returns:
            BacktestEnvironmentConfig: Configuration object containing factories and metadata.
        """
        from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig
        from quantrl_lab.environments.stock.single import SingleStockTradingEnv

        # Helper function to create a single environment factory
        def _create_factory(data: "pd.DataFrame"):
            return lambda: SingleStockTradingEnv(
                data=data,
                config=SingleStockEnvConfig(
                    initial_balance=initial_balance,
                    transaction_cost_pct=transaction_cost_pct,
                    slippage=slippage_pct,
                    window_size=window_size,
                    order_expiration_steps=order_expiration_steps,
                ),
                action_strategy=action_strategy,
                reward_strategy=reward_strategy,
                observation_strategy=observation_strategy,
                price_column="Close",  # Default to Close
            )

        # Capture parameters for reproducibility
        parameters = {
            "initial_balance": initial_balance,
            "transaction_cost_pct": transaction_cost_pct,
            "slippage_pct": slippage_pct,
            "window_size": window_size,
            "order_expiration_steps": order_expiration_steps,
            "action_strategy": action_strategy.__class__.__name__,
            "reward_strategy": reward_strategy.__class__.__name__,
            "observation_strategy": observation_strategy.__class__.__name__,
        }

        eval_factory = _create_factory(eval_data) if eval_data is not None else None

        return BacktestEnvironmentConfig(
            train_env_factory=_create_factory(train_data),
            test_env_factory=_create_factory(test_data),
            eval_env_factory=eval_factory,
            parameters=parameters,
            description=f"Standard Stock Env (Window: {window_size})",
        )
