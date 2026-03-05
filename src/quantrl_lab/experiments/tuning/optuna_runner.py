import os
from typing import Any, Callable, Dict, Optional, Union

import optuna
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install as install_rich_traceback

from ..backtesting.config.environment_config import BacktestEnvironmentConfig
from ..backtesting.core import ExperimentJob
from ..backtesting.runner import BacktestRunner

# Install rich traceback handler
install_rich_traceback()

# Set up rich console
console = Console()


class OptunaRunner:
    """A hyperparameter tuning runner using Optuna with SQLite
    storage."""

    def __init__(
        self,
        runner: BacktestRunner,
        storage_url: Optional[str] = None,
    ):
        """
        Initialize the runner with Optuna configuration.

        Args:
            runner: BacktestRunner instance
            storage_url: Optuna storage URL (defaults to sqlite:///optuna_studies.db)
        """
        self.runner = runner
        self.storage_url = storage_url or "sqlite:///optuna_studies.db"
        db_path = self.storage_url.replace("sqlite:///", "")
        console.print(
            Panel(
                f"[bold blue]Optuna Storage URL:[/bold blue] {self.storage_url}\n"
                f"[bold blue]Database file at:[/bold blue] [green]{os.path.abspath(db_path)}[/green]",
                title="[bold yellow]QuantRL-Lab Optuna Runner[/bold yellow]",
                border_style="blue",
            )
        )

    def create_objective_function(
        self,
        algo_class,
        env_config: Union[Dict[str, Any], BacktestEnvironmentConfig],
        search_space: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        optimization_metric: str = "test_avg_return_pct",
    ) -> Callable:
        """
        Create an objective function for Optuna optimization.

        Args:
            algo_class: RL algorithm class (PPO, SAC, A2C, etc.)
            env_config: Environment configuration object or dictionary
            search_space: Dictionary defining the hyperparameter search space
            fixed_params: Fixed parameters that won't be optimized
            total_timesteps: Number of training timesteps
            num_eval_episodes: Number of evaluation episodes
            optimization_metric: Metric to optimize (default: test_avg_return_pct).
                                 Can use any metric from MetricsCalculator (e.g., 'test_avg_sharpe_ratio').

        Returns:
            Callable: The objective function for Optuna.
        """
        fixed_params = fixed_params or {}

        # Normalize env_config
        if isinstance(env_config, dict):
            # Legacy support
            env_config_obj = BacktestEnvironmentConfig.from_dict(env_config)
        else:
            env_config_obj = env_config

        def objective(trial: optuna.Trial) -> float:
            try:
                # Sample hyperparameters from the search space
                params = self._sample_hyperparameters(trial, search_space)
                params.update(fixed_params)

                console.print(
                    f"[bold cyan]Starting Trial {trial.number}[/bold cyan] with params: [yellow]{params}[/yellow]"
                )

                # Create Job
                job = ExperimentJob(
                    algorithm_class=algo_class,
                    env_config=env_config_obj,
                    algorithm_config=params,
                    total_timesteps=total_timesteps,
                    num_eval_episodes=num_eval_episodes,
                    tags={"trial": str(trial.number), "tuner": "optuna"},
                )

                # Run the backtesting experiment
                result = self.runner.run_job(job)

                if result.status == "failed":
                    raise RuntimeError(f"Job failed: {result.error}")

                # Extract the target value for Optuna to optimize
                target_value = result.metrics.get(optimization_metric)
                if target_value is None:
                    console.print(
                        f"[bold yellow]⚠️ Optimization metric '[/bold yellow][cyan]{optimization_metric}[/cyan]"
                        f"[bold yellow]' not found in results.[/bold yellow] "
                        "Defaulting to -1000.0.",
                        style="yellow",
                    )
                    target_value = -1000.0

                console.print(
                    f"[bold green]Trial {trial.number} finished.[/bold green] "
                    f"[blue]{optimization_metric}[/blue] = [cyan]{target_value:.4f}[/cyan] ✓"
                )

                return target_value

            except Exception as e:
                console.print(
                    f"[bold red]❌ Trial {trial.number} failed with an exception:[/bold red] {str(e)}", style="red"
                )
                console.print_exception()
                raise optuna.exceptions.TrialPruned()

        return objective

    def _sample_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from the defined search space."""
        params = {}
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif param_type == "discrete_uniform":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], step=param_config["q"]
                )
            else:
                console.print(
                    f"[bold yellow]⚠️ Unknown parameter type:[/bold yellow] {param_type} for [cyan]{param_name}[/cyan]",
                    style="yellow",
                )
        return params

    def optimize_hyperparameters(
        self,
        algo_class,
        env_config: Dict[str, Any],
        search_space: Dict[str, Any],
        study_name: str,
        n_trials: int = 100,
        fixed_params: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 50000,
        num_eval_episodes: int = 5,
        optimization_metric: str = "test_avg_return_pct",
        direction: str = "maximize",
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """Run hyperparameter optimization using Optuna."""
        sampler = sampler or optuna.samplers.TPESampler(seed=42)
        pruner = pruner or optuna.pruners.MedianPruner()

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        objective_func = self.create_objective_function(
            algo_class=algo_class,
            env_config=env_config,
            search_space=search_space,
            fixed_params=fixed_params,
            total_timesteps=total_timesteps,
            num_eval_episodes=num_eval_episodes,
            optimization_metric=optimization_metric,
        )

        console.rule(
            (f"[bold blue]Starting optimization for {n_trials} " f"trials[/bold blue]"),
        )
        console.print()

        try:
            if n_jobs > 1 and self.storage_url.startswith("sqlite"):
                console.print(
                    "[bold yellow]⚠️ n_jobs > 1 is not safe with SQLite storage "
                    "— falling back to n_jobs=1.[/bold yellow]"
                )
                n_jobs = 1

            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
            )

            console.rule("[bold green]Optimization finished successfully[/bold green]")
            completed = [t for t in study.trials if t.value is not None]
            if completed:
                console.print(f"[bold blue]Best trial value:[/bold blue] [cyan]{study.best_value:.4f}[/cyan]")
                console.print("[bold blue]Best params:[/bold blue]")
                console.print(study.best_params, style="yellow")
            else:
                console.print("[bold yellow]⚠️ No trials completed successfully — all were pruned.[/bold yellow]")

        except Exception as e:
            console.print(f"[bold red]❌ Optimization loop failed with an exception:[/bold red] {str(e)}", style="red")
            console.print_exception()
            raise

        return study


# Utility functions for creating search spaces
def create_ppo_search_space() -> Dict[str, Any]:
    """Create a default search space for PPO hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "categorical", "choices": [256, 512, 1024, 2048, 4096]},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256, 512]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "gae_lambda": {"type": "float", "low": 0.8, "high": 1.0},
        "clip_range": {"type": "float", "low": 0.1, "high": 0.4},
        "ent_coef": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
    }


def create_sac_search_space() -> Dict[str, Any]:
    """Create a default search space for SAC hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "tau": {"type": "float", "low": 0.001, "high": 0.1},
        "train_freq": {"type": "categorical", "choices": [1, 4, 8, 16]},
        "gradient_steps": {"type": "categorical", "choices": [1, 2, 4, 8]},
        "target_update_interval": {"type": "categorical", "choices": [1, 2, 4, 8]},
    }


def create_a2c_search_space() -> Dict[str, Any]:
    """Create a default search space for A2C hyperparameters."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "categorical", "choices": [5, 16, 32, 64, 128]},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "gae_lambda": {"type": "float", "low": 0.8, "high": 1.0},
        "ent_coef": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
        "vf_coef": {"type": "float", "low": 0.1, "high": 1.0},
    }
