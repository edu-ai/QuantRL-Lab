from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

console = Console()


def train_model(
    algo_class: Union[type, str],
    env: VecEnv,
    config: Optional[Union[Dict, Any]] = None,
    total_timesteps: int = 10000,
    policy: str = "MlpPolicy",
    verbose: int = 1,
    suppress_logs: bool = False,
    **kwargs,
) -> BaseAlgorithm:
    """
    Train a reinforcement learning model using the specified algorithm.

    Args:
        algo_class: The algorithm class (PPO, A2C, SAC, etc.)
        env: The vectorized environment to train on
        config: Configuration object or dictionary with algorithm parameters
        total_timesteps: Number of timesteps to train for
        policy: Policy architecture to use
        verbose: Verbosity level
        suppress_logs: Whether to suppress training logs
        **kwargs: Additional parameters to pass to the algorithm

    Returns:
        Trained model
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Training model:[/bold green] [cyan]{task.description}[/cyan]"),
        console=console,
        disable=not verbose,
    ) as progress:
        if verbose:
            task = progress.add_task(
                f"[bold]{algo_class.__name__}[/bold] for [yellow]{total_timesteps}[/yellow] timesteps", total=None
            )

        # Build model parameters
        base_params = {
            "policy": policy,
            "env": env,
            "verbose": 0 if suppress_logs else verbose,  # Force verbose=0 when suppressing logs
        }

        # Handle both dict and config objects
        if config is not None:
            if isinstance(config, dict):
                config_dict = config.copy()  # It's already a dict
            else:
                config_dict = config.__dict__.copy()  # It's a config object
            base_params.update(config_dict)
        # Override with any additional kwargs
        base_params.update(kwargs)

        # Create and train the model
        model = algo_class(**base_params)
        model.learn(total_timesteps=total_timesteps)

        if verbose:
            progress.update(task, description="[green]✓ Training completed")

    return model
