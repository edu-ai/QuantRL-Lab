import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .config.environment_config import BacktestEnvironmentConfig


@dataclass
class ExperimentJob:
    """
    Defines a single atomic unit of work for the backtesting engine.

    Contains all necessary information to run and reproduce a specific
    experiment.
    """

    algorithm_class: Type
    env_config: BacktestEnvironmentConfig

    # Explicit algorithm configuration (e.g. {'learning_rate': 0.001})
    # If empty, uses the algorithm's default hyperparameters.
    algorithm_config: Dict[str, Any] = field(default_factory=dict)

    # Run parameters
    total_timesteps: int = 50000
    n_envs: int = 4
    num_eval_episodes: int = 5

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure tags exist
        if not self.tags:
            self.tags = {
                "algo": self.algorithm_class.__name__,
                "env": self.env_config.name,
            }


@dataclass
class ExperimentResult:
    """Standardized result object containing all artifacts from a
    job."""

    job: ExperimentJob
    metrics: Dict[str, float]  # Flattened metrics (train_return, test_sharpe, etc.)

    # Artifacts
    model: Any = None  # The trained model object
    train_episodes: List[Dict] = field(default_factory=list)
    test_episodes: List[Dict] = field(default_factory=list)
    top_features: Dict[str, float] = field(default_factory=dict)
    explanation_method: str = "Correlation"

    # Metadata
    status: str = "completed"  # completed, failed
    error: Optional[Exception] = None
    execution_time: float = 0.0


class JobGenerator:
    """Helper to generate combinatorial lists of jobs."""

    @staticmethod
    def generate_grid(
        algorithms: List[Type],
        env_configs: Dict[str, BacktestEnvironmentConfig],
        algorithm_configs: Optional[List[Dict[str, Any]]] = None,
        **job_kwargs,
    ) -> List[ExperimentJob]:
        """
        Generate a grid of experiments.

        Args:
            algorithms: List of algorithm classes
            env_configs: Dictionary of name -> BacktestEnvironmentConfig
            algorithm_configs: List of configuration dictionaries to try.
                             If None, uses a single empty dict (defaults).
            **job_kwargs: Common arguments for all jobs (total_timesteps, etc.)

        Returns:
            List[ExperimentJob]: List of jobs to be executed
        """
        if algorithm_configs is None:
            algorithm_configs = [{}]  # Single default run

        jobs = []
        for algo in algorithms:
            for env_name, env_conf in env_configs.items():
                for i, config in enumerate(algorithm_configs):
                    # Create tags
                    tags = {
                        "algo": algo.__name__,
                        "env": env_name,
                        "config_id": str(i) if len(algorithm_configs) > 1 else "default",
                    }

                    job = ExperimentJob(
                        algorithm_class=algo,
                        env_config=env_conf,
                        algorithm_config=config,
                        tags=tags,
                        **job_kwargs,
                    )
                    jobs.append(job)
        return jobs
