from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rich.console import Console

# Keep unused imports for future use, but add noqa comments to suppress warnings
from rich.panel import Panel  # noqa: F401
from rich.rule import Rule  # noqa: F401
from rich.table import Table
from stable_baselines3.common.env_util import make_vec_env

from .evaluation import evaluate_model, get_action_statistics
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
        self.results = {}

    def run_single_experiment(
        self,
        algo_class: type,
        env_config: Dict[str, Callable],
        config: Optional[Dict] = None,
        preset: str = "default",
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Any]:
        """Run a single experiment: train and evaluate a model.

        Args:
            algo_class (type): The algorithm class (e.g., PPO, A2C, SAC)
            env_config (Dict[str, Callable]): Dictionary with 'train_env_factory' and 'test_env_factory' keys
            config (Optional[Dict], optional): Custom configuration parameters for the algorithm.
                If provided, preset will be ignored. Example:
                config = {
                    'learning_rate': 0.001,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'gae_lambda': 0.95
                }
            preset (str, optional): Configuration preset to use if config is None.
                Options: "default", "explorative", "conservative". Defaults to "default".
            total_timesteps (int, optional): Total timesteps for training. Defaults to 50000.
            n_envs (int, optional): Number of parallel environments for training. Defaults to 4.
            num_eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the experiment.

        Example:
            # Create environment config:
            env_config = {
                'train_env_factory': create_train_env,
                'test_env_factory': create_test_env
            }

            # Using a preset:
            results = runner.run_single_experiment(PPO, env_config, preset="explorative")

            # Using custom parameters:
            custom_config = {
                'learning_rate': 0.0005,
                'n_steps': 1024,
                'batch_size': 32,
                'gamma': 0.95
            }
            results = runner.run_single_experiment(PPO, env_config, config=custom_config)
        """

        # Extract factories from env_config
        train_env_factory = env_config["train_env_factory"]
        test_env_factory = env_config["test_env_factory"]

        if self.verbose:
            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            console.print(f"[bold blue]RUNNING SINGLE EXPERIMENT: {algo_class.__name__}[/bold blue]")
            if config is not None:
                console.print(f"[yellow]Using custom configuration with {len(config)} parameters[/yellow]")
                console.print(f"[cyan]Custom params: {list(config.keys())}[/cyan]")
            else:
                console.print(f"[yellow]Using preset: {preset}[/yellow]")
            console.print(f"[yellow]Timesteps: {total_timesteps:,}[/yellow]")
            console.print(f"[bold blue]{'='*60}[/bold blue]")

        # 1. Training Phase
        if self.verbose:
            console.rule("[bold green]🔄 TRAINING PHASE 🔄[/bold green]")
            console.print("[bold green]Starting model training...[/bold green]")

        train_vec_env = make_vec_env(train_env_factory, n_envs=n_envs)

        model = train_model(
            algo_class=algo_class,
            env=train_vec_env,
            config=config,
            preset=preset,
            total_timesteps=total_timesteps,
            verbose=1 if self.verbose else 0,
        )

        train_vec_env.close()

        # 2. Evaluation Phase
        if self.verbose:
            console.rule("[bold blue]📊 EVALUATION PHASE 📊[/bold blue]")

        # Evaluate on training data

        if self.verbose:
            console.print(
                "\n[bold green]🔍 TRAIN EVALUATION:[/bold green] Running model on training dataset", style="green"
            )

        train_env = train_env_factory()
        train_rewards, train_episodes = evaluate_model(
            model=model, env=train_env, num_episodes=num_eval_episodes, verbose=self.verbose
        )
        train_env.close()

        # Evaluate on test data
        if self.verbose:
            console.print(
                "\n[bold blue]🧪 TEST EVALUATION:[/bold blue] Running model on unseen test dataset", style="blue"
            )

        test_env = test_env_factory()
        test_rewards, test_episodes = evaluate_model(
            model=model, env=test_env, num_episodes=num_eval_episodes, verbose=self.verbose
        )
        test_env.close()

        # 3. Calculate metrics
        train_return = self._calculate_average_return(train_episodes)
        test_return = self._calculate_average_return(test_episodes)

        # 4. Get action statistics
        train_action_stats = get_action_statistics(train_episodes)
        test_action_stats = get_action_statistics(test_episodes)

        results = {
            "model": model,
            "algo_class": algo_class.__name__,
            "config": config,
            "preset": preset,
            "total_timesteps": total_timesteps,
            "train_rewards": train_rewards,
            "test_rewards": test_rewards,
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "train_avg_reward": np.mean(train_rewards),
            "test_avg_reward": np.mean(test_rewards),
            "train_avg_return_pct": train_return,
            "test_avg_return_pct": test_return,
            "train_reward_std": np.std(train_rewards),
            "test_reward_std": np.std(test_rewards),
            "train_action_stats": train_action_stats,
            "test_action_stats": test_action_stats,
        }

        if self.verbose:
            console.rule("[bold yellow]📋 RESULTS SUMMARY 📋[/bold yellow]")

            # Create results table
            results_table = Table(title="Experiment Results", show_header=True)
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Train", style="green")
            results_table.add_column("Test", style="blue")

            train_return_color = "green" if train_return >= 0 else "red"
            test_return_color = "green" if test_return >= 0 else "red"

            results_table.add_row(
                "Average Return (%)",
                f"[{train_return_color}]{train_return:.2f}%[/{train_return_color}]",
                f"[{test_return_color}]{test_return:.2f}%[/{test_return_color}]",
            )
            results_table.add_row(
                "Average Reward", f"{results['train_avg_reward']:.2f}", f"{results['test_avg_reward']:.2f}"
            )

            console.print(results_table)

            # Print action statistics
            self._print_action_statistics(train_action_stats, test_action_stats)

        return results

    def run_algorithm_comparison(
        self,
        algorithms: List[type],
        env_config: Dict[str, Callable],
        preset: str = "default",
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple algorithms on the same environment.

        Args:
            algorithms (List[type]): List of algorithm classes to compare
            env_config (Dict[str, Callable]): Dictionary with 'train_env_factory' and 'test_env_factory' keys
            preset (str, optional): Configuration preset to use for all algorithms. Defaults to "default".
            total_timesteps (int, optional): Total timesteps for training. Defaults to 50000.
            n_envs (int, optional): Number of parallel environments for training. Defaults to 4.
            num_eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping algorithm names to their results
        """
        if self.verbose:
            console.print(f"\n[bold blue]{'='*80}[/bold blue]")
            console.print(f"[bold blue]ALGORITHM COMPARISON ({len(algorithms)} algorithms)[/bold blue]")
            console.print(f"[yellow]Preset: {preset}, Timesteps: {total_timesteps:,}[/yellow]")
            console.print(f"[bold blue]{'='*80}[/bold blue]")

        results = {}

        for algo_class in algorithms:
            try:
                algo_results = self.run_single_experiment(
                    algo_class=algo_class,
                    env_config=env_config,
                    preset=preset,
                    total_timesteps=total_timesteps,
                    n_envs=n_envs,
                    num_eval_episodes=num_eval_episodes,
                )
                results[algo_class.__name__] = algo_results

            except Exception as e:
                if self.verbose:
                    console.print(f"[red]ERROR with {algo_class.__name__}: {str(e)}[/red]")
                results[algo_class.__name__] = {"error": str(e)}

        # Print comparison summary
        if self.verbose:
            self._print_algorithm_comparison(results)

        return results

    def run_preset_comparison(
        self,
        algo_class: type,
        env_config: Dict[str, Callable],
        presets: List[str] = None,
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different configuration presets for the same algorithm.

        Args:
            algo_class (type): Algorithm class to test
            env_config (Dict[str, Callable]): Dictionary with 'train_env_factory' and 'test_env_factory' keys
            presets (List[str], optional): List of presets to compare (default: all presets)
            total_timesteps (int, optional): Training timesteps
            n_envs (int, optional): Number of parallel environments
            num_eval_episodes (int, optional): Number of evaluation episodes

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping preset names to their results
        """
        if presets is None:
            presets = ["default", "explorative", "conservative"]

        if self.verbose:
            console.print(f"\n[bold blue]{'='*80}[/bold blue]")
            console.print(f"[bold blue]PRESET COMPARISON: {algo_class.__name__} ({len(presets)} presets)[/bold blue]")
            console.print(f"[yellow]Presets: {', '.join(presets)}[/yellow]")
            console.print(f"[bold blue]{'='*80}[/bold blue]")

        results = {}

        for preset in presets:
            try:
                preset_results = self.run_single_experiment(
                    algo_class=algo_class,
                    env_config=env_config,
                    preset=preset,
                    total_timesteps=total_timesteps,
                    n_envs=n_envs,
                    num_eval_episodes=num_eval_episodes,
                )
                results[preset] = preset_results

            except Exception as e:
                if self.verbose:
                    console.print(f"[red]ERROR with preset {preset}: {str(e)}[/red]")
                results[preset] = {"error": str(e)}

        # Print comparison summary
        if self.verbose:
            self._print_preset_comparison(results, algo_class.__name__)

        return results

    def run_environment_comparison(
        self,
        algo_class: type,
        env_configs: Dict[str, Dict[str, Callable]],
        preset: str = "default",
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare algorithm performance across different environments.

        Args:
            algo_class (type): Algorithm class to test
            env_configs (Dict[str, Dict[str, Callable]]): Dictionary mapping environment names to their configuration
            preset (str, optional): Configuration preset to use. Defaults to "default".
            total_timesteps (int, optional): Training timesteps. Defaults to 50000.
            n_envs (int, optional): Number of parallel environments. Defaults to 4.
            num_eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping environment names to their results
        """
        if self.verbose:
            console.print(f"\n[bold blue]{'='*80}[/bold blue]")
            console.print(
                f"[bold blue]ENVIRONMENT COMPARISON: {algo_class.__name__}[/bold blue]"
                f" ({len(env_configs)} environments)[/bold blue]"
            )
            console.print(f"[yellow]Preset: {preset}, Environments: {', '.join(env_configs.keys())}[/yellow]")
            console.print(f"[bold blue]{'='*80}[/bold blue]")

        results = {}

        for env_name, env_config in env_configs.items():
            try:
                env_results = self.run_single_experiment(
                    algo_class=algo_class,
                    env_config=env_config,
                    preset=preset,
                    total_timesteps=total_timesteps,
                    n_envs=n_envs,
                    num_eval_episodes=num_eval_episodes,
                )
                results[env_name] = env_results

            except Exception as e:
                if self.verbose:
                    console.print(f"[red]ERROR with environment {env_name}: {str(e)}[/red]")
                results[env_name] = {"error": str(e)}

        # Print comparison summary
        if self.verbose:
            self._print_environment_comparison(results, algo_class.__name__)

        return results

    def run_comprehensive_backtest(
        self,
        algorithms: List[type],
        env_configs: Dict[str, Dict[str, Callable]],
        presets: List[str] = None,
        custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run comprehensive backtesting across algorithms, environments,
        and presets/custom configurations.

        Args:
            algorithms (List[type]): List of algorithm classes
            env_configs (Dict[str, Dict[str, Callable]]): Dictionary of environment configurations
            presets (List[str], optional): List of presets to test (default: ["default", "explorative"])
            custom_configs (Dict[str, Dict[str, Any]], optional): Dictionary mapping algorithm names
                to their custom configurations. Can be either:
                1. Direct dictionary: {'PPO': {'learning_rate': 0.001, 'n_steps': 1024}}
                2. Created using create_custom_config:
                {'PPO': BacktestRunner.create_custom_config(PPO, learning_rate=0.001)}

                Examples:
                # Method 1: Direct dictionary
                custom_configs = {
                    'PPO': {'learning_rate': 0.001, 'n_steps': 1024},
                    'SAC': {'learning_rate': 0.0005, 'batch_size': 128}
                }

                # Method 2: Using create_custom_config
                custom_configs = {
                    'PPO': BacktestRunner.create_custom_config(PPO, learning_rate=0.001, n_steps=1024),
                    'SAC': BacktestRunner.create_custom_config(SAC, learning_rate=0.0005, batch_size=128)
                }

                # Method 3: Mixed approach
                custom_configs = {
                    'PPO': BacktestRunner.create_custom_config(PPO, learning_rate=0.001, n_steps=1024),
                    'SAC': {'learning_rate': 0.0005, 'batch_size': 128}  # Direct dict
                }

            total_timesteps (int, optional): Training timesteps. Defaults to 50000.
            n_envs (int, optional): Number of parallel environments. Defaults to 4.
            num_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 5.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Nested dictionary mapping algorithm names
            to environment names to configuration results
        """
        if presets is None:
            presets = ["default", "explorative"]

        if custom_configs is None:
            custom_configs = {}

        # Validate and normalize custom configs
        normalized_custom_configs = {}
        for algo_name, config in custom_configs.items():
            if isinstance(config, dict):
                # Config is already a dictionary (either direct or from create_custom_config)
                normalized_custom_configs[algo_name] = config
            else:
                # Handle edge case where config might be something else
                raise ValueError(f"Custom config for {algo_name} must be a dictionary. Got {type(config)}")

        # Calculate total combinations for progress tracking
        total_combinations = 0
        for algo_class in algorithms:
            algo_name = algo_class.__name__
            if algo_name in normalized_custom_configs:
                # If custom config provided, run once per environment
                total_combinations += len(env_configs)
            else:
                # If no custom config, run for each preset per environment
                total_combinations += len(env_configs) * len(presets)

        if self.verbose:
            console.print(f"\n[bold blue]{'='*100}[/bold blue]")
            console.print("[bold blue]COMPREHENSIVE BACKTESTING[/bold blue]")
            console.print(f"[yellow]Algorithms: {[algo.__name__ for algo in algorithms]}[/yellow]")
            console.print(f"[yellow]Environments: {list(env_configs.keys())}[/yellow]")

            # Show configuration info
            custom_algos = [algo for algo in algorithms if algo.__name__ in normalized_custom_configs]
            preset_algos = [algo for algo in algorithms if algo.__name__ not in normalized_custom_configs]

            if custom_algos:
                console.print(f"[yellow]Custom configs: {[algo.__name__ for algo in custom_algos]}[/yellow]")
                # Show which parameters are being customized
                for algo in custom_algos:
                    algo_name = algo.__name__
                    config_params = list(normalized_custom_configs[algo_name].keys())
                    console.print(f"[dim]  {algo_name}: {config_params}[/dim]")

            if preset_algos:
                console.print(
                    f"[yellow]Preset configs: {[algo.__name__ for algo in preset_algos]} (presets: {presets})[/yellow]"
                )

            console.print(f"[cyan]Total combinations: {total_combinations}[/cyan]")
            console.print(f"[bold blue]{'='*100}[/bold blue]")

        all_results = {}

        for algo_class in algorithms:
            algo_name = algo_class.__name__
            all_results[algo_name] = {}

            for env_name, env_config in env_configs.items():
                all_results[algo_name][env_name] = {}

                # Check if custom config is provided for this algorithm
                if algo_name in normalized_custom_configs:
                    # Use custom configuration
                    custom_config = normalized_custom_configs[algo_name]
                    config_name = "custom"

                    if self.verbose:
                        console.print(f"\n[cyan]Running: {algo_name} + {env_name} + custom config[/cyan]")
                        console.print(f"[dim]Custom params: {list(custom_config.keys())}[/dim]")
                        # Show parameter values for debugging
                        for param, value in custom_config.items():
                            console.print(f"[dim]  {param}: {value}[/dim]")

                    try:
                        results = self.run_single_experiment(
                            algo_class=algo_class,
                            env_config=env_config,
                            config=custom_config,  # Pass custom config
                            preset=None,  # Preset is ignored when config is provided
                            total_timesteps=total_timesteps,
                            n_envs=n_envs,
                            num_eval_episodes=num_eval_episodes,
                        )
                        all_results[algo_name][env_name][config_name] = results

                    except Exception as e:
                        if self.verbose:
                            console.print(f"[red]ERROR: {str(e)}[/red]")
                        all_results[algo_name][env_name][config_name] = {"error": str(e)}

                else:
                    # Use presets (existing behavior)
                    for preset in presets:
                        if self.verbose:
                            console.print(f"\n[cyan]Running: {algo_name} + {env_name} + {preset}[/cyan]")

                        try:
                            results = self.run_single_experiment(
                                algo_class=algo_class,
                                env_config=env_config,
                                preset=preset,
                                total_timesteps=total_timesteps,
                                n_envs=n_envs,
                                num_eval_episodes=num_eval_episodes,
                            )
                            all_results[algo_name][env_name][preset] = results

                        except Exception as e:
                            if self.verbose:
                                console.print(f"[red]ERROR: {str(e)}[/red]")
                            all_results[algo_name][env_name][preset] = {"error": str(e)}

        # Print final summary
        if self.verbose:
            self._print_comprehensive_summary(all_results)

        return all_results

    @staticmethod
    def create_custom_config(algo_class: type, **kwargs) -> Dict[str, Any]:
        """
        Create a custom configuration dictionary for an algorithm.

        Args:
            algo_class (type): The algorithm class (PPO, A2C, SAC)
            **kwargs: Algorithm-specific parameters

        Returns:
            Dict[str, Any]: Configuration dictionary

        Example:
            # For PPO
            config = BacktestRunner.create_custom_config(
                PPO,
                learning_rate=0.001,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )

            # For A2C
            config = BacktestRunner.create_custom_config(
                A2C,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25
            )

            # For SAC
            config = BacktestRunner.create_custom_config(
                SAC,
                learning_rate=0.0003,
                buffer_size=1000000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef='auto'
            )
        """
        return kwargs

    def _calculate_average_return(self, episodes: List[Dict[str, Any]]) -> float:
        """
        Calculate average return percentage from episodes.

        Args:
            episodes (List[Dict[str, Any]]): List of episode results.

        Returns:
            float: Average return percentage.
        """
        valid_episodes = [ep for ep in episodes if "error" not in ep]
        if not valid_episodes:
            return 0.0

        returns = [((ep["final_value"] - ep["initial_value"]) / ep["initial_value"]) * 100 for ep in valid_episodes]
        return np.mean(returns)

    def _print_algorithm_comparison(self, results: Dict[str, Dict[str, Any]]):
        """
        Print algorithm comparison summary.

        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of algorithm results.
        """
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print("[bold magenta]ALGORITHM COMPARISON SUMMARY[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        # Create comparison table
        comparison_table = Table(title="Algorithm Performance", show_header=True, header_style="bold magenta")
        comparison_table.add_column("Algorithm", style="cyan", no_wrap=True)
        comparison_table.add_column("Train Return %", justify="right", style="green")
        comparison_table.add_column("Test Return %", justify="right", style="blue")
        comparison_table.add_column("Train Reward", justify="right", style="yellow")
        comparison_table.add_column("Test Reward", justify="right", style="yellow")
        comparison_table.add_column("Top Train Action", style="magenta")
        comparison_table.add_column("Top Test Action", style="magenta")

        for algo_name, result in results.items():
            if "error" not in result:
                train_return_color = "green" if result["train_avg_return_pct"] >= 0 else "red"
                test_return_color = "green" if result["test_avg_return_pct"] >= 0 else "red"

                # Get top actions
                train_actions = result.get("train_action_stats", {}).get("action_percentages", {})
                test_actions = result.get("test_action_stats", {}).get("action_percentages", {})

                top_train_action = max(train_actions, key=train_actions.get) if train_actions else "N/A"
                top_test_action = max(test_actions, key=test_actions.get) if test_actions else "N/A"

                top_train_pct = train_actions.get(top_train_action, 0.0) if train_actions else 0.0
                top_test_pct = test_actions.get(top_test_action, 0.0) if test_actions else 0.0

                comparison_table.add_row(
                    algo_name,
                    f"[{train_return_color}]{result['train_avg_return_pct']:.2f}%[/{train_return_color}]",
                    f"[{test_return_color}]{result['test_avg_return_pct']:.2f}%[/{test_return_color}]",
                    f"{result['train_avg_reward']:.2f}",
                    f"{result['test_avg_reward']:.2f}",
                    f"{top_train_action} ({top_train_pct:.1f}%)",
                    f"{top_test_action} ({top_test_pct:.1f}%)",
                )
            else:
                comparison_table.add_row(
                    algo_name,
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                )

        console.print(comparison_table)

    def _print_preset_comparison(self, results: Dict[str, Dict[str, Any]], algo_name: str):
        """
        Print preset comparison summary.

        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of preset results.
            algo_name (str): Name of the algorithm.
        """
        console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
        console.print("[bold magenta]PRESET COMPARISON SUMMARY - " + algo_name + "[/bold magenta]")
        console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

        # Create comparison table
        preset_table = Table(title=f"Preset Performance - {algo_name}", show_header=True, header_style="bold magenta")
        preset_table.add_column("Preset", style="cyan", no_wrap=True)
        preset_table.add_column("Train Return %", justify="right", style="green")
        preset_table.add_column("Test Return %", justify="right", style="blue")
        preset_table.add_column("Train Reward", justify="right", style="yellow")
        preset_table.add_column("Test Reward", justify="right", style="yellow")
        preset_table.add_column("Top Train Action", style="magenta")
        preset_table.add_column("Top Test Action", style="magenta")

        for preset, result in results.items():
            if "error" not in result:
                train_return_color = "green" if result["train_avg_return_pct"] >= 0 else "red"
                test_return_color = "green" if result["test_avg_return_pct"] >= 0 else "red"

                # Get top actions
                train_actions = result.get("train_action_stats", {}).get("action_percentages", {})
                test_actions = result.get("test_action_stats", {}).get("action_percentages", {})

                top_train_action = max(train_actions, key=train_actions.get) if train_actions else "N/A"
                top_test_action = max(test_actions, key=test_actions.get) if test_actions else "N/A"

                top_train_pct = train_actions.get(top_train_action, 0.0) if train_actions else 0.0
                top_test_pct = test_actions.get(top_test_action, 0.0) if test_actions else 0.0

                preset_table.add_row(
                    preset,
                    f"[{train_return_color}]{result['train_avg_return_pct']:.2f}%[/{train_return_color}]",
                    f"[{test_return_color}]{result['test_avg_return_pct']:.2f}%[/{test_return_color}]",
                    f"{result['train_avg_reward']:.2f}",
                    f"{result['test_avg_reward']:.2f}",
                    f"{top_train_action} ({top_train_pct:.1f}%)",
                    f"{top_test_action} ({top_test_pct:.1f}%)",
                )
            else:
                preset_table.add_row(
                    preset,
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                )

        console.print(preset_table)

    def _print_environment_comparison(self, results: Dict[str, Dict[str, Any]], algo_name: str):
        """
        Print environment comparison summary.

        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of environment results.
            algo_name (str): Name of the algorithm.
        """
        console.print("\n[bold magenta]" + "=" * 80 + "[/bold magenta]")
        console.print("[bold magenta]ENVIRONMENT COMPARISON SUMMARY - " + algo_name + "[/bold magenta]")
        console.print("[bold magenta]" + "=" * 80 + "[/bold magenta]")

        # Create comparison table
        env_table = Table(title=f"Environment Performance - {algo_name}", show_header=True, header_style="bold magenta")
        env_table.add_column("Environment", style="cyan", no_wrap=True)
        env_table.add_column("Train Return %", justify="right", style="green")
        env_table.add_column("Test Return %", justify="right", style="blue")
        env_table.add_column("Train Reward", justify="right", style="yellow")
        env_table.add_column("Test Reward", justify="right", style="yellow")
        env_table.add_column("Top Train Action", style="magenta")
        env_table.add_column("Top Test Action", style="magenta")

        for env_name, result in results.items():
            if "error" not in result:
                train_return_color = "green" if result["train_avg_return_pct"] >= 0 else "red"
                test_return_color = "green" if result["test_avg_return_pct"] >= 0 else "red"

                # Get top actions
                train_actions = result.get("train_action_stats", {}).get("action_percentages", {})
                test_actions = result.get("test_action_stats", {}).get("action_percentages", {})

                top_train_action = max(train_actions, key=train_actions.get) if train_actions else "N/A"
                top_test_action = max(test_actions, key=test_actions.get) if test_actions else "N/A"

                top_train_pct = train_actions.get(top_train_action, 0.0) if train_actions else 0.0
                top_test_pct = test_actions.get(top_test_action, 0.0) if test_actions else 0.0

                env_table.add_row(
                    env_name,
                    f"[{train_return_color}]{result['train_avg_return_pct']:.2f}%[/{train_return_color}]",
                    f"[{test_return_color}]{result['test_avg_return_pct']:.2f}%[/{test_return_color}]",
                    f"{result['train_avg_reward']:.2f}",
                    f"{result['test_avg_reward']:.2f}",
                    f"{top_train_action} ({top_train_pct:.1f}%)",
                    f"{top_test_action} ({top_test_pct:.1f}%)",
                )
            else:
                env_table.add_row(
                    env_name,
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                )

        console.print(env_table)

    def _print_comprehensive_summary(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Print comprehensive backtesting summary.

        Args:
            results (Dict[str, Dict[str, Dict[str, Any]]]): Dictionary of comprehensive results.
        """
        console.print(f"\n[bold magenta]{'='*100}[/bold magenta]")
        console.print("[bold magenta]COMPREHENSIVE BACKTESTING SUMMARY[/bold magenta]")
        console.print(f"[bold magenta]{'='*100}[/bold magenta]")

        # Create comprehensive table
        comprehensive_table = Table(
            title="Comprehensive Backtesting Results", show_header=True, header_style="bold magenta"
        )
        comprehensive_table.add_column("Algorithm", style="cyan", no_wrap=True)
        comprehensive_table.add_column("Environment", style="blue", no_wrap=True)
        comprehensive_table.add_column("Preset", style="yellow", no_wrap=True)
        comprehensive_table.add_column("Train Return %", justify="right", style="green")
        comprehensive_table.add_column("Test Return %", justify="right", style="green")
        comprehensive_table.add_column("Top Train Action", style="magenta")
        comprehensive_table.add_column("Top Test Action", style="magenta")

        for algo_name, algo_results in results.items():
            for env_name, env_results in algo_results.items():
                for preset, result in env_results.items():
                    if "error" not in result:
                        train_return_color = "green" if result["train_avg_return_pct"] >= 0 else "red"
                        test_return_color = "green" if result["test_avg_return_pct"] >= 0 else "red"

                        # Get top actions
                        train_actions = result.get("train_action_stats", {}).get("action_percentages", {})
                        test_actions = result.get("test_action_stats", {}).get("action_percentages", {})

                        top_train_action = max(train_actions, key=train_actions.get) if train_actions else "N/A"
                        top_test_action = max(test_actions, key=test_actions.get) if test_actions else "N/A"

                        top_train_pct = train_actions.get(top_train_action, 0.0) if train_actions else 0.0
                        top_test_pct = test_actions.get(top_test_action, 0.0) if test_actions else 0.0

                        comprehensive_table.add_row(
                            algo_name,
                            env_name,
                            preset,
                            f"[{train_return_color}]{result['train_avg_return_pct']:.2f}%[/{train_return_color}]",
                            f"[{test_return_color}]{result['test_avg_return_pct']:.2f}%[/{test_return_color}]",
                            f"{top_train_action} ({top_train_pct:.1f}%)",
                            f"{top_test_action} ({top_test_pct:.1f}%)",
                        )
                    else:
                        comprehensive_table.add_row(
                            algo_name,
                            env_name,
                            preset,
                            "[red]ERROR[/red]",
                            "[red]ERROR[/red]",
                            "[red]ERROR[/red]",
                            "[red]ERROR[/red]",
                        )

        console.print(comprehensive_table)

    def _print_action_statistics(self, train_action_stats: Dict[str, Any], test_action_stats: Dict[str, Any]):
        """
        Print action statistics for train and test phases.

        Args:
            train_action_stats (Dict[str, Any]): Training action statistics.
            test_action_stats (Dict[str, Any]): Testing action statistics.
        """
        console.print("\n[bold cyan]Action Statistics:[/bold cyan]")

        # Create action statistics table
        action_table = Table(title="Action Distribution", show_header=True)
        action_table.add_column("Action", style="cyan", no_wrap=True)
        action_table.add_column("Train Count", justify="right", style="green")
        action_table.add_column("Train %", justify="right", style="green")
        action_table.add_column("Test Count", justify="right", style="blue")
        action_table.add_column("Test %", justify="right", style="blue")

        # Get all unique actions from both train and test
        all_actions = set()
        all_actions.update(train_action_stats.get("action_counts", {}).keys())
        all_actions.update(test_action_stats.get("action_counts", {}).keys())

        for action in sorted(all_actions):
            train_count = train_action_stats.get("action_counts", {}).get(action, 0)
            train_pct = train_action_stats.get("action_percentages", {}).get(action, 0.0)
            test_count = test_action_stats.get("action_counts", {}).get(action, 0)
            test_pct = test_action_stats.get("action_percentages", {}).get(action, 0.0)

            action_table.add_row(action, str(train_count), f"{train_pct:.1f}%", str(test_count), f"{test_pct:.1f}%")

        console.print(action_table)

        # Print total steps summary
        train_steps = train_action_stats.get("total_steps", 0)
        test_steps = test_action_stats.get("total_steps", 0)

        steps_table = Table(title="Step Summary", show_header=True)
        steps_table.add_column("Phase", style="cyan")
        steps_table.add_column("Total Steps", justify="right", style="yellow")

        steps_table.add_row("Training", str(train_steps))
        steps_table.add_row("Testing", str(test_steps))

        console.print(steps_table)

    # Backward compatibility methods (deprecated - will be removed in future versions)
    def run_single_experiment_legacy(
        self,
        algo_class: type,
        train_env_factory: Callable,
        test_env_factory: Callable,
        config: Optional[Dict] = None,
        preset: str = "default",
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use run_single_experiment with env_config instead.

        Legacy method for backward compatibility.
        """
        console.print(
            "[yellow]WARNING: run_single_experiment_legacy is deprecated. "
            "Use run_single_experiment with env_config instead.[/yellow]"
        )

        env_config = {"train_env_factory": train_env_factory, "test_env_factory": test_env_factory}

        return self.run_single_experiment(
            algo_class=algo_class,
            env_config=env_config,
            config=config,
            preset=preset,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            num_eval_episodes=num_eval_episodes,
        )

    def run_algorithm_comparison_legacy(
        self,
        algorithms: List[type],
        train_env_factory: Callable,
        test_env_factory: Callable,
        preset: str = "default",
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        DEPRECATED: Use run_algorithm_comparison with env_config instead.

        Legacy method for backward compatibility.
        """
        console.print(
            "[yellow]WARNING: run_algorithm_comparison_legacy is deprecated. "
            "Use run_algorithm_comparison with env_config instead.[/yellow]"
        )

        env_config = {"train_env_factory": train_env_factory, "test_env_factory": test_env_factory}

        return self.run_algorithm_comparison(
            algorithms=algorithms,
            env_config=env_config,
            preset=preset,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            num_eval_episodes=num_eval_episodes,
        )

    def run_preset_comparison_legacy(
        self,
        algo_class: type,
        train_env_factory: Callable,
        test_env_factory: Callable,
        presets: List[str] = None,
        total_timesteps: int = 50000,
        n_envs: int = 4,
        num_eval_episodes: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        DEPRECATED: Use run_preset_comparison with env_config instead.

        Legacy method for backward compatibility.
        """
        console.print(
            "[yellow]WARNING: run_preset_comparison_legacy is deprecated. "
            "Use run_preset_comparison with env_config instead.[/yellow]"
        )

        env_config = {"train_env_factory": train_env_factory, "test_env_factory": test_env_factory}

        return self.run_preset_comparison(
            algo_class=algo_class,
            env_config=env_config,
            presets=presets,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            num_eval_episodes=num_eval_episodes,
        )

    @staticmethod
    def create_env_config(train_env_factory: Callable, test_env_factory: Callable) -> Dict[str, Callable]:
        """
        Helper method to create env_config from individual factory
        functions.

        Args:
            train_env_factory (Callable): Function that creates training environment
            test_env_factory (Callable): Function that creates test environment

        Returns:
            Dict[str, Callable]: Environment configuration dictionary

        Example:
            env_config = BacktestRunner.create_env_config(
                train_env_factory=create_train_env,
                test_env_factory=create_test_env
            )
        """
        return {"train_env_factory": train_env_factory, "test_env_factory": test_env_factory}

    # API Usage Examples (Updated for Consistent Design)
    """
    CONSISTENT API USAGE EXAMPLES:

    # 1. Create environment configuration (used by all methods)
    env_config = {
        'train_env_factory': create_train_env,
        'test_env_factory': create_test_env
    }

    # 2. Single experiment
    results = runner.run_single_experiment(PPO, env_config, preset="default")

    # 3. Algorithm comparison
    results = runner.run_algorithm_comparison([PPO, A2C, SAC], env_config, preset="default")

    # 4. Preset comparison
    results = runner.run_preset_comparison(PPO, env_config, presets=["default", "explorative"])

    # 5. Environment comparison (multiple environments)
    env_configs = {
        "conservative": {
            'train_env_factory': create_conservative_train_env,
            'test_env_factory': create_conservative_test_env
        },
        "aggressive": {
            'train_env_factory': create_aggressive_train_env,
            'test_env_factory': create_aggressive_test_env
        }
    }
    results = runner.run_environment_comparison(PPO, env_configs, preset="default")

    # 6. Comprehensive backtest
    results = runner.run_comprehensive_backtest([PPO, A2C], env_configs, presets=["default", "explorative"])

    # MIGRATION GUIDE:
    # Old API (deprecated):
    # runner.run_single_experiment(PPO, train_factory, test_factory)
    #
    # New API (consistent):
    # env_config = {'train_env_factory': train_factory, 'test_env_factory': test_factory}
    # runner.run_single_experiment(PPO, env_config)
    """
