from typing import List

import numpy as np


def generate_weights(n_combinations: int = 100, strategies_count: int = 5, alpha: float = 1.0) -> List[list]:
    """
    Generate random weight combinations using Dirichlet distribution.

    Args:
        n_combinations (int, optional): Number of combinations to generate. Defaults to 100.
        strategies_count (int, optional): Number of strategies (weights) in each combination. Defaults to 5.
        alpha (float, optional): Concentration parameter for Dirichlet distribution. Defaults to 1.0.

    Returns:
        List[list]: A list of weight combinations, each a list of floats.
    """
    combinations = []

    for _ in range(n_combinations):
        # alpha=1.0 gives uniform distribution, higher values concentrate around equal weights
        weights = np.random.dirichlet([alpha] * strategies_count)
        combinations.append(weights.tolist())

    return combinations
