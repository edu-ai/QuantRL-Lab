"""Tests for quantrl_lab.utils.math."""

import pytest

from quantrl_lab.utils.math import generate_weights


class TestGenerateWeights:
    def test_output_length(self):
        result = generate_weights(n_combinations=10, strategies_count=3)
        assert len(result) == 10

    def test_weights_sum_to_one(self):
        result = generate_weights(n_combinations=20, strategies_count=5)
        for weights in result:
            assert sum(weights) == pytest.approx(1.0, abs=1e-9)

    def test_strategies_count(self):
        result = generate_weights(n_combinations=5, strategies_count=7)
        for weights in result:
            assert len(weights) == 7

    def test_all_weights_non_negative(self):
        result = generate_weights(n_combinations=50, strategies_count=4)
        for weights in result:
            assert all(w >= 0.0 for w in weights)

    def test_default_params(self):
        result = generate_weights()
        assert len(result) == 100
        for weights in result:
            assert len(weights) == 5
            assert sum(weights) == pytest.approx(1.0, abs=1e-9)

    def test_high_alpha_concentrates_weights(self):
        """High alpha → weights closer to equal (1/n)."""
        import numpy as np

        result = generate_weights(n_combinations=200, strategies_count=4, alpha=1000.0)
        deviations = []
        for weights in result:
            expected = 1.0 / 4
            deviations.append(max(abs(w - expected) for w in weights))
        assert np.mean(deviations) < 0.05  # On average very close to equal weights

    def test_single_strategy(self):
        result = generate_weights(n_combinations=5, strategies_count=1)
        for weights in result:
            assert weights == pytest.approx([1.0])

    def test_returns_list_of_lists(self):
        result = generate_weights(n_combinations=3, strategies_count=2)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, list)
