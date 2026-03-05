# Strategies

## Base Interfaces

::: quantrl_lab.environments.core.interfaces
    options:
        show_root_heading: true
        members:
            - BaseActionStrategy
            - BaseObservationStrategy
            - BaseRewardStrategy

## Action Strategies

::: quantrl_lab.environments.stock.strategies.actions.standard
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.actions.time_in_force
    options:
        show_root_heading: true

## Observation Strategies

::: quantrl_lab.environments.stock.strategies.observations.feature_aware
    options:
        show_root_heading: true

## Reward Strategies

::: quantrl_lab.environments.stock.strategies.rewards.portfolio_value
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.sortino
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.sharpe
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.drawdown
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.turnover
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.invalid_action
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.boredom
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.execution_bonus
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.expiration
    options:
        show_root_heading: true

::: quantrl_lab.environments.stock.strategies.rewards.composite
    options:
        show_root_heading: true
