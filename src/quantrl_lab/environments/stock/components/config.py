from typing import Tuple

from pydantic import BaseModel, Field

from quantrl_lab.environments.core.config import CoreEnvConfig


class SimulationConfig(BaseModel):
    """Configuration for market simulation parameters."""

    transaction_cost_pct: float = Field(
        default=0.001, ge=0, lt=1, description="The percentage fee for each transaction."
    )
    slippage: float = Field(default=0.001, ge=0, lt=1, description="The slippage percentage for market orders.")
    order_expiration_steps: int = Field(
        default=5, gt=0, description="The number of steps before a pending order expires."
    )
    enable_shorting: bool = Field(default=False, description="Whether to allow short selling.")
    ignore_fees: bool = Field(default=False, description="Whether to ignore transaction costs.")


class RewardConfig(BaseModel):
    """Configuration for reward calculation parameters."""

    clip_range: Tuple[float, float] = Field(default=(-1.0, 1.0), description="Range to clip the final reward.")


class SingleStockEnvConfig(CoreEnvConfig):
    """Stock environment configuration, extending the core environment
    configuration."""

    # Core Defaults
    initial_balance: float = 100000.0
    window_size: int = 20
    price_column_index: int = 0

    # Components
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    class Config:
        from_attributes = True  # "ORM Mode"
