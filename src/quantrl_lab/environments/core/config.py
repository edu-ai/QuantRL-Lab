from typing import Optional

from pydantic import BaseModel, Field


class CoreEnvConfig(BaseModel):
    """Core environment configuration."""

    initial_balance: float = Field(..., gt=0, description="The initial cash balance for the agent.")
    window_size: int = Field(..., gt=0, description="The size of the observation window.")
    price_column_index: int = Field(..., ge=0, description="The column index for the price data.")
    max_episode_steps: Optional[int] = Field(default=None, description="Limit for the episode steps.")

    class Config:
        from_attributes = True
