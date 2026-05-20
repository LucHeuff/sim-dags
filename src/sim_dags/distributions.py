from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Categorical:
    """Categorical distribution."""

    name: str
    categories: int = Field(ge=1)
    parents: list[str] = Field(default_factory=list)
    unobserved: bool = False
    param_seed: int | None = None

    @field_validator("parents")
    @classmethod
    def unique_parents(cls, p: list[str]) -> list[str]:
        """Validate that parents are unique."""
        if len(p) != len(set(p)):
            msg = "parents must be unique."
            raise ValueError(msg)
        return p


@dataclass(frozen=True, slots=True)
class Binomial:
    """Binomial distribution, always has 2 categories."""

    name: str
    parents: list[str] = Field(default_factory=list)
    categories: int = Field(default=2, init=False)
    unobserved: bool = False
    param_seed: int | None = None

    @field_validator("parents")
    @classmethod
    def unique_parents(cls, p: list[str]) -> list[str]:
        """Validate that parents are unique."""
        if len(p) != len(set(p)):
            msg = "parents must be unique."
            raise ValueError(msg)
        return p
