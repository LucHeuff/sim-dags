import numpy as np
import pandera.polars as pa

from sim_dags.core import Generator


class MockGenerator(Generator):
    """Mock Generator, used for testing purposes."""

    def __init__(self, name: str, ancestors: list[str] | None = None) -> None:
        self.name = name
        self.ancestors = ancestors if ancestors is not None else []
        self.column_schema = pa.Column(int)

    def __call__(
        self,
        inputs: list[np.ndarray] | None,  # noqa: ARG002
        size: int,
        seed: int,
    ) -> np.ndarray:
        """Generate random numbers for testing purposes."""
        rng = np.random.default_rng(seed)
        return rng.integers(1, 100, size=size)

    def do(
        self,
        value: bool | float | np.ndarray,  # noqa: ARG002, FBT001
        size: int,
        seed: int,
    ) -> np.ndarray:
        """Generate random negative numbers for testing purposes."""
        rng = np.random.default_rng(seed)
        return rng.integers(-100, -1, size=size)
