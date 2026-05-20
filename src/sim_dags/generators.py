from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from sim_dags.distributions import Binomial, Categorical
from sim_dags.exceptions import InvalidDoValueError

# ---- Supporting functions


def get_do_name(name: str) -> str:  # pragma: no cover
    """Translate name to do(name)."""
    return f"do({name})"


def do_fixed(value: int, size: int) -> np.ndarray:
    """Intervene on a variable with a fixed value."""
    return np.repeat(value, size)


def do_uniform(categories: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Intervene on a variable uniformly."""
    values = np.arange(categories)
    counts = np.full(categories, size // categories)
    # rounded down, so might need to add on remainder
    counts[: size % categories] += 1
    assert (s := sum(counts)) == size, f"SUm adds to {s}, not {size}"

    samples = np.repeat(values, counts)
    rng.shuffle(samples)
    return samples


# --- Generator definitions


class Distribution(Protocol):
    """Interface for distributions."""

    name: str
    categories: int
    parents: list[str]
    unobserved: bool
    param_seed: int | None


class Generator(ABC):
    """Interface for generators."""

    distribution: Distribution
    parameters: np.ndarray

    @property
    def do_name(self) -> str:
        """Get name of intervened variable."""
        return get_do_name(self.distribution.name)

    @property
    def parents(self) -> int:
        """Get the number of ancestors for this variable."""
        return len(self.distribution.parents)

    @property
    def categories(self) -> int:
        """Get the number of categories for this variable."""
        return self.distribution.categories

    @property
    def name(self) -> str:
        """Get the name of this variable."""
        return self.distribution.name

    def _check_inputs(self, inputs: np.ndarray) -> None:
        """Check if inputs have the expected length."""
        shape = inputs.shape[0]
        assert shape == self.parents, (
            f"Got {shape} inputs when '{self.name}' has {self.parents} parents."
        )

    def _check_samples(self, samples: np.ndarray, size: int) -> np.ndarray:
        """Check if samples have the desired shape."""
        shape = samples.shape
        size_ = (size,)
        assert shape == size_, (
            f"Incorrect shape for samples of '{self.name}', got {shape} when expecting {size_}"  # noqa: E501
        )
        return samples

    def _check_values(self, value: int) -> None:
        """Check the desired value is valid."""
        categories = list(range(self.distribution.categories))
        if value not in categories:
            msg = f"Available categories for {self.name} are {categories}, but got do({self.name}={value})"  # noqa: E501
            raise InvalidDoValueError(msg)

    @abstractmethod
    def sample(
        self, inputs: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate samples without intervention."""
        ...

    def do(
        self,
        value: bool | int,  # noqa: FBT001
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate smaples under intervention."""
        if not isinstance(value, bool):
            self._check_values(value)
            samples = do_fixed(value, size)
        else:
            samples = do_uniform(self.categories, size, rng)
        return self._check_samples(samples, size)


class CategoricalGenerator(Generator):
    """Generates categorical samples for a single variable."""

    distribution: Categorical

    def __init__(
        self,
        variable: Categorical,
        parents: list[Distribution],
        alpha: int,
        rng: np.random.Generator,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable

        shape = [p.categories for p in parents] if len(parents) > 0 else ()
        # resetting the random generator if this Categorical has a fixed seed
        if variable.param_seed is not None:
            rng = np.random.default_rng(variable.param_seed)

        # dirichlet distribution with number of categories of current variable
        # as last dimension
        categories = self.distribution.categories
        self.parameters = rng.dirichlet(np.repeat(alpha, categories), size=shape)

    def sample(
        self, inputs: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate categorical samples without intervention."""
        self._check_inputs(inputs)
        p = self.parameters[*inputs] if self.parents != 0 else self.parameters
        s = None if self.parents != 0 else size
        samples = rng.multinomial(1, pvals=p, size=s).argmax(axis=1)
        return self._check_samples(samples, size)


class BinomialGenerator(Generator):
    """Generates binomial samples for a single variable."""

    distribution: Binomial

    def __init__(
        self,
        variable: Binomial,
        parents: list[Distribution],
        rng: np.random.Generator,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable
        shape = [p.categories for p in parents] if len(parents) > 0 else ()

        # resetting the random generator if this Binomial has a fixed seed
        if variable.param_seed is not None:
            rng = np.random.default_rng(variable.param_seed)
        self.parameters = rng.uniform(size=shape)

    def sample(
        self, inputs: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate binomial samples without intervention."""
        self._check_inputs(inputs)
        p = self.parameters[*inputs] if self.parents != 0 else self.parameters
        s = None if self.parents != 0 else size

        samples = rng.binomial(1, p=p, size=s)
        return self._check_samples(samples, size)
