from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import networkx as nx
import numpy as np
import pandera.polars as pa
import polars as pl
from pydantic import Field
from pydantic.dataclasses import dataclass

from sim_dags.exceptions import (
    InvalidDoValueError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
)


def _get_do_name(name: str) -> str:  # pragma: no cover
    """Translate name to do(name)."""
    return f"do({name})"


class Distribution(Protocol):
    """Interface for distributions."""

    name: str
    categories: int
    ancestors: list[str]


@dataclass(frozen=True)
class Categorical:
    """Categorical distribution."""

    name: str
    categories: int = Field(ge=1)
    ancestors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Binomial:
    """Binomial distribution, always has 2 categories (0 and 1)."""

    name: str
    ancestors: list[str] = Field(default_factory=list)
    categories: int = Field(default=2, init=False)


class Generator(ABC):
    """Interface for generators."""

    distribution: Distribution
    parameters: np.ndarray
    do_parameters: np.ndarray

    @property
    def do_name(self) -> str:
        """Get name of intervened variable."""
        return _get_do_name(self.distribution.name)

    @property
    def ancestors(self) -> int:
        """Get the number of ancestors for this variable."""
        return len(self.distribution.ancestors)

    @property
    def name(self) -> str:
        """Get the name of this variable."""
        return self.distribution.name

    def _check_inputs(self, inputs: np.ndarray) -> None:
        """Check if inputs have the expected length."""
        shape = inputs.shape[0]
        assert shape == self.ancestors, (
            f"Got {shape} inputs when '{self.name}' has {self.ancestors} ancestors."
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
    def sample(self, inputs: np.ndarray, size: int, seed: int) -> np.ndarray:
        """Generate samples without intervention."""
        ...

    @abstractmethod
    def do(
        self,
        value: bool | int,  # noqa: FBT001
        size: int,
        seed: int,
    ) -> np.ndarray:
        """Generate smaples under intervention."""
        ...


class CategoricalGenerator(Generator):
    """Generates categorical samples for a single variable."""

    distribution: Categorical

    def __init__(
        self,
        variable: Categorical,
        ancestors: list[Distribution],
        seed: int,
        alpha: int,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable

        shape = [anc.categories for anc in ancestors] if len(ancestors) > 0 else ()
        rng = np.random.default_rng(seed)

        # dirichlet distribution with number of categories of current variable
        # as last dimension
        categories = self.distribution.categories
        self.parameters = rng.dirichlet(np.repeat(alpha, categories), size=shape)
        self.do_parameters = np.repeat(1 / categories, categories)

    def sample(self, inputs: np.ndarray, size: int, seed: int) -> np.ndarray:
        """Generate categorical samples without intervention."""
        self._check_inputs(inputs)
        rng = np.random.default_rng(seed)
        p = self.parameters[*inputs] if self.ancestors != 0 else self.parameters
        s = None if self.ancestors != 0 else size
        samples = rng.multinomial(1, pvals=p, size=s).argmax(axis=1)
        return self._check_samples(samples, size)

    def do(
        self,
        value: bool | int,  # noqa: FBT001
        size: int,
        seed: int,
    ) -> np.ndarray:
        """Generate categorical samples under intervention."""
        if not isinstance(value, bool):
            self._check_values(value)
            samples = np.repeat(value, size)
        else:
            rng = np.random.default_rng(seed)
            samples = rng.choice(
                self.distribution.categories, p=self.do_parameters, size=size
            )
        return self._check_samples(samples, size)


class BinomialGenerator(Generator):
    """Generates binomial samples for a single variable."""

    distribution: Binomial

    def __init__(
        self,
        variable: Binomial,
        ancestors: list[Distribution],
        seed: int,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable
        shape = [anc.categories for anc in ancestors] if len(ancestors) > 0 else ()

        rng = np.random.default_rng(seed)
        self.parameters = rng.uniform(size=shape)

    def sample(self, inputs: np.ndarray, size: int, seed: int) -> np.ndarray:
        """Generate binomial samples without intervention."""
        self._check_inputs(inputs)
        rng = np.random.default_rng(seed)
        p = self.parameters[*inputs] if self.ancestors != 0 else self.parameters
        s = None if self.ancestors != 0 else size
        samples = rng.binomial(1, p=p, size=s)
        return self._check_samples(samples, size)

    def do(self, value: int, size: int, seed: int) -> np.ndarray:
        """Generate binomial samples under intervention."""
        if not isinstance(value, bool):
            self._check_values(value)
            samples = np.repeat(value, size)
        else:
            rng = np.random.default_rng(seed)
            samples = rng.binomial(1, p=0.5, size=size)
        return self._check_samples(samples, size)


class DAGSimulator:
    """Simulate samples from a DAG.

    Intended for validating estimands derived from DAGs.
    """

    graph: nx.DiGraph
    topological_sort: list[str]
    distributions: dict[str, Distribution]
    generators: dict[str, Generator]
    schema: pa.DataFrameSchema

    def __init__(
        self,
        distributions: Sequence[Distribution],
        alpha: int = 2,
        seed: int = 12345,
    ) -> None:
        """Parse the generators into a DAG.

        Args:
            distributions: list of distributions the DAG is built from
            alpha (Optional): parameter for Dirichlet distributions
            seed (Optional): for generating parameters.

        """
        self.graph = nx.DiGraph()

        self.distributions = {d.name: d for d in distributions}

        # setting up the DAG from the provided distributions
        nodes = self.distributions.keys()
        edges = [(anc, d.name) for d in distributions for anc in d.ancestors]

        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

        # Sanity checks
        assert nx.is_directed_acyclic_graph(self.graph), (
            "Provided distributions do not form a DAG."
        )

        # checking if nodes were added through edges that do not have a distribution
        if len(miss := (set(self.graph.nodes).difference(nodes))) > 0:
            msg = f"{miss} are mentioned as ancestors but do not have an associated distribution."  # noqa: E501
            raise MissingDistributionError(msg)

        # setting up the generators -> requires knowing distributions of ancestors

        def get_generator(node: str) -> Generator:
            """Fetch generator for a specific node."""
            variable = self.distributions[node]
            ancestors = [
                self.distributions[anc] for anc in nx.ancestors(self.graph, node)
            ]
            match variable:
                case Binomial():
                    return BinomialGenerator(variable, ancestors, seed)
                case Categorical():
                    return CategoricalGenerator(variable, ancestors, seed, alpha)
                case _:
                    msg = f"No known generator for {variable.__class__.__name__}"
                    raise UnknownDistributionError(msg)

        self.generators = {node: get_generator(node) for node in self.graph.nodes}

        assert len(self.distributions) == len(self.generators), (
            "Unequal number of generators and distributions"
        )

        # setting up class attributes that only need to be calculated once.
        self.topological_sort = list(nx.topological_sort(self.graph))
        self.schema = pa.DataFrameSchema(
            columns={
                d.name: pa.Column(int, pa.Check.isin(list(range(d.categories))))
                for d in distributions
            },
            strict=True,
        )

    def sample(
        self,
        size: int,
        seed: int = 0,
        *,
        do: dict[str, int | bool] | None = None,
        rename_do: bool = True,
    ) -> pl.DataFrame:
        """Sample from the DAG.

        Args:
            size: number of samples (rows in the output DataFrame)
            seed (Optional): seed for random number generator.
            alpha (Optional): alpha parameter for Dirichlet distributions
            do (Optional): dictionary of intervention variables.
                        set {"x" : 1} for do(x) = 1
                        or {"x": True} to give all values of x an equal probability.
            rename_do (Optional): whether to rename intervened variables
                                (e.g x -> do(x)). Defaults to True.

        Returns:
            polars.DataFrame containing samples.

        """
        # validating and processing inputs
        if do is not None:
            do_nodes = set(do)
            nodes = set(self.graph.nodes)
            if len(m := do_nodes.difference(nodes)) > 0:
                msg = f"\n\t{m}\ndo not appear in the DAG, available variables are\n\t{nodes} "  # noqa: E501
                raise UnknownDoVariableError(msg)

        else:
            do = {}

        results: dict[str, np.ndarray] = {}
        rename: dict[str, str] = {}

        for node in self.topological_sort:
            generator = self.generators[node]
            if node in do:
                results[node] = generator.do(do[node], size, seed)
                rename[node] = generator.do_name
            else:
                ancestors = nx.ancestors(self.graph, node)
                inputs = np.asarray([results[anc] for anc in ancestors])
                results[node] = generator.sample(inputs, size, seed)

        # applying rename only if desired.
        rename = rename if rename_do else {}

        return self.schema.validate(pl.DataFrame(results)).rename(rename)
