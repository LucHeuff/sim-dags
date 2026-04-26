from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from itertools import combinations
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
    parents: list[str]
    unobserved: bool = False


@dataclass(frozen=True)
class Categorical:
    """Categorical distribution."""

    name: str
    categories: int = Field(ge=1)
    parents: list[str] = Field(default_factory=list)
    unobserved: bool = False


@dataclass(frozen=True)
class Binomial:
    """Binomial distribution, always has 2 categories (0 and 1)."""

    name: str
    parents: list[str] = Field(default_factory=list)
    categories: int = Field(default=2, init=False)
    unobserved: bool = False


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
    def parents(self) -> int:
        """Get the number of ancestors for this variable."""
        return len(self.distribution.parents)

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
        parents: list[Distribution],
        seed: int,
        alpha: int,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable

        shape = [p.categories for p in parents] if len(parents) > 0 else ()
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
        p = self.parameters[*inputs] if self.parents != 0 else self.parameters
        s = None if self.parents != 0 else size
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
        parents: list[Distribution],
        seed: int,
    ) -> None:
        """Set parameters for this generator."""
        self.distribution = variable
        shape = [p.categories for p in parents] if len(parents) > 0 else ()
        rng = np.random.default_rng(seed)
        self.parameters = rng.uniform(size=shape)

    def sample(self, inputs: np.ndarray, size: int, seed: int) -> np.ndarray:
        """Generate binomial samples without intervention."""
        self._check_inputs(inputs)
        rng = np.random.default_rng(seed)
        p = self.parameters[*inputs] if self.parents != 0 else self.parameters
        s = None if self.parents != 0 else size

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


def _over(graph: nx.DiGraph, variables: list[str]) -> nx.DiGraph:
    """Remove edges pointing into variables from graph."""
    g = graph.copy()
    edges = [edge for edge in g.edges if edge[1] in variables]
    g.remove_edges_from(edges)
    return g


def _under(graph: nx.DiGraph, variables: list[str]) -> nx.DiGraph:
    """Remove edges coming out of variables from graph."""
    g = graph.copy()
    edges = [edge for edge in g.edges if edge[0] in variables]
    g.remove_edges_from(edges)
    return g


def _find_minimal_adjustment_set(
    available: list[str], open_paths: list[list[str]]
) -> list[list[str]] | None:
    """Find minimal adjustment set for these variables."""
    # If there are no available nodes, then there is no adjustment set
    if len(available) == 0:
        return None

    # Finding how often each node appears in the open paths
    frequency = [
        (node, sum(node in path for path in open_paths)) for node in available
    ]
    # if any single node appears as often as the number of open paths,
    # combinations do not need to be searched.

    if (max_ := max(n for _, n in frequency)) == len(open_paths):
        return [[node] for node, n in frequency if n == max_]

    # Nodes that do not appear in any path are irrelevant, ignoring these
    relevant = [node for node, n in frequency if n > 0]

    if len(relevant) == 0:  # returning when none of the nodes are relevant
        return None

    adjustment = []
    min_size = len(relevant) + 1  # adding 1 to avoid early stopping

    for size in range(2, len(relevant) + 1):  # not using min_size cause that changes
        # if we already found smaller sets than this, these aren't minimal.
        if size > min_size:
            break
        # check for combinations of this size whether they close all open paths
        for c in combinations(relevant, size):
            if all(any(node in path for node in c) for path in open_paths):
                adjustment.append(list(c))
                min_size = min(
                    len(c), min_size
                )  # update min size if this is smaller

    return adjustment if len(adjustment) > 0 else None


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
        edges = [(anc, d.name) for d in distributions for anc in d.parents]

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
            parents = [self.distributions[p] for p in self.graph.predecessors(node)]
            match variable:
                case Binomial():
                    return BinomialGenerator(variable, parents, seed)
                case Categorical():
                    return CategoricalGenerator(variable, parents, seed, alpha)
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
                parents = list(self.graph.predecessors(node))
                inputs = np.asarray([results[anc] for anc in parents])
                results[node] = generator.sample(inputs, size, seed)

        # applying rename only if desired.
        rename = rename if rename_do else {}

        return self.schema.validate(pl.DataFrame(results)).rename(rename)

    @cached_property
    def unobserved(self) -> set[str]:
        """Return list of unobserved nodes."""
        return {d.name for d in self.distributions.values() if d.unobserved}

    def backdoor_criterion(
        self, exposure: str, outcome: str, do: list[str] | None = None
    ) -> None:
        """Find and display adjustment sets using the backdoor criterion.

        Args:
            exposure: variable from where the causal path starts
            outcome: variable where the causal path ends
            do (Optional): list of variables that are intervened on.

        Returns:
            Nothing, but prints adjustment sets to the terminal.

        """
        # should make sure the desired causal path exists in the first place
        if not nx.has_path(self.graph, exposure, outcome):
            msg = f"The path {exposure} -> {outcome} does not appear in the DAG."
            return print(msg)  # noqa: T201

        # This message is going to be added to conditionally
        msg = f"Causal effect of {exposure} -> {outcome}.\n"

        do = [] if do is None else do  # making sure do is a list
        # making a copy of the graph, removing edges into do-variables if required
        # and removing edges pointing out of the exposure.
        graph = _over(self.graph, do)
        graph = _under(graph, [exposure])

        # Backdoor are the remaining undirected paths from exposure to outcome
        # converting lists into tuples because those are hashable
        backdoor_paths = list(
            nx.all_simple_paths(graph.to_undirected(), exposure, outcome)
        )
        if len(backdoor_paths) == 0:
            msg += "No backdoor paths found, so no adjustment is necessary."
            return print(msg)  # noqa: T201

        # Finding colliders, as these close backdoor paths by default and should not
        # be adjusted for
        colliders = {
            c for _, c, _ in nx.dag.colliders(graph) if c not in [exposure, outcome]
        }

        # Removing backdoor paths that contain a collider
        if len(colliders) == 0:
            # No need to iterate over lists (slow) if there are no colliders
            open_paths = backdoor_paths
        else:
            open_paths = [
                path
                for path in backdoor_paths
                for collider in colliders
                if collider not in path
            ]
        if len(open_paths) == 0:
            msg += "No open backdoor paths found, so no adjustment is necessary."
            return print(msg)  # noqa: T201

        # Adding open paths to the message, a bit involved due to tuples not joining
        str_paths = [f"[{','.join(list(path))}]" for path in open_paths]
        n = len(open_paths)
        plur = "path" if n == 1 else "paths"  # being pedantic with plurailty
        msg += f"Found {n} open {plur}:\n  {'\n  '.join(str_paths)}\n"

        # Find minimal adjustment set -> minimal list of nodes that close all paths.
        # Any of these nodes must be ancestors of the exposure, and must be observed
        # and must NOT be colliders
        available = set(nx.ancestors(graph, exposure)) - self.unobserved - colliders

        # If there are no available nodes, then there is no adjustment set
        if len(available) == 0:
            msg += "No adjustment sets found."
            return print(msg)  # noqa: T201

        # Finding minimal adjustment sets
        adjustment = _find_minimal_adjustment_set(list(available), open_paths)

        if adjustment is None:
            msg += "No adjustment sets found."
            return print(msg)  # noqa: T201

        str_adj = [f"[{','.join(list(set_))}]" for set_ in adjustment]
        msg += f"Available adjustment sets:\n  {'\n  '.join(str_adj)}"
        return print(msg)  # noqa: T201

    def conditional_independencies(self, do: list[str] | None = None) -> None:
        """Display implied conditional independencies for this DAG.

        Args:
            do (Optional): variables that are being intervened on.

        Returns:
            Nothing, but prints conditional independencies to the console.
        """
        # Applying
        do = [] if do is None else do
        graph = _over(self.graph, do)
        testable = {}
        untestable = {}
        for c in combinations(graph.nodes, 2):
            left, right = c
            indep = nx.find_minimal_d_separator(graph, left, right)
            if indep is not None:
                if left in self.unobserved or right in self.unobserved:
                    untestable[f"{left} ⫫ {right}"] = list(indep)
                else:
                    testable[f"{left} ⫫ {right}"] = list(indep)

        if len(testable) == 0 and len(untestable) == 0:
            msg = "The model does not imply any conditional independencies."
            return print(msg)  # noqa: T201

        msg = "The model implies the following conditional independencies"

        if len(do) > 0:
            str_do = [f"do({var})" for var in do]
            msg += " under " + ",".join(str_do)

        msg += ":"

        def stringify(d: dict[str, list[str]]) -> str:

            return "\n  ".join(
                [
                    f"{ind} | " + ",".join(list(s)) if len(s) > 0 else ind
                    for ind, s in d.items()
                ]
            )

        if len(testable) > 0:
            msg += f"\nTestable:\n  {stringify(testable)}"
        if len(untestable) > 0:
            msg += f"\nUntestable (some variables are unobserved):\n  {stringify(untestable)}"  # noqa: E501

        return print(msg)  # noqa: T201
