import os
from collections import Counter
from collections.abc import Sequence
from functools import cached_property
from typing import Protocol

import numpy as np
import pandera.polars as pa
import polars as pl
from pydantic.dataclasses import dataclass

from sim_dags.distributions import Binomial, Categorical
from sim_dags.exceptions import (
    DuplicateVariableError,
    UnknownDistributionError,
    UnknownDoVariableError,
    VariableNotInDAGError,
)
from sim_dags.generators import BinomialGenerator, CategoricalGenerator, Generator
from sim_dags.graph_algorithms import dagitty_code
from sim_dags.graphs import CIOptions, DirectedAcyclicGraph

# finding the number of cores to use for calculating conditional independencies
cores = os.cpu_count()
MAX_CORES = 1 if cores is None else cores // 2
# ---- Supporting functions


class Distribution(Protocol):
    """Interface for distributions."""

    name: str
    categories: int
    parents: list[str]
    unobserved: bool
    param_seed: int | None


@dataclass(slots=True, frozen=True)
class ConditionalIndependecies:
    """Container for partial outputs of conditional independencies."""

    testable: dict[str, list[list[str]]]
    untestable: dict[str, list[list[str]]]


class DAGSimulator:
    """Simulate samples from a DAG.

    Intended for validating estimands derived from DAGs.
    """

    dag: DirectedAcyclicGraph
    distributions: dict[str, Distribution]
    generators: dict[str, Generator]
    schema: pa.DataFrameSchema

    def __init__(
        self,
        distributions: Sequence[Distribution],
        alpha: float = 1,
        seed: int = 12345,
    ) -> None:
        """Parse the generators into a DAG.

        Args:
            distributions: list of distributions the DAG is built from
            alpha (Optional): parameter for Dirichlet distributions
            seed (Optional): for generating parameters.

        """
        # Sanity check, each variable should only appear once in distributions
        unique_vars = {dist.name for dist in distributions}

        if len(distributions) != len(unique_vars):
            count = Counter([dist.name for dist in distributions])
            duplicates = [key for key, value in count.items() if value > 1]
            msg = f"Variables {duplicates} are duplicated, please check your distributions."  # noqa: E501
            raise DuplicateVariableError(msg)

        self.distributions = {d.name: d for d in distributions}

        # setting up the DAG from the provided distributions
        nodes = self.distributions.keys()
        edges = [(anc, d.name) for d in distributions for anc in d.parents]

        self.dag = DirectedAcyclicGraph(nodes, edges)
        # setting up the generators -> requires knowing distributions of ancestors
        rng = np.random.default_rng(seed)

        def get_generator(node: str) -> Generator:
            """Fetch generator for a specific node."""
            variable = self.distributions[node]
            parents = [self.distributions[p] for p in self.dag.parents[node]]
            match variable:
                case Binomial():
                    return BinomialGenerator(variable, parents, rng)
                case Categorical():
                    return CategoricalGenerator(variable, parents, alpha, rng)
                case _:
                    msg = f"No known generator for {variable.__class__.__name__}"
                    raise UnknownDistributionError(msg)

        self.generators = {node: get_generator(node) for node in self.dag.nodes}

        assert len(self.distributions) == len(self.generators), (
            "Unequal number of generators and distributions"
        )

        # setting up class attributes that only need to be calculated once.
        self.schema = pa.DataFrameSchema(
            columns={
                d.name: pa.Column(int, pa.Check.isin(list(range(d.categories))))
                for d in distributions
            },
            strict=True,
        )

    def _check_nodes(self, nodes: list[str]) -> None:
        missing = set(nodes) - set(self.dag.nodes)
        if len(missing) > 0:
            msg = f"{missing} do not appear in the DAG."
            raise VariableNotInDAGError(msg)

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
            nodes = set(self.dag.nodes)
            if len(m := do_nodes.difference(nodes)) > 0:
                msg = f"\n\t{m}\ndo not appear in the DAG, available variables are\n\t{nodes} "  # noqa: E501
                raise UnknownDoVariableError(msg)

        else:
            do = {}

        results: dict[str, np.ndarray] = {}
        rename: dict[str, str] = {}

        rng = np.random.default_rng(seed)

        for node in self.dag.topological_sort:
            generator = self.generators[node]
            if node in do:
                results[node] = generator.do(do[node], size, rng)
                rename[node] = generator.do_name
            else:
                parents = list(self.dag.parents[node])
                inputs = np.asarray([results[anc] for anc in parents])
                results[node] = generator.sample(inputs, size, rng)

        # applying rename only if desired.
        rename = rename if rename_do else {}

        return self.schema.validate(pl.DataFrame(results)).rename(rename)

    @cached_property
    def unobserved(self) -> set[str]:
        """Return list of unobserved nodes."""
        return {d.name for d in self.distributions.values() if d.unobserved}

    @cached_property
    def variables(self) -> list[str]:
        """Return a list of variables in the DAG."""
        return sorted(self.distributions.keys())

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
        self._check_nodes([exposure, outcome])
        if do is not None:
            self._check_nodes(do)

        self.dag.backdoor_criterion(exposure, outcome, do, self.unobserved)

    def conditional_independencies(
        self,
        over: list[str] | None = None,
        under: list[str] | None = None,
        ignore: list[str] | None = None,
        show: CIOptions = "testable",
        max_cores: int = MAX_CORES,
    ) -> None:
        """Display implied conditional independencies for this DAG.

        Args:
            over (Optional): remove edges pointing into these variables.
            under (Optional): remove edges coming out of these variables.
            ignore (Optional): variables to omit from result.
            show (Optional): which conditional independencies to show.
                 One of 'testable', 'untestable' or 'both'. Defaults to 'testable'
            max_cores (Optional): number of cores to calculate independencies with.
                                  Defaults to half the logical cores in the system.

        Returns:
            Nothing, but prints conditional independencies to the console.
        """
        # Sanity check for input variables
        if over is not None:
            self._check_nodes(over)
        if under is not None:
            self._check_nodes(under)
        if ignore is not None:
            self._check_nodes(ignore)
            ignore_ = set(ignore)
        else:
            ignore_ = set()

        self.dag.conditional_independencies(
            over, under, ignore_, self.unobserved, max_cores, show
        )

    def is_d_separator(
        self,
        x: str | set[str],
        y: str | set[str],
        z: str | set[str],
        over: list[str] | None = None,
        under: list[str] | None = None,
    ) -> bool:
        """Test whether z d-separates x from y.

        Args:
            x: node or set of nodes
            y: node or set of nodes
            z: node or set of nodes, tested as separating set
            over (Optional): all arrows pointing into these nodes are removed
            under (Optional): all arrows pointing out of these nodes are removed

        Returns:
            boolean indicating if z is a d-separator in the (mutilated) graph.
        """
        x = {x} if not isinstance(x, set) else x
        y = {y} if not isinstance(y, set) else y
        z = {z} if not isinstance(z, set) else z
        return self.dag.is_d_separator(x, y, z, over, under)

    def find_d_separators(
        self,
        x: str | set[str],
        y: str | set[str],
        over: list[str] | None = None,
        under: list[str] | None = None,
        *,
        include_unobserved: bool = False,
    ) -> None:
        """Find sets that d-separate x and y.

        Args:
            x: node or set of nodes
            y: node or set of nodes
            over (Optional): all arrows pointing into these nodes are removed
            under (Optional): all arrows pointing out of these nodes are removed
            include_unobserved (Optional): include unobserved variables in search

        Returns:
            nothing, but prints d-separating sets if any exist.

        """
        x = {x} if not isinstance(x, set) else x
        y = {y} if not isinstance(y, set) else y

        unobserved = set() if include_unobserved else self.unobserved

        self.dag.find_d_separators(x, y, unobserved, over, under)

    def dagitty_code(
        self, over: list[str] | None = None, under: list[str] | None = None
    ) -> None:
        """Convert DAG to dagitty code, optionally under mutilation.

        Args:
            over (Optional): all arrows pointing into these nodes are removed
            under (Optional): all arrows pointing out of these nodes are removed

        Returns:
            prints dagitty code string to console.

        """
        edges = self.dag.mutilate(over, under)

        print(dagitty_code(edges, self.dag.topological_generations, self.unobserved))  # noqa: T201
