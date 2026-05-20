from collections import Counter
from collections.abc import Sequence
from functools import cached_property
from typing import Literal, Protocol

import networkx as nx
import numpy as np
import pandera.polars as pa
import polars as pl
from pydantic.dataclasses import dataclass

from sim_dags.distributions import Binomial, Categorical
from sim_dags.exceptions import (
    DuplicateVariableError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
    VariableNotInDAGError,
)
from sim_dags.generators import BinomialGenerator, CategoricalGenerator, Generator
from sim_dags.graph_algorithms import (
    backdoor_criterion,
    calculate_node_positions,
    conditional_independencies,
    mutilate,
)

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
        # Sanity check, each variable should only appear once in distributions
        unique_vars = {dist.name for dist in distributions}

        if len(distributions) != len(unique_vars):
            count = Counter([dist.name for dist in distributions])
            duplicates = [key for key, value in count.items() if value > 1]
            msg = f"Variables {duplicates} are duplicated, please check your distributions."  # noqa: E501
            raise DuplicateVariableError(msg)

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
        rng = np.random.default_rng(seed)

        def get_generator(node: str) -> Generator:
            """Fetch generator for a specific node."""
            variable = self.distributions[node]
            parents = [self.distributions[p] for p in self.graph.predecessors(node)]
            match variable:
                case Binomial():
                    return BinomialGenerator(variable, parents, rng)
                case Categorical():
                    return CategoricalGenerator(variable, parents, alpha, rng)
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

    def _check_nodes(self, nodes: list[str]) -> None:
        missing = set(nodes) - set(self.graph.nodes)
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
            nodes = set(self.graph.nodes)
            if len(m := do_nodes.difference(nodes)) > 0:
                msg = f"\n\t{m}\ndo not appear in the DAG, available variables are\n\t{nodes} "  # noqa: E501
                raise UnknownDoVariableError(msg)

        else:
            do = {}

        results: dict[str, np.ndarray] = {}
        rename: dict[str, str] = {}

        rng = np.random.default_rng(seed)

        for node in self.topological_sort:
            generator = self.generators[node]
            if node in do:
                results[node] = generator.do(do[node], size, rng)
                rename[node] = generator.do_name
            else:
                parents = list(self.graph.predecessors(node))
                inputs = np.asarray([results[anc] for anc in parents])
                results[node] = generator.sample(inputs, size, rng)

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
        self._check_nodes([exposure, outcome])
        # should make sure the desired causal path exists in the first place
        if not nx.has_path(self.graph, exposure, outcome):
            msg = f"The path {exposure} -> {outcome} does not appear in the DAG."
            return print(msg)  # noqa: T201

        # This message is going to be added to conditionally
        msg = f"Causal effect of {exposure} -> {outcome}.\n"

        if do is not None:
            self._check_nodes(do)
        else:
            do = []  # making sure do is a list

        backdoor = backdoor_criterion(
            self.graph, exposure, outcome, do, self.unobserved
        )
        return print(msg + repr(backdoor))  # noqa: T201

    def conditional_independencies(
        self,
        do: list[str] | None = None,
        ignore: list[str] | None = None,
        show: Literal["testable", "untestable", "both"] = "testable",
    ) -> None:
        """Display implied conditional independencies for this DAG.

        Args:
            do (Optional): variables that are being intervened on.
            ignore (Optional): variables to omit from result.
            show (Optional): which conditional independencies to show.
                 One of 'testable', 'untestable' or 'both'. Defaults to 'testable'

        Returns:
            Nothing, but prints conditional independencies to the console.
        """
        if do is not None:
            # Sanity check for do variables
            self._check_nodes(do)

        if ignore is not None:
            self._check_nodes(ignore)

        cond = conditional_independencies(self.graph, do, ignore, self.unobserved)

        msg = "The graph implies the following conditional independencies"

        if do is not None:
            str_do = [f"do({var})" for var in do]
            msg += " under " + ",".join(str_do)

        msg += ":\n"

        match show:
            case "testable":
                print(msg + cond.render_testable)  # noqa: T201
            case "untestable":
                print(msg + cond.render_untestable)  # noqa: T201
            case "both":
                print(msg + repr(cond))  # noqa: T201

    def mutilate(
        self, over: list[str] | None = None, under: list[str] | None = None
    ) -> nx.DiGraph:
        """Mutilate the graph.

        Args:
            over: removes all arrows pointing into these nodes
            under: removes all arrows pointing out of these nodes

        Returns:
            nx.DiGraph under mutilation

        """
        if over is not None:
            self._check_nodes(over)
        if under is not None:
            self._check_nodes(under)
        return mutilate(self.graph, over, under)

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

        Return:
            boolean indicating if z is a d-separator in the (mutilated) graph.
        """
        graph = self.mutilate(over, under)
        return nx.is_d_separator(graph, x, y, z)

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
        graph = self.mutilate(over, under)

        pos = calculate_node_positions(graph)

        # determining bounding box
        min_y = min(pos.select(pl.col("y").min()).item(), 0) - 0.05
        max_y = max(pos.select(pl.col("y").max()).item(), 1) + 0.05
        bounding_box = f'bb="-0.05,{min_y:.3f},1.05,{max_y:.3f}"'

        def parse_node(row: dict[str, str | float]) -> str:
            node = row["node"]
            latent = "latent," if node in self.unobserved else ""
            return f'{row["node"]} [{latent}pos="{row["x"]:.3f},{row["y"]:.3f}"]'

        nodes = "\n".join(parse_node(row) for row in pos.to_dicts())
        edges = "\n".join(f"{u} -> {v}" for (u, v) in graph.edges)

        print(f"dag {{\n{bounding_box}\n{nodes}\n{edges}\n}}")  # noqa: T201
