from abc import ABC, abstractmethod
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandera.polars as pa
import polars as pl

from sim_dags.exceptions import UnknownDoVariableError


def _get_do_name(name: str) -> str:
    """Translate name to do(name)."""
    return f"do({name})"


class Generator(ABC):
    """Interface for generator objects."""

    name: str
    do_name: str
    ancestors: list[str]
    column_schema: pa.Column

    @abstractmethod
    def __call__(  # noqa: D102
        self, inputs: list[np.ndarray] | None, size: int, seed: int
    ) -> np.ndarray: ...

    @abstractmethod
    def do(  # noqa: D102
        self,
        value: bool | float | np.ndarray,  # noqa: FBT001
        size: int,
        seed: int,
    ) -> np.ndarray: ...

    @property
    def do_name(self) -> str:
        """Return intervened name of variable."""
        return _get_do_name(self.name)


class Simulator:
    """Simulate samples from a DAG."""

    graph: nx.DiGraph
    topological_sort: list[str]
    schema: pa.DataFrameSchema

    def __init__(self, generators: Sequence[Generator]) -> None:
        """Parse the generators into a DAG."""
        self.graph = nx.DiGraph()

        nodes = [(g.name, {"generator": g}) for g in generators]
        edges = [(anc, g.name) for g in generators for anc in g.ancestors]

        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

        assert self.graph.is_directed(), "Provided generators do not make a DAG."

        self.topological_sort = list(nx.topological_sort(self.graph))
        self.schema = pa.DataFrameSchema(
            columns={g.name: g.column_schema for g in generators}, strict=True
        )

    def sample(
        self,
        size: int,
        seed: int = 0,
        *,
        do: dict[str, bool | float | np.ndarray] | None = None,
    ) -> pl.DataFrame:
        """Sample from the DAG.

        Args:
            size: number of samples (rows in the output DataFrame)
            do (Optional): dictionary of intervention variables and their values.
            seed (Optional): seed for random number generator.

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
            generator = self.graph.nodes[node]["generator"]
            if node in do:
                results[node] = generator.do(do[node], size, seed)
                rename[node] = generator.do_name
            else:
                ancestors = nx.ancestors(self.graph, node)
                inputs = (
                    [results[anc] for anc in ancestors]
                    if len(ancestors) > 0
                    else None
                )
                results[node] = generator(inputs, size, seed)

        return self.schema.validate(pl.DataFrame(results)).rename(rename)
