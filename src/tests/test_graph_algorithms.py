import polars as pl
import pytest
from sim_dags.graph_algorithms import (
    calculate_node_positions,
    dagitty_code,
)
from sim_dags.graphs import Edges, Node, NodeSequence


@pytest.fixture
def nodes() -> list[Node]:
    """Nodes of the DAG above."""
    return list("abcdefg")


@pytest.fixture
def edges() -> Edges:
    """Edges of the DAG above."""
    return {
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "e"),
        ("d", "f"),
        ("d", "g"),
        ("e", "g"),
    }


@pytest.fixture
def topological_generations() -> list[NodeSequence]:
    """Topological sort of the DAG above."""
    return [["a"], ["b", "c"], ["d", "e"], ["f", "g"]]


def test_calculate_node_positions(
    nodes: list[Node], topological_generations: list[NodeSequence]
) -> None:
    """Test calculate_node_positions()."""
    pos = calculate_node_positions(topological_generations).sort("node")

    assert isinstance(pos, pl.DataFrame), "wrong datatype"
    assert pos.columns == ["node", "x", "y"], "wrong columns"
    assert pos["node"].to_list() == nodes, "wrong nodes"


def test_dagitty_code(
    edges: Edges, topological_generations: list[NodeSequence]
) -> None:
    """Test dagitty_code."""
    assert isinstance(dagitty_code(edges, topological_generations, set()), str)
