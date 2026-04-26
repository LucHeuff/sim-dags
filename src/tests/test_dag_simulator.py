from dataclasses import dataclass

import hypothesis.strategies as st
import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from sim_dags import Binomial, Categorical, DAGSimulator
from sim_dags.dag_simulator import (
    Distribution,
    _find_minimal_adjustment_set,
    _over,
    _under,
)
from sim_dags.exceptions import (
    InvalidDoValueError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
)

# --- Tests for supportive functions


@pytest.fixture
def graph() -> nx.DiGraph:
    """Basic network graph."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("a", "c"), ("b", "c")]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def test_over(graph: nx.DiGraph) -> None:
    """Test _over()."""
    assert set(_over(graph, ["a"]).edges) == set(graph.edges), (
        "over(a) edges incorrect"
    )
    assert set(_over(graph, ["b"]).edges) == {
        ("a", "c"),
        ("b", "c"),
    }, "over(b) edges incorrect"
    assert set(_over(graph, ["c"]).edges) == {("a", "b")}, "over(c) edges incorrect"
    assert set(_over(graph, ["a", "b"]).edges) == {("a", "c"), ("b", "c")}, (
        "over(a, b) edges incorrect"
    )
    assert set(_over(graph, ["a", "b", "c"]).edges) == set(), (
        "over(a, b, c) edges incorrect"
    )


def test_under(graph: nx.DiGraph) -> None:
    """Test _under()."""
    assert set(_under(graph, ["a"]).edges) == {("b", "c")}, (
        "under(a) edges incorrect"
    )
    assert set(_under(graph, ["b"]).edges) == {
        ("a", "b"),
        ("a", "c"),
    }, "under(b) edges incorrect"
    assert set(_under(graph, ["c"]).edges) == set(graph.edges), (
        "under(c) edges incorrect"
    )
    assert set(_under(graph, ["a", "b"]).edges) == set(), (
        "under(a, b) edges incorrect"
    )
    assert set(_under(graph, ["a", "b", "c"]).edges) == set(), (
        "under(a, b, c) edges incorrect"
    )


def test_find_minimal_adjustment_set() -> None:
    """Test _find_minimal_adjustment_set()."""
    assert _find_minimal_adjustment_set(["b"], [["a", "b", "c"]]) == [["b"]], (
        "incorrect adjustment set."
    )
    assert _find_minimal_adjustment_set([], [["a", "b", "c"]]) is None, (
        "incorrect adjustment set."
    )
    assert _find_minimal_adjustment_set(["d"], [["a", "b", "c"]]) is None, (
        "incorrect adjustment set."
    )
    assert _find_minimal_adjustment_set(
        ["b", "c", "d"],
        [["a", "c", "e"], ["a", "b", "c", "d", "e"], ["a", "b", "d", "e"]],
    ) == [["b", "c"], ["c", "d"]], "Incorrect adjustment set"


# ---- Tests for DAGSimulator


@pytest.fixture
def simulator() -> DAGSimulator:
    """Basic DAG simulator."""
    dists = [
        Categorical("x", 4),
        Categorical("z", 3, ["x"]),
        Binomial("y", ["x", "z"]),
    ]

    return DAGSimulator(dists)


def test_basic_dag_simulator(simulator: DAGSimulator) -> None:
    """Simple test of DAGSimulator."""
    size = 100

    samples = simulator.sample(size)
    do_samples = simulator.sample(size, do={"x": True})
    do_x_1_samples = simulator.sample(size, do={"x": 1})
    do_y_samples = simulator.sample(size, do={"y": True})
    do_y_1_samples = simulator.sample(size, do={"y": 1})

    assert len(samples) == size, "samples has the wrong size"
    assert len(do_samples) == size, "do_samples has the wrong size"
    assert len(do_x_1_samples) == size, "do_1_samples has the wrong size"
    assert len(do_y_samples) == size, "do_y_samples has the wrong size"
    assert len(do_y_1_samples) == size, "do_y_1_samples has the wrong size"

    # Testing separate functions that just print to console
    simulator.backdoor_criterion("x", "y")
    simulator.backdoor_criterion("x", "z")
    simulator.backdoor_criterion("z", "y")
    simulator.backdoor_criterion("x", "y", do=["z"])

    simulator.conditional_independencies()
    simulator.conditional_independencies(do=["x"])


def test_dag_simulator_raises_invalid_do_error(simulator: DAGSimulator) -> None:
    """Test if DAGSimulator raises InvalidDoValueError."""
    with pytest.raises(InvalidDoValueError):
        simulator.sample(10, do={"x": 10})


def test_dag_simulator_raises_unknown_do_error(simulator: DAGSimulator) -> None:
    """Test if DAGSimulator raises UnknownDoVariableError."""
    with pytest.raises(UnknownDoVariableError):
        simulator.sample(10, do={"p": True})


@dataclass
class FakeDistribution:
    """Fake distribution."""

    name: str
    categories: int
    parents: list[str]
    unobserved: bool = False


def test_dag_simulator_raises_unknown_distribution() -> None:
    """Test if DAGSimulator raises UnknownDistributionError."""
    dists = [Categorical("x", 4), FakeDistribution("z", 3, ["x"])]
    with pytest.raises(UnknownDistributionError):
        DAGSimulator(dists)


def test_dag_simulator_raises_missing_distribution() -> None:
    """Test if DAGSimulator raises MissingDistributionError."""
    dists = [Categorical("x", 4), Binomial("y", ["x", "z"])]
    with pytest.raises(MissingDistributionError):
        DAGSimulator(dists)


# ---- Randomised integration test


@st.composite
def sample_distribution(
    draw: st.DrawFn, name: str, ancestors: list[str]
) -> Distribution:
    """Sample a random distribution."""
    binom = draw(st.booleans())

    if binom:
        return Binomial(name, ancestors)
    categories = draw(st.integers(3, 50))
    return Categorical(name, categories, ancestors)


def sample_dag(nodes: list[str], seed: int) -> dict[str, list[str]]:
    """Sample a random DAG."""
    rng = np.random.default_rng(seed)
    shape = (len(nodes), len(nodes))

    # generating matrix of random 1s and 0s through Binomial distribution
    matrix = rng.binomial(1, p=0.7, size=shape)
    # Taking the upper triangle without the diagonal as the adjaceny matrix
    # always makes a DAG
    adj_matrix = np.triu(matrix, k=1)

    # extracting ancestors directly from the adjacency matrix ->
    # each column indicates whether the node is an ancestor
    return {
        node: np.compress(adj_matrix[:, i], nodes).astype(str).tolist()  # ty:ignore[no-matching-overload]
        for (i, node) in enumerate(nodes)
    }


@st.composite
def sample_interventions(
    draw: st.DrawFn, distributions: list[Distribution]
) -> dict[str, int | bool]:
    """Sample a randomly generated set of interventions."""
    do = draw(
        st.lists(st.booleans(), min_size=(n := len(distributions)), max_size=n)
    )

    def get_intervention(distribution: Distribution) -> int | bool:
        specific = draw(st.booleans())
        if specific:
            return draw(
                st.integers(min_value=0, max_value=distribution.categories - 1)
            )
        return True

    return {
        dist.name: get_intervention(dist)
        for (dist, do_) in zip(distributions, do, strict=False)
        if do_
    }


@dataclass
class DAGSimulatorStrategy:
    """Container for DAGSimulator testing strategy."""

    distributions: list[Distribution]
    alpha: int
    size: int
    dist_seed: int
    sample_seed: int
    do: dict[str, int | bool]
    rename_do: bool


@st.composite
def dag_simulator_strategy(draw: st.DrawFn) -> DAGSimulatorStrategy:
    """Generate a random DAG and sampling instruction."""
    nodes = draw(
        st.lists(
            st.characters(categories=["Ll"]), min_size=3, max_size=5, unique=True
        )
    )
    dag_seed = draw(st.integers(min_value=0))
    dist_seed = draw(st.integers(min_value=0))
    sample_seed = draw(st.integers(min_value=0))
    alpha = draw(st.integers(min_value=1, max_value=5))
    size = draw(st.integers(min_value=10, max_value=20))
    rename_do = draw(st.booleans())

    dag = sample_dag(nodes, dag_seed)

    distributions = [draw(sample_distribution(node, dag[node])) for node in nodes]
    do = draw(sample_interventions(distributions))

    return DAGSimulatorStrategy(
        distributions, alpha, size, dist_seed, sample_seed, do, rename_do
    )


@given(dag_simulator_strategy())
@settings(deadline=600)
def test_dag_simulator(s: DAGSimulatorStrategy) -> None:
    """Randomised test of DAGSimulator."""
    dag_simulator = DAGSimulator(s.distributions, s.alpha, s.dist_seed)

    samples = dag_simulator.sample(
        s.size, s.sample_seed, do=s.do, rename_do=s.rename_do
    )
    assert len(samples) == s.size, "size does not match set value."
