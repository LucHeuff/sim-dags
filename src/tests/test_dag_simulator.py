from dataclasses import dataclass

import hypothesis.strategies as st
import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from sim_dags import Binomial, Categorical, DAGSimulator
from sim_dags.dag_simulator import Distribution
from sim_dags.exceptions import (
    InvalidDoValueError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
)


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
    ancestors: list[str]


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

    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph, nodelist=nodes)
    assert nx.is_directed_acyclic_graph(graph), "graph is not a DAG"

    return {node: list(nx.ancestors(graph, node)) for node in nodes}


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
    size = draw(st.integers(min_value=10, max_value=100))
    rename_do = draw(st.booleans())

    dag = sample_dag(nodes, dag_seed)

    distributions = [draw(sample_distribution(node, dag[node])) for node in nodes]
    do = draw(sample_interventions(distributions))

    return DAGSimulatorStrategy(
        distributions, alpha, size, dist_seed, sample_seed, do, rename_do
    )


@given(dag_simulator_strategy())
@settings(deadline=350)
def test_dag_simulator(s: DAGSimulatorStrategy) -> None:
    """Randomised test of DAGSimulator."""
    dag_simulator = DAGSimulator(s.distributions, s.alpha, s.dist_seed)

    samples = dag_simulator.sample(
        s.size, s.sample_seed, do=s.do, rename_do=s.rename_do
    )
    assert len(samples) == s.size, "size does not match set value."
