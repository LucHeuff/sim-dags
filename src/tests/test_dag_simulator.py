from dataclasses import dataclass

import hypothesis.strategies as st
import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_equal, assert_raises
from sim_dags import Binomial, Categorical, DAGSimulator
from sim_dags.dag_simulator import (
    Distribution,
    _find_minimal_adjustment_set,
    _find_minimal_d_separators,
    _over,
    _under,
)
from sim_dags.example_generators import (
    get_collider_simulator,
    get_fork_simulator,
    get_pipe_simulator,
)
from sim_dags.exceptions import (
    InvalidDoValueError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
    VariableNotInDAGError,
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


def test_find_minimal_d_separators() -> None:
    """Test _find_minimal_d_separators()."""
    # Simple model with no
    simple_graph = nx.DiGraph()
    simple_graph.add_edges_from([("x", "z"), ("x", "y"), ("z", "y")])
    assert _find_minimal_d_separators(simple_graph, "x", "y") is None, (
        "Simple graph shouldn't have d-separators"
    )
    assert _find_minimal_d_separators(simple_graph, "x", "z") is None, (
        "Simple graph shouldn't have d-separators"
    )
    assert _find_minimal_d_separators(simple_graph, "y", "z") is None, (
        "Simple graph shouldn't have d-separators"
    )

    # Somewhat more complicated model
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("v", "r"),
            ("w", "r"),
            ("w", "y"),
            ("x", "v"),
            ("x", "y"),
            ("z", "x"),
            ("z", "y"),
        ]
    )
    assert _find_minimal_d_separators(graph, "x", "r") == [["v"]], (
        "wrong d-separators for x ⫫ r"
    )
    assert _find_minimal_d_separators(graph, "x", "w") == [[]], (
        "wrong d-separators for x ⫫ w"
    )
    assert _find_minimal_d_separators(graph, "y", "v") == [["x"]], (
        "wrong d-separators for y ⫫ v"
    )
    assert _find_minimal_d_separators(graph, "y", "r") == [["w", "x"], ["v", "w"]], (
        "wrong d-separators for y ⫫ r"
    )
    assert _find_minimal_d_separators(graph, "v", "w") == [[]], (
        "wrong d-separators for v ⫫ w"
    )


# ---- Tests for DAGSimulator


@pytest.fixture
def simulator() -> DAGSimulator:
    """Basic DAG simulator."""
    dists = [
        Binomial("u1"),
        Binomial("u2", unobserved=True),
        Categorical("x", 4, ["u1"]),
        Categorical("z", 3, ["x", "u2"]),
        Binomial("y", ["x", "z"]),
        Binomial("w", ["y", "z"]),
    ]

    return DAGSimulator(dists)


@pytest.fixture
def m_model_simulator() -> DAGSimulator:
    """M model simulator."""
    return DAGSimulator(
        [
            Binomial("w"),
            Binomial("v"),
            Binomial("z", ["w", "v"]),
            Binomial("x", ["w"]),
            Binomial("y", ["v", "x"]),
        ]
    )


@pytest.fixture
def complex_model_simulator() -> DAGSimulator:
    """M model simulator with unobserved fork."""
    return DAGSimulator(
        [
            Binomial("v"),
            Binomial("w"),
            Categorical("r", 4, ["v", "w"]),
            Categorical("z", 3, unobserved=True),
            Categorical("x", 4, ["v", "z"]),
            Binomial("y", ["w", "x", "z"]),
        ]
    )


def test_dag_simulator_sample(simulator: DAGSimulator) -> None:
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


def test_dag_path_has_collider() -> None:
    """Test DAGSimulator._path_has_collider()."""
    collider = get_collider_simulator()

    assert not collider._path_has_collider(["x", "y"]), (  # noqa: SLF001
        "x -> y should be too short for collider"
    )
    assert collider._path_has_collider(["x", "z", "y"]), (  # noqa: SLF001
        "x -> z <- y should contain collider"
    )


def test_dag_backdoor(
    m_model_simulator: DAGSimulator, complex_model_simulator: DAGSimulator
) -> None:
    """Test DAGSimulator._backdoor()."""
    pipe = get_pipe_simulator()
    pipe_back = pipe._backdoor("x", "y", [])  # noqa: SLF001
    assert pipe_back.backdoor_paths == [], "Pipe should have no backdoor paths."
    assert pipe_back.open_paths == [], "Pipe should have no open paths"
    assert pipe_back.adjustment_sets == [], "Pipe doesn't need adjustment"

    fork = get_fork_simulator()
    fork_back = fork._backdoor("x", "y", [])  # noqa: SLF001
    assert fork_back.backdoor_paths == [["x", "z", "y"]], (
        "Fork should have x -> z -> y backdoor path"
    )
    assert fork_back.open_paths == [["x", "z", "y"]], (
        "Fork should have x -> z -> y open path"
    )
    assert fork_back.adjustment_sets == [["z"]], "Fork adjustment set should be {z}"

    m_back = m_model_simulator._backdoor("x", "y", [])  # noqa: SLF001
    assert m_back.backdoor_paths == [["x", "w", "z", "v", "y"]], (
        "M model should have one backdoor path"
    )
    assert m_back.open_paths == [], "M backdoor paths should be closed by collider z"
    assert m_back.adjustment_sets == [], "M should need no adjustment"

    # Model with an open backdoor path and available variables, but no adjustment set
    unobs_back = complex_model_simulator._backdoor("x", "y", [])  # noqa: SLF001
    assert unobs_back.backdoor_paths == [["x", "z", "y"]], (
        "unobs model should have x -> z -> y backdoor path"
    )
    assert unobs_back.open_paths == [["x", "z", "y"]], (
        "unobs model should have x -> z -> y open path"
    )
    assert unobs_back.adjustment_sets == [], (
        "unobs model should have no adjustment set"
    )

    # Fork with unobserved to test if there are no variables available
    unobs_fork = DAGSimulator(
        [
            Categorical("z", 3, unobserved=True),
            Categorical("x", 4, ["z"]),
            Binomial("y", ["x", "z"]),
        ]
    )
    unobs_fork_back = unobs_fork._backdoor("x", "y", [])  # noqa: SLF001
    assert unobs_fork_back.backdoor_paths == [["x", "z", "y"]], (
        "Unobserved Fork should have x -> z -> y backdoor path"
    )
    assert unobs_fork_back.open_paths == [["x", "z", "y"]], (
        "Unobserved Fork should have x -> z -> y open path"
    )
    assert unobs_fork_back.adjustment_sets == [], (
        "Unobserved Fork adjustment set should be empty"
    )


def test_backdoor_criterion(
    m_model_simulator: DAGSimulator, complex_model_simulator: DAGSimulator
) -> None:
    """Test if DAGSimulator.backdoor_criterion() work as intended."""
    # M model -> collider that blocks only backdoor path
    m_model_simulator.backdoor_criterion("x", "y")
    m_model_simulator.backdoor_criterion("y", "x")  # test non-existing path
    # complex model has unobserved confound and no adjustment set
    complex_model_simulator.backdoor_criterion("x", "y")
    # Testing when there are no backdoor paths
    pipe = get_pipe_simulator()
    pipe.backdoor_criterion("x", "y")
    # Testing when there are backdoor paths
    fork = get_fork_simulator()
    fork.backdoor_criterion("x", "y")


def test_conditional(complex_model_simulator: DAGSimulator) -> None:
    """Test DAGSimulator._conditional()."""
    cond = complex_model_simulator._conditional([], [])  # noqa: SLF001
    assert len(cond.testable) == 4, (  # noqa: PLR2004
        "Complex model should have 4 testable independencies"
    )
    assert cond.testable["v ⫫ w"] == [[]], "v ⫫ w independencies incorrect"
    assert cond.testable["w ⫫ x"] == [[]], "w ⫫ x independencies incorrect"
    assert cond.testable["r ⫫ x"] == [["v"]], "r ⫫ x independencies incorrect"
    assert cond.testable["r ⫫ y"] == [["v", "w"]], "r ⫫ y independencies incorrect"

    assert len(cond.untestable) == 4, (  # noqa: PLR2004
        "Complex model should have 4 untestable independencies."
    )
    assert cond.untestable["v ⫫ (z)"] == [[]], "v ⫫ (z) independencies incorrect"
    assert cond.untestable["v ⫫ y"] == [["x", "(z)"]], (
        "v ⫫ y independencies incorrect"
    )
    assert cond.untestable["w ⫫ (z)"] == [[]], "w ⫫ (z) independencies incorrect"
    assert cond.untestable["r ⫫ (z)"] == [[]], "r ⫫ (z) independencies incorrect"

    ign_cond = complex_model_simulator._conditional([], ["z"])  # noqa: SLF001
    assert len(ign_cond.untestable) == 0, (
        "Ignoring z should leave no untestable independencies"
    )

    do_cond = complex_model_simulator._conditional(["z"], [])  # noqa: SLF001
    assert do_cond.untestable == cond.untestable, (
        "Intervening on Z shouldn't change anything"
    )
    assert do_cond.testable == cond.testable, (
        "Intervening on Z shouldn't change anything"
    )


def test_conditional_indepencencies(complex_model_simulator: DAGSimulator) -> None:
    """Test if all the options in conditional_independencies work as intended."""
    pipe_model = DAGSimulator(
        [Binomial("x"), Binomial("z", ["x"]), Binomial("y", ["z"])]
    )
    pipe_model.conditional_independencies()
    pipe_unobs_model = DAGSimulator(
        [Binomial("x"), Binomial("z", ["x"], unobserved=True), Binomial("y", ["z"])]
    )
    pipe_unobs_model.conditional_independencies()
    pipe_unobs_model.conditional_independencies(do=["x"])
    complete_model = DAGSimulator(
        [Binomial("x"), Binomial("z", ["x"]), Binomial("y", ["z", "x"])]
    )
    complete_model.conditional_independencies()

    complex_model_simulator.conditional_independencies()
    complex_model_simulator.conditional_independencies(ignore=["z"])


def test_dag_simulator_raises_invalid_do_error(simulator: DAGSimulator) -> None:
    """Test if DAGSimulator raises InvalidDoValueError."""
    with pytest.raises(InvalidDoValueError):
        simulator.sample(10, do={"x": 10})


def test_dag_simulator_raises_unknown_do_error(simulator: DAGSimulator) -> None:
    """Test if DAGSimulator raises UnknownDoVariableError."""
    with pytest.raises(UnknownDoVariableError):
        simulator.sample(10, do={"p": True})


def test_backdoor_criterion_raises_missing_variable(simulator: DAGSimulator) -> None:
    """Test if DAGSimulator.backdoor_criterion() raises VariableNotInDAGError."""
    with pytest.raises(VariableNotInDAGError):
        simulator.backdoor_criterion("t", "u")
    with pytest.raises(VariableNotInDAGError):
        simulator.backdoor_criterion("x", "y", do=["t"])


def test_conditional_indepencencies_raises_missing_variable(
    simulator: DAGSimulator,
) -> None:
    """Test if DAGSimulator.conditional_independencies() raises VariableNotInDAGError."""  # noqa: E501
    # with pytest.raises(VariableNotInDAGError):
    #     simulator.conditional_independencies(do=["t", "u"])


def test_fix_seeds() -> None:
    """Test if fixing seeds works as intended."""
    dag1 = DAGSimulator(
        [Binomial("x"), Categorical("y", 4, param_seed=10)], seed=12345
    )
    dag2 = DAGSimulator(
        [Binomial("x"), Categorical("y", 4, param_seed=10)], seed=54321
    )

    x1 = dag1.generators["x"].parameters
    x2 = dag2.generators["x"].parameters
    y1 = dag1.generators["y"].parameters
    y2 = dag2.generators["y"].parameters

    assert_equal(y1, y2, err_msg="Parameters are different, but should be the same")
    with assert_raises(AssertionError):
        assert_equal(x1, x2, err_msg="Parameters should not be equal")


@dataclass
class FakeDistribution:
    """Fake distribution."""

    name: str
    categories: int
    parents: list[str]
    unobserved: bool = False
    param_seed: int | None = None


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
