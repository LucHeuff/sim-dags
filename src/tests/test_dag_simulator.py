from dataclasses import dataclass

import pytest
from numpy.testing import assert_equal, assert_raises
from sim_dags import Binomial, Categorical, DAGSimulator
from sim_dags.example_generators import (
    get_fork_simulator,
    get_pipe_simulator,
)
from sim_dags.exceptions import (
    DuplicateVariableError,
    InvalidDoValueError,
    MissingDistributionError,
    UnknownDistributionError,
    UnknownDoVariableError,
    VariableNotInDAGError,
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

    complex_model_simulator.conditional_independencies(show="untestable")
    complex_model_simulator.conditional_independencies(show="both")


def test_mutilate(simulator: DAGSimulator) -> None:
    """Test mutilate."""
    no_change = simulator.mutilate()

    assert simulator.graph.nodes == no_change.nodes, "graph shouldn't change"
    assert simulator.graph.edges == no_change.edges, "graph shouldn't change"

    over_x = simulator.mutilate(over=["x"])
    assert ("u1", "x") not in over_x.edges, "incorrect edge"

    under_y = simulator.mutilate(under=["y"])
    assert ("y", "w") not in under_y.edges, "incorrect edge"


def test_is_d_separator(simulator: DAGSimulator) -> None:
    """Test is_d_separator()."""
    assert simulator.is_d_separator("y", "u1", "x")
    assert not simulator.is_d_separator("x", "w", "y")


def test_dagitty_code(simulator: DAGSimulator) -> None:
    """Test dagitty_code()."""
    simulator.dagitty_code()


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
    with pytest.raises(VariableNotInDAGError):
        simulator.conditional_independencies(do=["t", "u"])


def test_fix_seeds() -> None:
    """Test if fixing seeds works as intended."""
    dag1 = DAGSimulator(
        [
            Binomial("x"),
            Categorical("y", 4, param_seed=10),
            Binomial("z", param_seed=5),
        ],
        seed=12345,
    )
    dag2 = DAGSimulator(
        [
            Binomial("x"),
            Categorical("y", 4, param_seed=10),
            Binomial("z", param_seed=5),
        ],
        seed=54321,
    )

    x1 = dag1.generators["x"].parameters
    x2 = dag2.generators["x"].parameters
    y1 = dag1.generators["y"].parameters
    y2 = dag2.generators["y"].parameters
    z1 = dag1.generators["z"].parameters
    z2 = dag2.generators["z"].parameters

    assert_equal(y1, y2, err_msg="Parameters are different, but should be the same")
    assert_equal(z1, z2, err_msg="Parameters are different, but should be the same")
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


def test_dag_simulator_raises_duplicate_variable() -> None:
    """Test if DAGSimulator raises DuplicateVariableError."""
    dists = [Binomial("x"), Categorical("x", 3)]
    with pytest.raises(DuplicateVariableError):
        DAGSimulator(dists)
