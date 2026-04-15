import pytest
from sim_dags.example_generators import (
    DAG1Params,
    SimpleDAGParams,
    generate_collider,
    generate_dag1,
    generate_fork,
    generate_pipe,
)


@pytest.fixture
def params() -> SimpleDAGParams:
    """Parameters for simple DAGs."""
    return SimpleDAGParams(54321)


def test_generate_pipe(params: SimpleDAGParams) -> None:
    """Test generate_pipe."""
    size, seed = 100, 12345
    sim = generate_pipe(params, size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_fork(params: SimpleDAGParams) -> None:
    """Test generate_fork."""
    size, seed = 100, 12345
    sim = generate_fork(params, size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_collider(params: SimpleDAGParams) -> None:
    """Test generate_collider."""
    size, seed = 100, 12345
    sim = generate_collider(params, size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_dag1() -> None:
    """Test generate_dag1."""
    params = DAG1Params(54321)
    size, seed = 100, 12345
    sim = generate_dag1(params, size, seed)

    assert len(sim) == size, "Samples have the wrong size"
