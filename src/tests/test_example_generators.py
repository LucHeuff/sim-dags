from sim_dags.example_generators import (
    generate_collider,
    generate_dag1,
    generate_fork,
    generate_pipe,
)


def test_generate_pipe() -> None:
    """Test generate_pipe."""
    size, seed = 100, 12345
    sim = generate_pipe(size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_fork() -> None:
    """Test generate_fork."""
    size, seed = 100, 12345
    sim = generate_fork(size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_collider() -> None:
    """Test generate_collider."""
    size, seed = 100, 12345
    sim = generate_collider(size, seed)

    assert len(sim) == size, "Samples have the wrong size"


def test_generate_dag1() -> None:
    """Test generate_dag1."""
    size, seed = 100, 12345
    sim = generate_dag1(size, seed)

    assert len(sim) == size, "Samples have the wrong size"
