from sim_dags.example_generators import (
    get_collider_simulator,
    get_dag1_simulator,
    get_fork_simulator,
    get_pipe_simulator,
)


def test_get_pipe_simulator() -> None:
    """Test get_pipe_simulator()."""
    size, seed = 100, 12346
    sim = get_pipe_simulator()

    samples = sim.sample(size, seed)
    do_samples = sim.sample(size, seed, do={"x": True})

    assert len(samples) == size, "samples have incorrect size"
    assert len(do_samples) == size, "do_samples have incorrect size"


def test_get_fork_simulator() -> None:
    """Test get_fork_simulator()."""
    size, seed = 100, 12346
    sim = get_fork_simulator()

    samples = sim.sample(size, seed)
    do_samples = sim.sample(size, seed, do={"x": True})

    assert len(samples) == size, "samples have incorrect size"
    assert len(do_samples) == size, "do_samples have incorrect size"


def test_get_collider_simulator() -> None:
    """Test get_collider_simulator()."""
    size, seed = 100, 12346
    sim = get_collider_simulator()

    samples = sim.sample(size, seed)
    do_samples = sim.sample(size, seed, do={"x": True})

    assert len(samples) == size, "samples have incorrect size"
    assert len(do_samples) == size, "do_samples have incorrect size"


def test_get_dag1_simulator() -> None:
    """Test get_dag1_simulator()."""
    size, seed = 100, 12346
    sim = get_dag1_simulator()

    samples = sim.sample(size, seed)
    do_samples = sim.sample(size, seed, do={"x": True})

    assert len(samples) == size, "samples have incorrect size"
    assert len(do_samples) == size, "do_samples have incorrect size"
