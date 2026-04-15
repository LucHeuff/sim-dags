import pytest
from sim_dags.core import Simulator
from sim_dags.exceptions import UnknownDoVariableError
from sim_dags.generators import MockGenerator


def test_simulator() -> None:
    """Test Simulator object."""
    nodes = [
        MockGenerator("X"),
        MockGenerator("Z", ["X"]),
        MockGenerator("Y", ["X", "Z"]),
    ]

    sim = Simulator(nodes)

    size = 100

    samples = sim.sample(size)
    do_samples = sim.sample(size, do={"X": True})

    assert len(samples) == size, "samples has incorrect length"
    assert len(do_samples) == size, "do_samples has incorrect length"
    assert "do(X)" in do_samples.columns, "do(X) does not appear in columns"

    with pytest.raises(UnknownDoVariableError):
        sim.sample(size, do={"x": True})
