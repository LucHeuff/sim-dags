import pytest
from sim_dags.distributions import Binomial, Categorical


def test_categorical() -> None:
    """Test Categorical()."""
    Categorical("x", 3, ["y", "z"])

    with pytest.raises(ValueError):  # noqa: PT011
        Categorical("x", 3, ["y", "y"])


def test_binomial() -> None:
    """Test Binomial()."""
    Binomial("x", ["y", "z"])

    with pytest.raises(ValueError):  # noqa: PT011
        Binomial("x", ["y", "y"])
