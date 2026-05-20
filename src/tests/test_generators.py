import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal
from sim_dags.generators import do_fixed, do_uniform


def test_basic_do_fixed() -> None:
    """Basic test of _do_fixed()."""
    value, size = 2, 13
    do = do_fixed(value, size)
    assert len(do) == size, "do has the wrong size"
    assert len(np.unique(do)) == 1, "do doesn't have 1 unique value"
    assert np.unique(do)[0] == value, "do has incorrect value"


@given(value=st.integers(0, 15), size=st.integers(100, 1000))
def test_do_fixed(value: int, size: int) -> None:
    """Randomized test of _do_fixed()."""
    do = do_fixed(value, size)
    assert len(do) == size, "do has the wrong size"
    assert len(np.unique(do)) == 1, "do doesn't have 1 unique value"
    assert np.unique(do)[0] == value, "do has incorrect value"


def test_basic_do_uniform() -> None:
    """Basic test of _do_uniform()."""
    rng = np.random.default_rng(12345)
    categories, size = 7, 23  # both primes
    do = do_uniform(categories, size, rng)
    assert len(do) == size, "do has the wrong size"
    assert len(np.unique(do)) == categories, "do has incorrect categories"

    categories, size = 5, 100
    do = do_uniform(categories, size, rng)
    values, counts = np.unique(do, return_counts=True)
    frequencies = counts / size
    assert_array_equal(
        values, np.arange(categories), err_msg="do has incorrect values"
    )
    assert np.all(frequencies == 1 / categories), "do has incorrect frequencies"


@given(categories=st.integers(1, 29), size=st.integers(100, 1000))
def test_do_uniform(categories: int, size: int) -> None:
    """Randomized test of _do_uniform()."""
    rng = np.random.default_rng(12345)
    do = do_uniform(categories, size, rng)
    values, counts = np.unique(do, return_counts=True)
    frequencies = counts / size
    assert len(do) == size, "do has the wrong size"
    assert len(values) == categories, "do has incorrect categories"
    assert_array_equal(
        values, np.arange(categories), err_msg="do has incorrect values"
    )
    assert_allclose(
        frequencies,
        np.repeat(1 / categories, categories),
        err_msg="do as incorrect frequencies",
        atol=0.01,
    )
