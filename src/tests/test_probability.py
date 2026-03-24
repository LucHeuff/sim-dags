from dataclasses import dataclass
from functools import partial

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xarray as xr
from hypothesis import given
from polars.testing import assert_frame_equal
from scipy import stats
from sim_dags.exceptions import (
    InvalidGridStepsError,
    InvalidPriorDistributionError,
    InvalidPriorShapeError,
    VariableDoesNotExistError,
)
from sim_dags.probability import (
    QueryParts,
    _count,
    _get_name,
    _grid_approx,
    _log_grid_approx,
    _parse_query,
    _permutations,
    log_p_grid,
    log_p_grid_array,
    p,
    p_array,
    p_grid,
    p_grid_array,
)
from sim_dags.utils import to_df


@dataclass
class ParseStrategy:
    """Contains parts for _parse_query() strategy."""

    name: str
    event: list[str]
    given: list[str] | None
    variables: list[str]
    df: pl.DataFrame
    query: str
    error: bool
    event_error: bool
    given_error: bool


@st.composite
def parse_query_strategy(draw: st.DrawFn) -> ParseStrategy:
    """Strategy for testing _parse_query()."""
    event_error, given_error = draw(st.lists(st.booleans(), min_size=2, max_size=2))
    # Making sure given_none is never true when it is supposed to given an error
    given_none = False if given_error else draw(st.booleans())

    # setting up possible variables, string so uppercase is error conditions
    variables = "abcdefg"
    vocab = set(variables)
    error_vocab = set(variables.upper())
    df = pl.DataFrame({v: i for i, v in enumerate(vocab)})
    event_vocab = error_vocab if event_error else vocab

    event = draw(
        st.lists(
            st.sampled_from(list(event_vocab)), min_size=1, max_size=len(vocab) - 1
        )
    )
    given_vocab = error_vocab if given_error else set(vocab) - set(event)
    given_ = (
        None
        if given_none
        else draw(
            st.lists(
                st.sampled_from(list(given_vocab)),
                min_size=1,
                max_size=len(given_vocab),
            )
        )
    )

    query = (
        ",".join(event)
        if given_ is None
        else ",".join(event) + "|" + ",".join(given_)
    )

    variables = event if given_ is None else event + given_

    return ParseStrategy(
        _get_name(query),
        event,
        given_,
        variables,
        df,
        query,
        event_error or given_error,
        event_error,
        given_error,
    )


@given(s=parse_query_strategy())  # ty:ignore[missing-argument]
def test_parse_query(s: ParseStrategy) -> None:
    """Test _parse_query()."""
    if s.error:
        with pytest.raises(VariableDoesNotExistError):
            _parse_query(s.df, s.query)
    else:
        q = _parse_query(s.df, s.query)
        assert q.name == s.name, "name does not match what is expected"
        assert q.event == s.event, "event does not match what is expected"
        assert q.given == s.given, "given does not match what is expected"
        assert q.variables == s.variables, "variables do not match what is expected"


def test_permutations() -> None:
    """Test _permutations()."""
    df = pl.DataFrame(
        {"a": ["a", "a", "b", "b"], "b": [1, 2, 3, 4], "c": [5, 6, 7, 8]}
    )
    # only variables matters for _permutations
    q = QueryParts("", [], None, ["a", "b"])
    target = pl.DataFrame(
        {
            "a": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "b": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )

    assert_frame_equal(_permutations(df, q), target, check_row_order=False)


@dataclass
class ProbabilityStrategy:
    """Contains parts for p() strategy."""

    data: pl.DataFrame
    pw: pl.DataFrame
    px_w: pl.DataFrame
    pz_xw: pl.DataFrame
    py_zxw: pl.DataFrame
    pyx_zw: pl.DataFrame


@pytest.fixture
def static_strategy() -> ProbabilityStrategy:
    """Static strategy for basic testing."""
    data = pl.DataFrame(
        {
            "y": [0, 0, 1, 1, 1, 1, 0, 0],
            "x": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "z": ["x", "x", "x", "x", "y", "y", "y", "y"],
            "w": ["w", "w", "w", "z", "z", "z", "w", "z"],
        }
    )
    pw = pl.DataFrame({"w": ["w", "z"], "P(w)": [0.5, 0.5]})
    px_w = pl.DataFrame(
        {
            "x": ["a", "a", "b", "b"],
            "w": ["w", "z", "w", "z"],
            "P(x|w)": [0.5, 0.5, 0.5, 0.5],
        }
    )
    pz_xw = pl.DataFrame(
        {
            "z": ["x", "x", "x", "y", "y", "y"],
            "x": ["a", "b", "b", "a", "b", "b"],
            "w": ["w", "w", "z", "z", "w", "z"],
            "P(z|x,w)": [1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        }
    )
    py_zxw = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1],
            "z": ["x", "y", "y", "x", "x", "y"],
            "x": ["a", "b", "b", "b", "b", "a"],
            "w": ["w", "w", "z", "w", "z", "z"],
            "P(y|z,x,w)": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    pyx_zw = pl.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1],
            "x": ["a", "b", "b", "a", "b", "b"],
            "z": ["x", "y", "y", "y", "x", "x"],
            "w": ["w", "w", "z", "z", "w", "z"],
            "P(y,x|z,w)": [2 / 3, 1, 1 / 3, 2 / 3, 1 / 3, 1],
        }
    )

    return ProbabilityStrategy(data, pw, px_w, pz_xw, py_zxw, pyx_zw)


def test_p(static_strategy: ProbabilityStrategy) -> None:
    """Test p()."""
    s = static_strategy
    assert_frame_equal(p(s.data, "w"), s.pw)
    assert_frame_equal(p(s.data, "x|w"), s.px_w)
    assert_frame_equal(p(s.data, "z|x,w"), s.pz_xw)
    assert_frame_equal(p(s.data, "y|z,x,w"), s.py_zxw)
    assert_frame_equal(p(s.data, "y,x|z,w"), s.pyx_zw)


def test_p_array(static_strategy: ProbabilityStrategy) -> None:
    """Test p_array() using property based test."""
    s = static_strategy

    def get_p(query: str) -> pl.DataFrame:
        name = _get_name(query)
        return to_df(p_array(s.data, query)).filter(pl.col(name) != 0)

    assert_equal = partial(assert_frame_equal, check_row_order=False)

    assert_equal(get_p("w"), s.pw)
    assert_equal(get_p("x|w"), s.px_w)
    assert_equal(get_p("z|x,w"), s.pz_xw)
    assert_equal(get_p("y|z,x,w"), s.py_zxw)
    assert_equal(get_p("y,x|z,w"), s.pyx_zw)


@dataclass
class GridApproxStrategy:
    """Container for grid approximation strategy."""

    k: int
    n: int
    steps: int
    prior: np.ndarray | None
    log_prior: np.ndarray | None
    grid_error: bool
    prior_error: bool
    beta: np.ndarray
    log_beta: np.ndarray


@st.composite
def get_grid_approx_strategy(draw: st.DrawFn) -> GridApproxStrategy:
    """Strategy for testing grid_approx."""
    grid_error = draw(st.booleans())
    # grid_error makes step size invalid, and automatically makes prior size invalid.
    # So easier to just always have prior_error when grid_error.
    prior_error = True if grid_error else draw(st.booleans())
    prior_none = False if prior_error else draw(st.booleans())

    steps = (
        draw(st.integers(max_value=0))
        if grid_error
        else draw(st.integers(100, 1000))
    )
    sample_steps = max(steps, 1)
    n = draw(st.integers(1, 30))
    k = draw(st.integers(0, max_value=n))
    if prior_error:
        prior = np.repeat(0, sample_steps + 10)
        log_prior = prior
    elif prior_none:
        prior = None
        log_prior = None
    else:
        prior = np.asarray(
            draw(
                st.lists(
                    st.floats(1, 100), min_size=sample_steps, max_size=sample_steps
                )
            )
        )
        log_prior = np.asarray(
            draw(
                st.lists(
                    st.floats(-100, 0), min_size=sample_steps, max_size=sample_steps
                )
            )
        )
    p = np.linspace(0, 1, sample_steps)
    beta = stats.beta.pdf(p, 1 + k, 1 + n - k)
    log_beta = stats.beta.logpdf(p, 1 + k, 1 + n - k)

    return GridApproxStrategy(
        k, n, steps, prior, log_prior, grid_error, prior_error, beta, log_beta
    )


@given(s=get_grid_approx_strategy())  # ty:ignore[missing-argument]
def test_grid_approx(s: GridApproxStrategy) -> None:
    """Test _grid_approx()."""
    if s.grid_error:
        with pytest.raises(InvalidGridStepsError):
            _grid_approx(s.k, s.n, s.steps, s.prior)
    elif s.prior_error:
        with pytest.raises(InvalidPriorShapeError):
            _grid_approx(s.k, s.n, s.steps, s.prior)
    else:
        grid = _grid_approx(s.k, s.n, s.steps, s.prior)
        grid_len = len(grid)
        assert grid_len == s.steps, f"Expected length {s.steps} but got {grid_len}"
        if s.prior is None:
            density = grid["density"].to_numpy()
            L2 = np.sqrt(np.power(density - s.beta, 2))  # noqa: N806
            assert L2.mean() <= 0.011, "Density mismatch with Beta distribution"  # noqa: PLR2004


@given(s=get_grid_approx_strategy())  # ty:ignore[missing-argument]
def test_log_grid_approx(s: GridApproxStrategy) -> None:
    """Test _log_grid_approx()."""
    if s.grid_error:
        with pytest.raises(InvalidGridStepsError):
            _log_grid_approx(s.k, s.n, s.steps, s.log_prior)
    elif s.prior_error:
        with pytest.raises(InvalidPriorShapeError):
            _log_grid_approx(s.k, s.n, s.steps, s.log_prior)
    else:
        grid = _log_grid_approx(s.k, s.n, s.steps, s.log_prior)
        grid_len = len(grid)
        assert grid_len == s.steps, f"Expected length {s.steps} but got {grid_len}"
        if s.log_prior is None:
            density = grid["log_density"].to_numpy()
            L2 = np.sqrt(np.power(density - s.log_beta, 2))  # noqa: N806
            assert np.nanmean(L2) <= 0.1, "Density mismatch with Beta distribution"  # noqa: PLR2004


def test_p_grid(static_strategy: ProbabilityStrategy) -> None:
    """Test p_grid()."""
    s = static_strategy

    def test_grid(query: str, true: pl.DataFrame, grid_steps: int) -> None:
        q = _parse_query(s.data, query)
        grid = p_grid(s.data, query, grid_steps)
        test = (
            grid.with_columns(
                (pl.col("density") == pl.col("density").max())
                .over(q.variables)
                .alias("max")
            )
            .filter(pl.col("max"))
            .select([*q.variables, "p"])
            .rename({"p": q.name})
        )

        assert grid.select(pl.col("p").n_unique()).item() == grid_steps, (
            "Incorrect grid steps."
        )
        assert_frame_equal(
            test,
            true,
            check_exact=False,
            check_row_order=False,
            check_dtypes=False,
            abs_tol=0.1,
        )

    # Testing grid outputs
    test_grid("w", s.pw, 70)
    test_grid("x|w", s.px_w, 100)
    test_grid("z|x,w", s.pz_xw, 135)
    test_grid("y|z,x,w", s.py_zxw, 205)
    test_grid("y,x|z,w", s.pyx_zw, 300)

    # Testing custom prior
    steps = 100
    grid = p_grid(s.data, "w", grid_steps=steps, prior=np.linspace(0, 10, 100))
    assert grid.select(pl.col("p").n_unique()).item() == steps, (
        "Incorrect grid steps."
    )

    # Testing include_zeros
    q = _parse_query(s.data, "y,x|z,w")
    perms = _permutations(_count(s.data, q), q)

    grid_perms = (
        p_grid(s.data, "y,x|z,w", include_zeros=True)
        .select(["y", "x", "z", "w"])
        .unique()
    )
    assert_frame_equal(
        grid_perms,
        perms,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Testing errors
    with pytest.raises(VariableDoesNotExistError):
        p_grid(s.data, "f")
    with pytest.raises(InvalidGridStepsError):
        p_grid(s.data, "w", -1)
    with pytest.raises(InvalidPriorShapeError):
        p_grid(s.data, "w", grid_steps=150, prior=np.repeat(5, 50))
    with pytest.raises(InvalidPriorDistributionError):
        p_grid(s.data, "w", grid_steps=50, prior=np.repeat(-5, 50))


def test_log_p_grid(static_strategy: ProbabilityStrategy) -> None:
    """Test p_grid()."""
    s = static_strategy

    def test_grid(query: str, true: pl.DataFrame, grid_steps: int) -> None:
        q = _parse_query(s.data, query)
        grid = log_p_grid(s.data, query, grid_steps)
        test = (
            grid.with_columns(density=pl.col("log_density").exp())
            .with_columns(
                (pl.col("density") == pl.col("density").max())
                .over(q.variables)
                .alias("max")
            )
            .filter(pl.col("max"))
            .select([*q.variables, "p"])
            .rename({"p": q.name})
        )

        assert grid.select(pl.col("p").n_unique()).item() == grid_steps, (
            "Incorrect grid steps."
        )
        assert_frame_equal(
            test,
            true,
            check_exact=False,
            check_row_order=False,
            check_dtypes=False,
            abs_tol=0.1,
        )

    # Testing grid outputs
    test_grid("w", s.pw, 70)
    test_grid("x|w", s.px_w, 101)
    test_grid("z|x,w", s.pz_xw, 135)
    test_grid("y|z,x,w", s.py_zxw, 205)
    test_grid("y,x|z,w", s.pyx_zw, 300)

    # Testing custom prior
    steps = 100
    grid = log_p_grid(
        s.data, "w", grid_steps=steps, log_prior=np.linspace(-10, 0, 100)
    )
    assert grid.select(pl.col("p").n_unique()).item() == steps, (
        "Incorrect grid steps."
    )

    # Testing include_zeros
    q = _parse_query(s.data, "y,x|z,w")
    perms = _permutations(_count(s.data, q), q)

    grid_perms = (
        log_p_grid(s.data, "y,x|z,w", include_zeros=True)
        .select(["y", "x", "z", "w"])
        .unique()
    )
    assert_frame_equal(
        grid_perms,
        perms,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Testing errors
    with pytest.raises(VariableDoesNotExistError):
        log_p_grid(s.data, "f")
    with pytest.raises(InvalidGridStepsError):
        log_p_grid(s.data, "w", -1)
    with pytest.raises(InvalidPriorShapeError):
        log_p_grid(s.data, "w", grid_steps=150, log_prior=np.repeat(0, 50))
    with pytest.raises(InvalidPriorDistributionError):
        log_p_grid(s.data, "w", grid_steps=50, log_prior=np.repeat(5, 50))


def test_p_grid_array(static_strategy: ProbabilityStrategy) -> None:
    """Test p_grid_array()."""
    s = static_strategy

    def test_grid(query: str, true: pl.DataFrame, grid_steps: int) -> None:
        q = _parse_query(s.data, query)
        grid = to_df(p_grid_array(s.data, query, grid_steps))
        test = (
            grid.with_columns(
                (pl.col(q.name) == pl.col(q.name).max())
                .over(q.variables)
                .alias("max"),
                # When all variables are equal to the max,
                # these are the uniform prior, meaning this combination
                # did not appear in the data.
                (pl.col(q.name) == pl.col(q.name).max())
                .all()
                .over(q.variables)
                .alias("all_max"),
            )
            .filter(pl.col("max") & pl.col("all_max").not_())
            .select([*q.variables, "p"])
            .rename({"p": q.name})
        )

        assert grid.select(pl.col("p").n_unique()).item() == grid_steps, (
            "Incorrect grid steps."
        )
        assert_frame_equal(
            test,
            true,
            check_exact=False,
            check_row_order=False,
            check_dtypes=False,
            abs_tol=0.1,
        )

    # Testing grid outputs
    test_grid("w", s.pw, 70)
    test_grid("x|w", s.px_w, 100)
    test_grid("z|x,w", s.pz_xw, 135)
    test_grid("y|z,x,w", s.py_zxw, 205)
    test_grid("y,x|z,w", s.pyx_zw, 300)

    # Testing custom prior
    steps = 100
    grid = p_grid_array(s.data, "w", grid_steps=steps, prior=np.linspace(0, 10, 100))
    assert to_df(grid).select(pl.col("p").n_unique()).item() == steps, (
        "Incorrect grid steps."
    )

    # Testing errors
    with pytest.raises(VariableDoesNotExistError):
        p_grid_array(s.data, "f")
    with pytest.raises(InvalidGridStepsError):
        p_grid_array(s.data, "w", -1)
    with pytest.raises(InvalidPriorShapeError):
        p_grid_array(s.data, "w", grid_steps=150, prior=np.repeat(5, 50))
    with pytest.raises(InvalidPriorDistributionError):
        p_grid(s.data, "w", grid_steps=50, prior=np.repeat(-5, 50))


def test_log_p_grid_array(static_strategy: ProbabilityStrategy) -> None:
    """Test log_p_grid_array()."""
    s = static_strategy

    def test_grid(query: str, true: pl.DataFrame, grid_steps: int) -> None:
        q = _parse_query(s.data, query)
        grid = to_df(log_p_grid_array(s.data, query, grid_steps))
        test = (
            grid.with_columns(
                (pl.col(f"log {q.name}") == pl.col(f"log {q.name}").max())
                .over(q.variables)
                .alias("max"),
                # When all variables are equal to the max,
                # these are the uniform prior, meaning this combination
                # did not appear in the data.
                (pl.col(f"log {q.name}") == pl.col(f"log {q.name}").max())
                .all()
                .over(q.variables)
                .alias("all_max"),
            )
            .filter(pl.col("max") & pl.col("all_max").not_())
            .select([*q.variables, "p"])
            .rename({"p": q.name})
        )

        assert grid.select(pl.col("p").n_unique()).item() == grid_steps, (
            "Incorrect grid steps."
        )
        assert_frame_equal(
            test,
            true,
            check_exact=False,
            check_row_order=False,
            check_dtypes=False,
            abs_tol=0.1,
        )

    # Testing grid outputs
    test_grid("w", s.pw, 70)
    test_grid("x|w", s.px_w, 101)
    test_grid("z|x,w", s.pz_xw, 135)
    test_grid("y|z,x,w", s.py_zxw, 205)
    test_grid("y,x|z,w", s.pyx_zw, 300)

    # Testing custom prior
    steps = 100
    grid = log_p_grid_array(
        s.data, "w", grid_steps=steps, log_prior=np.linspace(-10, 0, 100)
    )
    assert to_df(grid).select(pl.col("p").n_unique()).item() == steps, (
        "Incorrect grid steps."
    )

    # Testing errors
    with pytest.raises(VariableDoesNotExistError):
        log_p_grid_array(s.data, "f")
    with pytest.raises(InvalidGridStepsError):
        log_p_grid_array(s.data, "w", -1)
    with pytest.raises(InvalidPriorShapeError):
        log_p_grid_array(s.data, "w", grid_steps=150, log_prior=np.repeat(-5, 50))
    with pytest.raises(InvalidPriorDistributionError):
        log_p_grid(s.data, "w", grid_steps=50, log_prior=np.repeat(5, 50))
