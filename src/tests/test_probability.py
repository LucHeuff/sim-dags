from dataclasses import dataclass
from functools import partial

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given
from polars.testing import assert_frame_equal
from scipy import stats
from sim_dags.exceptions import (
    IllegalColumnNameError,
    InvalidGridStepsError,
    InvalidPriorDistributionError,
    InvalidPriorShapeError,
    VariableDoesNotExistError,
)
from sim_dags.probability import (
    ILLEGAL_NAMES,
    QueryParts,
    _count,
    _get_name,
    _parse_query,
    _permutations,
    p,
    p_array,
)
from sim_dags.utils import to_df


@dataclass
class ParseStrategy:
    """Contains parts for _parse_query() strategy."""

    name: str | None
    expected_name: str
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

    # Determining whether to use a custom name
    custom_name = draw(st.booleans())

    expected_name = (
        draw(st.text(st.characters(categories=["Ll"])))
        if custom_name
        else _get_name(query)
    )
    name = expected_name if custom_name else None

    variables = event if given_ is None else event + given_

    return ParseStrategy(
        name,
        expected_name,
        event,
        given_,
        variables,
        df,
        query,
        event_error or given_error,
        event_error,
        given_error,
    )


@given(s=parse_query_strategy())
def test_parse_query(s: ParseStrategy) -> None:
    """Test _parse_query()."""
    if s.error:
        with pytest.raises(VariableDoesNotExistError):
            _parse_query(s.df, s.query, s.name)
    else:
        q = _parse_query(s.df, s.query, s.name)
        assert q.name == s.expected_name, "name does not match what is expected"
        assert q.event == s.event, "event does not match what is expected"
        assert q.given == s.given, "given does not match what is expected"
        assert q.variables == s.variables, "variables do not match what is expected"


def test_parse_query_strategy_illegal() -> None:
    """Test if _parse_query() raises on illegal column names."""
    df = pl.DataFrame({name: [1, 2] for name in ILLEGAL_NAMES})
    with pytest.raises(IllegalColumnNameError):
        _parse_query(df, "_p|_k,_n", None)


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
        draw(st.integers(max_value=0)) if grid_error else draw(st.integers(10, 100))
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
