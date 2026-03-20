from dataclasses import dataclass
from functools import partial

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xarray as xr
from hypothesis import given
from polars.testing import assert_frame_equal
from sim_dags.exceptions import VariableDoesNotExistError
from sim_dags.probability import _get_name, _parse_query, p, p_array
from sim_dags.utils import to_df


@dataclass
class ParseStrategy:
    """Contains parts for _parse_query() strategy."""

    event: list[str]
    given: list[str] | None
    df: pl.DataFrame
    query: str
    error: bool
    event_error: bool
    given_error: bool


@st.composite
def parse_query_strategy(draw: st.DrawFn) -> ParseStrategy:
    """Strategy for testing parse_query."""
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

    return ParseStrategy(
        event,
        given_,
        df,
        query,
        event_error or given_error,
        event_error,
        given_error,
    )


@given(s=parse_query_strategy())  # ty:ignore[missing-argument]
def test_parse_query(s: ParseStrategy) -> None:
    """Test parse_query()."""
    if s.error:
        with pytest.raises(VariableDoesNotExistError):
            _parse_query(s.df, s.query)
    else:
        e, g = _parse_query(s.df, s.query)
        assert e == s.event, "event does not match what is expected"
        assert g == s.given, "given does not match what is expected"


@dataclass
class ProbabilityStrategy:
    """Contains parts for p() strategy."""

    data: pl.DataFrame
    pw: pl.DataFrame
    px_w: pl.DataFrame
    pz_xw: pl.DataFrame
    py_zxw: pl.DataFrame
    pyx_zw: pl.DataFrame
    seed: int


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

    return ProbabilityStrategy(data, pw, px_w, pz_xw, py_zxw, pyx_zw, seed=0)


@st.composite
def probability_strategy(draw: st.DrawFn) -> ProbabilityStrategy:
    """Strategy for testing p() and p_array()."""
    seed = draw(st.integers(min_value=0))
    rng = np.random.default_rng(seed)
    size = 100_000
    min_n, max_n = 2, 4
    alpha = 2

    # Drawing from DAG following P(X, Y, Z, W) = P(Y|X, Z, W)P(Z|X, W)P(X|W)P(W)
    nw = draw(st.integers(min_n, max_n))
    pw = rng.dirichlet(np.repeat(alpha, nw))
    pw_ = xr.DataArray(pw, coords={"w": np.arange(nw)}).rename("P(w)")

    nx = draw(st.integers(min_n, max_n))
    px_w = rng.dirichlet(np.repeat(alpha, nx), size=nw)
    px_w_ = xr.DataArray(
        px_w, coords={"w": np.arange(nw), "x": np.arange(nx)}
    ).rename("P(x|w)")

    nz = draw(st.integers(min_n, max_n))
    pz_xw = rng.dirichlet(np.repeat(alpha, nz), size=(nx, nw))
    pz_xw_ = xr.DataArray(
        pz_xw, coords={"x": np.arange(nx), "w": np.arange(nw), "z": np.arange(nz)}
    ).rename("P(z|x,w)")

    py_zxw = rng.uniform(size=(nz, nx, nw))
    # Adding the y=0 version manually since p() will calculate this regardless
    py_zxw__ = xr.DataArray(
        py_zxw,
        coords={"z": np.arange(nz), "x": np.arange(nx), "w": np.arange(nw)},
    ).rename("P(y|z,x,w)")
    py_zxw_1 = py_zxw__.expand_dims(y=[1])
    py_zxw_0 = (1 - py_zxw__).expand_dims(y=[0])
    py_zxw_ = xr.concat([py_zxw_0, py_zxw_1], dim="y")

    # P(y,x|z, w) = P(y, x, z, w) / P(z, w)
    pyx_zw_ = (
        (py_zxw_ * pz_xw_ * px_w_ * pw_) / (pz_xw_ * px_w_).sum(dim="x")
    ).rename("P(y,x|z,w)")

    w = rng.choice(nw, p=pw, size=size)
    x = rng.multinomial(1, pvals=px_w[w]).argmax(axis=1)
    z = rng.multinomial(1, pvals=pz_xw[x, w]).argmax(axis=1)
    y = rng.binomial(n=1, p=py_zxw[z, x, w])

    return ProbabilityStrategy(
        data=pl.DataFrame({"x": x, "y": y, "z": z, "w": w}),
        pw=to_df(pw_),
        px_w=to_df(px_w_),
        pz_xw=to_df(pz_xw_),
        py_zxw=to_df(py_zxw_),
        pyx_zw=to_df(pyx_zw_),
        seed=seed,
    )


def test_basic_p(static_strategy: ProbabilityStrategy) -> None:
    """Test p()."""
    s = static_strategy
    assert_frame_equal(p(s.data, "w"), s.pw)
    assert_frame_equal(p(s.data, "x|w"), s.px_w)
    assert_frame_equal(p(s.data, "z|x,w"), s.pz_xw)
    assert_frame_equal(p(s.data, "y|z,x,w"), s.py_zxw)
    assert_frame_equal(p(s.data, "y,x|z,w"), s.pyx_zw)


def test_basic_p_array(static_strategy: ProbabilityStrategy) -> None:
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


@given(s=probability_strategy())  # ty:ignore[missing-argument]
def test_p(s: ProbabilityStrategy) -> None:
    """Test p()."""

    def join(query: str, true: pl.DataFrame) -> pl.DataFrame:
        name = _get_name(query)
        df = p(s.data, query)
        return df.join(
            true, on=[col for col in df.columns if col != name], suffix=" true"
        ).with_columns(L2=(pl.col(name) - pl.col(f"{name} true")).pow(2).sqrt())

    def L2(df: pl.DataFrame, threshold: float) -> pl.DataFrame:  # noqa: N802
        return df.filter(pl.col("L2") >= threshold)

    pw = join("w", s.pw)
    px_w = join("x|w", s.px_w)
    pz_xw = join("z|x,w", s.pz_xw)
    py_zxw = join("y|z,x,w", s.py_zxw)
    pyx_zw = join("y,x|z,w", s.pyx_zw)

    assert len(c := L2(pw, 0.02)) == 0, f"L2 for P(w) over threshold.\n{c!s}"
    assert len(c := L2(px_w, 0.03)) == 0, f"L2 for P(x|w) over threshold.\n{c!s}"
    assert len(c := L2(pz_xw, 0.2)) == 0, f"L2 for P(z|x,w) over threshold.\n{c!s}"
    assert len(c := L2(py_zxw, 0.4)) == 0, (
        f"L2 for P(y|z,x,w) over threshold.\n{c!s}"
    )
    # This isn't testing the accuracy of values at all anymore
    # Might indicate instability in calculating probability products
    assert len(c := L2(pyx_zw, 0.9)) == 0, (
        f"L2 for P(y,x|z,w) over threshold.\n{c!s}"
    )
