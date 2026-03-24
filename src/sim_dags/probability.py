from dataclasses import dataclass
from itertools import product

import numpy as np
import polars as pl
import xarray as xr
from numpy.testing import assert_allclose, assert_almost_equal
from scipy import stats

from sim_dags.exceptions import (
    InvalidGridStepsError,
    InvalidPriorError,
    VariableDoesNotExistError,
)


@dataclass
class QueryParts:
    """Relevant parts of a query."""

    name: str
    event: list[str]
    given: list[str] | None
    variables: list[str]


def _get_name(query: str) -> str:
    return f"P({query})"


def _parse_query(data: pl.DataFrame, query: str) -> QueryParts:
    """Parse query to relevant parts and perform checks with data."""
    name = _get_name(query)

    if "|" not in query:
        e, g = query, None
    else:
        e, g = query.split("|")

    event = [e_.strip() for e_ in e.split(",")]
    given = [g_.strip() for g_ in g.split(",")] if g is not None else None

    variables = event + given if given is not None else event

    if len(miss := [var for var in variables if var not in data.columns]) > 0:
        msg = f"Variables {miss} do not appear in data."
        raise VariableDoesNotExistError(msg)

    return QueryParts(name, event, given, variables)


def _count(data: pl.DataFrame, q: QueryParts) -> pl.DataFrame:
    """Count number of occurrences of event and given if applicable."""
    if q.given is None:
        return (
            data.group_by(q.event).agg(k=pl.len()).with_columns(n=pl.lit(len(data)))
        )
    return (
        data.group_by(q.variables)
        .agg(k=pl.len())
        .with_columns(n=pl.col("k").sum().over(q.given))
    )


def _permutations(df: pl.DataFrame, q: QueryParts) -> pl.DataFrame:
    """Generates all possible permutations from the combination of variables.

    Assumes df to be the output of _count()
    """
    permutations = list(
        product(*[df[var].unique().to_list() for var in q.variables])
    )
    return pl.DataFrame(
        [dict(zip(q.variables, per, strict=True)) for per in permutations]
    )


def _p(data: pl.DataFrame, q: QueryParts, *, include_zeros: bool) -> pl.DataFrame:
    """Calculate probability from a query."""
    df = _count(data, q).with_columns(p=pl.col("k") / pl.col("n"))

    # --- Sanity checks
    if q.given is None:
        # sum of all probabilities should be (almost) 1
        _sum = df.select(pl.col("p").sum()).item()
        assert_almost_equal(_sum, 1, err_msg="Probabilities do not add to 1")

    else:
        # Probabilities in each group should add to (almost) 1
        _sum = (
            df.group_by(q.given)
            .agg(pl.col("p").sum())
            .select(pl.col("p"))
            .to_numpy()
        )
        assert_allclose(_sum, 1, err_msg="Probabilities do not add to 1")

    if include_zeros:
        df = (
            _permutations(df, q)
            .join(df, on=q.variables, how="left")
            .with_columns(pl.col("p").fill_null(0))
        )

    return df.select([*q.variables, "p"]).rename({"p": q.name}).sort(q.variables)


def p(
    data: pl.DataFrame, query: str, *, include_zeros: bool = False
) -> pl.DataFrame:
    """Calculate probability based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"
        include_zeros (Optional): whether combination that do not appear in
                      data should also be included

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.

    """
    q = _parse_query(data, query)

    return _p(data, q, include_zeros=include_zeros)


def p_array(data: pl.DataFrame, query: str) -> xr.DataArray:
    """Calculate probability array based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.

    """
    q = _parse_query(data, query)
    # Conversion using pandas,
    # since that makes sure the values end up in the right place
    p_ = _p(data, q, include_zeros=True)
    # Also my first successful application of a MultiIndex
    return p_.to_pandas().set_index(q.variables).to_xarray()[q.name]


def _grid_approx(
    k: int, n: int, grid_steps: int, prior: np.ndarray | None
) -> pl.DataFrame:
    """Calculate grid approximation with Binomial(n, k)."""
    assert k <= n, f"n should be smaller than k, got {k = } > {n = }"
    if grid_steps <= 0:
        msg = f"Grid steps cannot be less than 0, got {grid_steps}"
        raise InvalidGridStepsError(msg)

    p = np.linspace(0, 1, grid_steps)
    prior = stats.uniform.pdf(p) if prior is None else prior

    if (s := prior.shape) != (grid_steps,):
        msg = f"Prior ({s}) should have the same length as grid ({grid_steps})"
        raise InvalidPriorError(msg)

    bayes = stats.binom.pmf(k, n, p) * prior
    density = bayes / np.trapezoid(bayes, p)

    assert_almost_equal(
        np.trapezoid(density, p), 1, err_msg="Density does not integrate to 1"
    )

    return pl.DataFrame({"p": p, "density": density})


def _p_grid(
    data: pl.DataFrame,
    q: QueryParts,
    grid_steps: int,
    prior: np.ndarray | None,
    *,
    include_zeros: bool,
) -> pl.DataFrame:
    """Calculate probability grid from a query."""
    counts = _count(data, q)
    if include_zeros:
        # Adding all possible permutations and setting k=0 and n=0 when no counts
        counts = (
            _permutations(counts, q)
            .join(counts, on=q.variables, how="left")
            .with_columns(pl.col("k").fill_null(0), pl.col("n").fill_null(0))
        )

    def approx_count(d: dict) -> pl.DataFrame:
        k = d.pop("k")
        n = d.pop("n")
        return _grid_approx(k, n, grid_steps, prior).with_columns(
            **{key: pl.lit(d[key]) for key in d}
        )

    return pl.concat([approx_count(d) for d in counts.to_dicts()])


def p_grid(
    data: pl.DataFrame,
    query: str,
    grid_steps: int = 100,
    prior: np.ndarray | None = None,
    *,
    include_zeros: bool = False,
) -> pl.DataFrame:
    """Calculate probability density based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"
        grid_steps: number of steps in grid approximation
        prior (Optional): prior distribution for grid approximation.
                        Should be of the same lenght as grid_steps.
                        Defaults to a uniform distribution.
        include_zeros (Optional): whether combination that do not appear in
                      data should also be included

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.
        InvalidGridStepsError if the number of grid steps is 0 or less
        InvalidPriorError if the prior has a different length from grid_steps

    """
    q = _parse_query(data, query)
    return _p_grid(data, q, grid_steps, prior, include_zeros=include_zeros)


def p_grid_array(
    data: pl.DataFrame,
    query: str,
    grid_steps: int = 100,
    prior: np.ndarray | None = None,
) -> xr.DataArray:
    """Calculate probability density based on a query, and return as DataArray.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"
        grid_steps: number of steps in grid approximation
        prior (Optional): prior distribution for grid approximation.
                        Should be of the same lenght as grid_steps.
                        Defaults to a uniform distribution.
        include_zeros (Optional): whether combination that do not appear in
                      data should also be included

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.
        InvalidGridStepsError if the number of grid steps is 0 or less
        InvalidPriorError if the prior has a different length from grid_steps

    """
    q = _parse_query(data, query)
    grid = _p_grid(data, q, grid_steps, prior, include_zeros=True)

    return (
        grid.to_pandas()
        .set_index([*q.variables, "p"])
        .to_xarray()["density"]
        .rename(q.name)
    )


# TODO Log distributions?
