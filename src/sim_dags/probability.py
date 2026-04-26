from dataclasses import dataclass
from itertools import product

import numpy as np
import polars as pl
import xarray as xr
from numpy.testing import assert_allclose, assert_almost_equal
from scipy import stats
from scipy.special import logsumexp

from sim_dags.exceptions import (
    IllegalColumnNameError,
    InvalidGridStepsError,
    InvalidPriorDistributionError,
    InvalidPriorShapeError,
    VariableDoesNotExistError,
)

ILLEGAL_NAMES = {"_k", "_n", "_p"}


@dataclass
class QueryParts:
    """Relevant parts of a query."""

    name: str
    event: list[str]
    given: list[str] | None
    variables: list[str]


def _get_name(query: str) -> str:
    return f"P({query})"


def _parse_query(data: pl.DataFrame, query: str, name: str | None) -> QueryParts:
    """Parse query to relevant parts and perform checks with data."""
    if len(illegal := (set(data.columns) & ILLEGAL_NAMES)) > 0:
        msg = f"Found column names {illegal} in data. These column names are not allowed, as they are used internally.\nThey are also not very pretty."  # noqa: E501
        raise IllegalColumnNameError(msg)

    name = _get_name(query) if name is None else name

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
            data.group_by(q.event)
            .agg(_k=pl.len())
            .with_columns(_n=pl.lit(len(data)))
        )
    return (
        data.group_by(q.variables)
        .agg(_k=pl.len())
        .with_columns(_n=pl.col("_k").sum().over(q.given))
        # .unique()
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
    df = _count(data, q).with_columns(_p=pl.col("_k") / pl.col("_n"))

    # --- Sanity checks
    # Making sure there are no duplicates in the dataframe after counting
    assert len(df.filter(df.is_duplicated())) == 0, (
        "Counts contain duplicates. This usually happens due to column name collisions."  # noqa: E501
    )

    if q.given is None:
        # sum of all probabilities should be (almost) 1
        _sum = df.select(pl.col("_p").sum()).item()
        assert_almost_equal(_sum, 1, err_msg="Probabilities do not add to 1")

    else:
        # Probabilities in each group should add to (almost) 1
        _sum = (
            df.group_by(q.given)
            .agg(pl.col("_p").sum())
            .select(pl.col("_p"))
            .to_numpy()
        )
        assert_allclose(_sum, 1, err_msg="Probabilities do not add to 1")

    if include_zeros:
        df = (
            _permutations(df, q)
            .join(df, on=q.variables, how="left")
            .with_columns(pl.col("_p").fill_null(0))
        )

    return df.select([*q.variables, "_p"]).rename({"_p": q.name}).sort(q.variables)


def p(
    data: pl.DataFrame,
    query: str,
    name: str | None = None,
    *,
    include_zeros: bool = False,
) -> pl.DataFrame:
    """Calculate probability based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"
        name: desired name of probability column. Defaults to P(<query>).
        include_zeros (Optional): whether combination that do not appear in
                      data should also be included

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.

    """
    q = _parse_query(data, query, name)

    return _p(data, q, include_zeros=include_zeros)


def p_array(data: pl.DataFrame, query: str, name: str | None = None) -> xr.DataArray:
    """Calculate probability array based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z"
        name: desired name of probability column. Defaults to P(<query>).

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.

    """
    q = _parse_query(data, query, name)
    # Conversion using pandas,
    # since that makes sure the values end up in the right place
    p_ = _p(data, q, include_zeros=True)
    # Also my first successful application of a MultiIndex
    return p_.to_pandas().set_index(q.variables).to_xarray()[q.name]
