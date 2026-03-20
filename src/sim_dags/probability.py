from dataclasses import dataclass
from functools import cache
from itertools import product

import polars as pl
import xarray as xr
from attr.filters import include
from numpy.testing import assert_allclose, assert_almost_equal

from sim_dags.exceptions import VariableDoesNotExistError


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
        # Getting all possible permutations of the values in the selected variables
        permutations = list(
            product(*[df[var].unique().to_list() for var in q.variables])
        )
        perm = pl.DataFrame(
            [dict(zip(q.variables, per, strict=True)) for per in permutations]
        )

        df = perm.join(df, on=q.variables, how="left").with_columns(
            pl.col("p").fill_null(0)
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
    """Calculate probability based on a query.

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


# TODO grid approximation toevoegen?
# TODO probability distribution toevoegen?
