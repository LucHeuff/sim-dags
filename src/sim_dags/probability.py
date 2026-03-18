from itertools import product

import polars as pl
import xarray as xr
from numpy.testing import assert_allclose, assert_almost_equal

from sim_dags.exceptions import VariableDoesNotExistError


def _get_name(query: str) -> str:  # pragma: no cover
    return f"P({query})"


def _parse_query(
    data: pl.DataFrame, query: str
) -> tuple[list[str], list[str] | None]:
    """Parse query to relevant parts and perform checks with data."""
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

    return event, given


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
    name = _get_name(query)
    event, given = _parse_query(data, query)
    variables = event + given if given is not None else event

    if given is None:
        df = data.group_by(event).agg(p=pl.len() / len(data))
        # Sanity check: sum of all probabilities should be (almost) 1
        sum_ = df.select(pl.col("p").sum()).item()
        assert_almost_equal(sum_, 1, err_msg="Probabilities do not add to 1")

    else:
        df = (
            data.group_by(variables)
            .agg(n=pl.len())
            .with_columns(p=pl.col("n") / pl.col("n").sum().over(given))
        )

        sum_ = (
            df.group_by(given).agg(pl.col("p").sum()).select(pl.col("p")).to_numpy()
        )

        assert_allclose(sum_, 1, err_msg="Probabilities do not add to 1")

    if include_zeros:
        # Getting all possible permutations of the values in the selected variables
        permutations = list(
            product(*[df[var].unique().to_list() for var in variables])
        )
        perm = pl.DataFrame(
            [dict(zip(variables, per, strict=True)) for per in permutations]
        )

        df = perm.join(df, on=variables, how="left").with_columns(
            pl.col("p").fill_null(0)
        )

    return df.select([*variables, "p"]).rename({"p": name}).sort(variables)


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
    name = _get_name(query)
    # Conversion using pandas,
    # since that makes sure the values end up in the right place
    p_ = p(data, query, include_zeros=True)
    variables = [col for col in p_.columns if col != name]
    # Also my first successful application of a MultiIndex
    return p_.to_pandas().set_index(variables).to_xarray()[name]


# TODO grid approximation toevoegen?
# TODO probability distribution toevoegen?
