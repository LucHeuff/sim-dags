from dataclasses import dataclass
from itertools import product

import numpy as np
import polars as pl
import xarray as xr
from numpy.testing import assert_allclose, assert_almost_equal
from scipy import stats

from sim_dags.exceptions import (
    IllegalColumnNameError,
    VariableDoesNotExistError,
    VariableNotBinomialError,
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
        name (Optional): desired name of probability column. Defaults to P(<query>).
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
        name (Optional): desired name of probability column. Defaults to P(<query>).

    Returns:
        polars DataFrame containing probabilities

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.

    """
    q = _parse_query(data, query, name)
    p_ = _p(data, q, include_zeros=True)
    # Conversion using pandas,
    # since that makes sure the values end up in the right place
    # Also my first successful application of a MultiIndex
    return p_.to_pandas().set_index(q.variables).to_xarray()[q.name]


# ---- calculating probability distributions using beta or grid approximation


def _p_dist(
    data: pl.DataFrame, q: QueryParts, steps: int, prior: np.ndarray | None
) -> pl.DataFrame:
    """Calculate probability distribution from a query."""
    # This distribution only makes sense if event is a single variable
    assert len(q.event) == 1, (
        f"Probability distribution doesn't make sense for joint distributions; got event={q.event}"  # noqa: E501
    )
    e = q.event[0]

    df = _count(data, q)
    # Sanity check for variable names
    assert len(df.filter(df.is_duplicated())) == 0, (
        "Counts contain duplicates. This usually happens due to column name collisions."  # noqa: E501
    )
    # Making sure there are no duplicates in the dataframe after counting
    if not (df.schema[e] == pl.Int64 and df.select(pl.col(e).max()).item() == 1):
        msg = f"Variable '{e}' doesn't seem to be Binomial"
        raise VariableNotBinomialError(msg)

    # Removing the e = 0 condition, this is simply the opposite of e = 1
    df = df.filter(pl.col(e) == 1)

    # Finding all permutations
    df = (
        _permutations(df, q)
        .join(df, on=q.variables, how="left")
        .with_columns(pl.col(["_k", "_n"]).fill_null(0))
    )

    def get_dist(k: int, n: int) -> dict[str, np.ndarray]:
        grid_length = len(prior) if prior is not None else steps
        p = np.linspace(0, 1, grid_length)

        if prior is None:
            return {"p": p, "density": stats.beta.pdf(p, k + 1, n - k + 1)}
        bayes = stats.binom.pmf(k, n, p) * prior
        return {"p": p, "density": bayes / np.trapezoid(bayes, p)}

    dists = []
    for d in df.to_dicts():
        k = d.pop("_k")
        n = d.pop("_n")
        _df = pl.DataFrame(get_dist(k, n)).join(pl.DataFrame(d), how="cross")
        dists.append(_df)

    data = pl.concat(dists)

    # Sanity checking that density should add up to (almost) one for all groups
    if q.given is None:
        _sum = np.trapezoid(data["density"].to_numpy(), data["p"].to_numpy())
        assert_allclose(
            _sum, 1, rtol=1e-3, err_msg="Probability density does not interate to 1."
        )
    else:
        _sum = np.asarray(
            [
                np.trapezoid(d["density"].to_numpy(), d["p"].to_numpy())
                for _, d in data.group_by(q.given)
            ]
        )
        assert_allclose(
            _sum,
            1,
            rtol=1e-3,
            err_msg="Probability densities do not integrate to 1.",
        )

    return data.rename({"density": q.name}).sort(q.variables)


def p_distribution(
    data: pl.DataFrame,
    query: str,
    name: str | None = None,
    steps: int = 100,
    prior: np.ndarray | None = None,
) -> pl.DataFrame:
    """Calculate probability distribution based on a query.

    Args:
        data: dataset from which probability is to be calculated
        query: desired probability, eg. "Y|X, Z".
               Note that joint distributions for given are not allowed!
        name (Optional): desired name of probability column. Defaults to P(<query>).
        steps (Optional): Number of grid steps in distribution.
                          Ignored if prior is set.
        prior (Optional): prior distribution to use in Bayes formula.
                          Defaults to flat prior.

    Returns:
        polars DataFrame containing probability distributions

    Raises:
        VariableDoesNotExistError if a variable does not appear in the data.
        VariableNotBinomialError if the event is not Binomial

    """
    q = _parse_query(data, query, name)
    return _p_dist(data, q, steps, prior)


# ---- Function that checks for conditional independence


def check_conditional_independence(
    data: pl.DataFrame, x: str, y: str, z: str, alpha: float = 0.05
) -> bool:
    """Check for conditional independence using Conditional Mutual Information.

    References:
        https://en.wikipedia.org/wiki/Conditional_mutual_information
        https://en.wikipedia.org/wiki/G-test

    Args:
        data: dataset from which the conditional independence is to be checked.
        x: representing X in X ⫫ Y | Z
        y: representing Y in X ⫫ Y | Z
        z: representing Z in X ⫫ Y | Z. If multiple variables, express as "A,B,C" etc.
        alpha: critical value for χ² critical value

    Returns:
        boolean indicating if conditional independence holds in data.

    """  # noqa: E501
    pxyz = p_array(data, f"{x},{y},{z}")
    pxy_z = p_array(data, f"{x},{y}|{z}")
    px_z = p_array(data, f"{x}|{z}")
    py_z = p_array(data, f"{y}|{z}")

    # Conditional Mutual information term: Σx,y,z P(x,y,z)log(P(x,y|z)/P(x|z)P(y|z))
    ratio = pxy_z / (px_z * py_z)
    # masking cases where ration = 0, avoiding warnings from np.log
    cmi = (pxyz * np.log(ratio.where(ratio > 0))).sum().item()

    # G-test for likelihood ratio
    nx = p_array(data, x).size
    ny = p_array(data, y).size
    nz = p_array(data, z).size
    dof = (nx - 1) * (ny - 1) * nz

    test_statistic = 2 * len(data) * cmi
    critical_value = stats.chi2.ppf(1 - alpha, dof)

    return (test_statistic <= critical_value).item()
