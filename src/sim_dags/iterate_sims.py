from typing import Protocol

import altair as alt
import pandera.polars as pa
import polars as pl


class CompareFunction(Protocol):  # noqa: D101
    def __call__(self, size: int, seed: int) -> pl.DataFrame: ...  # noqa: D102


class SimFunction(Protocol):  # noqa: D101
    def __call__(self, size: int, seed: int) -> pl.DataFrame: ...  # noqa: D102


class EstimandFunction(Protocol):  # noqa: D101
    def __call__(self, samples: pl.DataFrame) -> pl.DataFrame: ...  # noqa: D102


compare_schema = pa.DataFrameSchema(
    columns={"estimand": pa.Column(str), "value": pa.Column(float)}, strict=True
)


def build_compare_function(
    intervention_sim: SimFunction,
    intervention: EstimandFunction,
    observation_sim: SimFunction,
    estimands: EstimandFunction | list[EstimandFunction],
) -> CompareFunction:
    """Build a CompareFunction for use in iterate_samples().

    Assumes that all the distributions have the same relevant variables
    and variable names.
    For example, if estimating P(y|x) and ∑z P(y|x,z)P(z), then all
    dataframes are assumed to have only "x" and "y" as columns.

    Args:
        intervention_sim: function(size, seed) that generates intervention samples
        intervention: function(samples) that calculates intervention distribution
        observation_sim: function(size, seed) that generates observation samples
        estimands: (list of) function(samples) that calculates estimands distribution

    Returns:
        CompareFunction tobe used in iterate_samples().
    """
    estimands_: list[EstimandFunction] = (
        estimands if isinstance(estimands, list) else [estimands]
    )  # ty:ignore[invalid-assignment]

    def simulate_function(size: int, seed: int) -> pl.DataFrame:

        int_ = intervention_sim(size, seed)
        obs_ = observation_sim(size, seed)

        do = intervention(int_)
        # join columns should be the common columns between samples and intervention
        on = set(do.columns) & set(int_.columns)
        do_cols = set(do.columns).difference(on)
        assert len(do_cols) == 1, "Intervention should only add one column."
        do_col = next(iter(do_cols))

        ests = pl.concat([est(obs_) for est in estimands_], how="align")
        assert ests.null_count().sum_horizontal().item() == 0, (
            "Nulls appear in estimands. Increase sample sizes, or make sure to set p(*, include_zeros=True) on any of the relevant distributions."  # noqa: E501
        )

        # figuring out join columns for estimands
        on_ = set(ests.columns) & set(obs_.columns)
        assert on == on_, (
            f"intervention and estimands should result in the same common columns,\nbut got {on} for intervention and {on_} for estimands.\nIf you are using interventions, remember to set rename_do=False as a parameter on DAGSimulator.sample!"  # noqa: E501
        )
        est_names = list(set(ests.columns).difference(on_))

        return (
            do.join(ests, on=list(on))
            .with_columns(
                [
                    (pl.col(do_col) - pl.col(est_name)).pow(2).sqrt().alias(est_name)
                    for est_name in est_names
                ]
            )
            .select(pl.col(est_names).mean())
            .unpivot(variable_name="estimand")
        )

    return simulate_function


def iterate_simulations(
    compare: CompareFunction,
    n_sizes: int = 5,
    n_seeds: int = 10,
    start_order: int = 2,
    seed_offset: int = 12345,
) -> pl.DataFrame:
    """Iterate over generated samples.

    Args:
        compare: CompareFunction returning a single comparison DataFrame for iteration
            DataFrame must consist of an "estimand" column
               and a "value" column containing the value of interest
        n_sizes: number of orders of magnitude for size.
        n_seeds: number of seeds for sample generation
        start_order: at which order of magnitude to start sizes. Defaults to 2 (=100)
        seed_offset: starting offset for seeds

    Returns:
        pl.DataFrame with combined results per iteration
    """  # noqa: E501
    assert n_seeds > 1, f"n_seeds must be greater than 1, got {n_seeds}"

    def get_sims(
        compare: CompareFunction, size: int, seed_offset: int
    ) -> pl.DataFrame:
        seeds = [seed_offset + n for n in range(n_seeds)]

        return (
            pl.concat(
                [
                    compare_schema.validate(compare(size=size, seed=seed))
                    for seed in seeds
                ]
            )
            .group_by("estimand")
            .agg(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std"),
            )
            .with_columns(
                (pl.col("mean") - pl.col("std")).alias("mean - std"),
                (pl.col("mean") + pl.col("std")).alias("mean + std"),
                size=pl.lit(size),
            )
        )

    assert n_sizes >= 1, f"n_sizes must be >= 1, got {n_sizes} "

    def get_sizes(compare: CompareFunction, seed_offset: int) -> pl.DataFrame:
        sizes = [10**n for n in range(start_order, start_order + n_sizes)]
        return pl.concat([get_sims(compare, s, seed_offset) for s in sizes])

    return get_sizes(compare, seed_offset)


def plot_simulations(data: pl.DataFrame) -> alt.LayerChart:
    """Plot data generated by iterate_samples()."""
    base = (
        alt.Chart(data)
        .encode(
            alt.X("size").scale(type="log").title("Sample size"),
            alt.Color(
                "estimand",
                legend=alt.Legend(
                    labelLimit=0, orient="bottom", direction="vertical"
                ),
            ),
        )
        .properties(width=600, height=300)
    )

    max_ = data.select(pl.col("mean + std").max()).item()

    mean = base.mark_line(interpolate="monotone").encode(alt.Y("mean:Q").title(None))
    uncertainty = base.mark_area(
        opacity=0.2, interpolate="monotone", clip=True
    ).encode(
        alt.Y("mean - std:Q").scale(domain=(0, max_)),
        alt.Y2("mean + std:Q"),
    )

    return uncertainty + mean
