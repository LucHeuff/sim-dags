import subprocess
from collections.abc import Callable

import altair as alt
import polars as pl

from sim_dags.generators import (
    SimpleDAGParams,
    generate_collider,
    generate_fork,
    generate_pipe,
)
from sim_dags.probability import p, p_array
from sim_dags.utils import Chart, default_chart_config, to_df

subprocess.run(["clear"], shell=True, check=True)

SEED = 12345
NZ = 4
NX = 3
N_SIZES = 6
N_SEEDS = 15

SUM_ = "∑z P(y|x,z)P(z)"

params = SimpleDAGParams(SEED, NZ, NX)


def compare_estimands(
    gen: Callable,
    title: str,
    n_sizes: int = 5,
    n_seeds: int = 10,
    nz: int = 4,
    nx: int = 3,
    seed_offset: int = 12345,
) -> Chart:
    """Compare estimands in simple DAGs through simulations."""

    def get_sim(size: int, seed: int) -> pl.DataFrame:
        params = SimpleDAGParams(seed, nz, nx)
        sim = gen(size, params)
        do_sim = gen(size, params, do_x=True)

        py_x = p(sim, "y|x")
        py_do_x = p(do_sim, "y|do(x)").rename({"do(x)": "x"})

        py_xz_pz = to_df(
            (p_array(sim, "y|x,z") * p_array(sim, "z")).sum(dim="z").rename(SUM_)
        )
        return (
            py_do_x.join(py_x, on=["y", "x"])
            .join(py_xz_pz, on=["y", "x"])
            .with_columns(
                (pl.col("P(y|do(x))") - pl.col("P(y|x)")).pow(2).sqrt().alias("L2"),
                (pl.col("P(y|do(x))") - pl.col(SUM_)).pow(2).sqrt().alias("L2_sum"),
            )
            .select(pl.col(["L2", "L2_sum"]).mean())
            .unpivot(variable_name="estimand", value_name="L2")
            .with_columns(
                pl.col("estimand").replace({"L2": "P(y|x)", "L2_sum": SUM_})
            )
        )

    def get_sim_seeds(size: int, n_seeds: int, seed_offset: int) -> pl.DataFrame:
        seeds = [seed_offset + n for n in range(n_seeds)]

        return (
            pl.concat([get_sim(size, seed) for seed in seeds])
            .group_by("estimand")
            .agg(pl.col("L2").mean().alias("mean"), pl.col("L2").std().alias("std"))
            .with_columns(
                (pl.col("mean") - pl.col("std")).alias("mean - std"),
                (pl.col("mean") + pl.col("std")).alias("mean + std"),
                size=pl.lit(size),
            )
        )

    def get_sim_sizes(n_sizes: int, n_seeds: int, seed_offset: int) -> pl.DataFrame:
        sizes = [10**n for n in range(1, n_sizes)]
        return pl.concat(
            [get_sim_seeds(size, n_seeds, seed_offset) for size in sizes]
        )

    data = get_sim_sizes(n_sizes, n_seeds, seed_offset)

    base = (
        alt.Chart(data)
        .encode(
            alt.X("size").scale(type="log").title("Sample size"),
            alt.Color("estimand"),
        )
        .properties(width=700, height=300, title=title)
    )

    means = base.mark_line(interpolate="monotone").encode(
        alt.Y("mean:Q").title("Difference estimand to intervention")
    )

    uncertainty = base.mark_area(opacity=0.2, interpolate="monotone").encode(
        alt.Y("mean - std:Q"), alt.Y2("mean + std:Q")
    )

    return means + uncertainty


def compare_dags() -> None:
    """Runner function combining comparison for all the DAGs."""
    pipe_chart = compare_estimands(
        generate_pipe,
        title="Comparison for pipe DAG",
        n_sizes=N_SIZES,
        n_seeds=N_SEEDS,
    )
    fork_chart = compare_estimands(
        generate_fork,
        title="Comparison for fork DAG",
        n_sizes=N_SIZES,
        n_seeds=N_SEEDS,
    )
    collider_chart = compare_estimands(
        generate_collider,
        title="Comparison for collider DAG",
        n_sizes=N_SIZES,
        n_seeds=N_SEEDS,
    )

    default_chart_config(alt.vconcat(pipe_chart, fork_chart, collider_chart)).save(
        "test.png"
    )


if __name__ == "__main__":
    compare_dags()
