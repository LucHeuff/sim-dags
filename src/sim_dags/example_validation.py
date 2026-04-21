from functools import partial
from typing import Protocol

import altair as alt
import polars as pl

from sim_dags.example_generators import (
    DAG1Params,
    SimpleDAGParams,
    generate_collider,
    generate_dag1,
    generate_fork,
    generate_pipe,
)
from sim_dags.iterate_sims import SimulateFunction, iterate_samples, plot_samples
from sim_dags.probability import p, p_array
from sim_dags.utils import Chart, default_chart_config, to_df


class GenerateFunction(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, size: int, seed: int, *, do_x: bool = False
    ) -> pl.DataFrame: ...


# slightly more involved due to having three simple DAGs
def get_simple_generator(gen: GenerateFunction) -> SimulateFunction:
    """Makes a SimulateFunction for the chosen generator."""
    obs_ = "P(y|x)"
    sum_ = "∑z P(y|x,z)P(z)"

    def func(size: int, seed: int) -> pl.DataFrame:
        sim = gen(size=size, seed=seed)
        do_sim = gen(size=size, seed=seed, do_x=True)

        py_x = p(sim, "y|x")
        py_do_x = p(do_sim, "y|do(x)")
        py_xz_pz = to_df(
            (p_array(sim, "y|x,z") * p_array(sim, "z")).sum(dim="z").rename(sum_)
        )

        return (
            py_x.join(py_xz_pz, on=["y", "x"])
            .join(py_do_x, left_on=["y", "x"], right_on=["y", "do(x)"])
            .with_columns(
                (pl.col("P(y|do(x))") - pl.col(obs_)).pow(2).sqrt().alias(obs_),
                (pl.col("P(y|do(x))") - pl.col(sum_)).pow(2).sqrt().alias(sum_),
            )
            .select(pl.col([obs_, sum_]).mean())
            .unpivot(variable_name="estimand")
        )

    return func


def compare_simple_dags() -> alt.VConcatChart:
    """Generate comparison for simple DAGs."""
    params = SimpleDAGParams(12345)

    pipe = partial(generate_pipe, params=params)
    fork = partial(generate_fork, params=params)
    collider = partial(generate_collider, params=params)

    n_sizes = 5
    n_seeds = 10

    pipe_chart = plot_samples(
        iterate_samples(get_simple_generator(pipe), n_sizes=n_sizes, n_seeds=n_seeds)
    ).properties(title="Comparison for Pipe DAG (correct: P(y|x))")
    fork_chart = plot_samples(
        iterate_samples(get_simple_generator(fork), n_sizes=n_sizes, n_seeds=n_seeds)
    ).properties(title="Comparison for Fork DAG (correct: ∑z P(y|x,z)P(z))")
    collider_chart = plot_samples(
        iterate_samples(
            get_simple_generator(collider), n_sizes=n_sizes, n_seeds=n_seeds
        )
    ).properties(title="Comparison for Collider DAG (correct: P(y|x))")

    return default_chart_config(
        alt.vconcat(pipe_chart, fork_chart, collider_chart)
    ).configure_range(category=["orangered", "forestgreen"])


def compare_dag1() -> Chart:
    """Generate comparison for DAG1."""
    params = DAG1Params(12345)

    obs_ = "P(y|x)"
    wrong_est_ = "∑w P(y|x, w)P(w)"
    est_ = "∑w ∑z P(y|x, z, w)P(z)P(w)"

    def sim_func(size: int, seed: int) -> pl.DataFrame:
        sim = generate_dag1(size=size, params=params, seed=seed)
        do_sim = generate_dag1(size=size, params=params, seed=seed, do_x=True)

        py_x = p(sim, "y|x")
        py_do_x = p(do_sim, "y|do(x)")

        est = to_df(
            (p_array(sim, "y|x, z, w") * p_array(sim, "z") * p_array(sim, "w"))
            .sum(dim=["z", "w"])
            .rename(est_)
        )
        wrong_est = to_df(
            (p_array(sim, "y|x,w") * p_array(sim, "w"))
            .sum(dim="w")
            .rename(wrong_est_)
        )

        on = ["y", "x"]

        return (
            py_x.join(est, on=on)
            .join(wrong_est, on=on)
            .join(py_do_x, left_on=on, right_on=["y", "do(x)"])
            .with_columns(
                (pl.col("P(y|do(x))") - pl.col(obs_)).pow(2).sqrt().alias(obs_),
                (pl.col("P(y|do(x))") - pl.col(est_)).pow(2).sqrt().alias(est_),
                (pl.col("P(y|do(x))") - pl.col(wrong_est_))
                .pow(2)
                .sqrt()
                .alias(wrong_est_),
            )
            .select(pl.col([obs_, est_, wrong_est_]).mean())
            .unpivot(variable_name="estimand")
        )

    n_sizes = 5
    n_seeds = 10

    return (
        default_chart_config(
            plot_samples(iterate_samples(sim_func, n_sizes, n_seeds))
        )
        .properties(
            title="Comparison of DAG 1 (correct: ∑w ∑z P(y|x, z, w)P(z)P(w))"
        )
        .configure_range(category=["indigo", "orangered", "forestgreen"])
    )


def main() -> None:
    """Main runner."""
    compare_dag1().save("test.png")


if __name__ == "__main__":
    import subprocess

    subprocess.run(["clear"], shell=True, check=True)
    main()
    print("=== Klaar ====")  # noqa: T201
