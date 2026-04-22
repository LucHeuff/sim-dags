import altair as alt
import polars as pl

from sim_dags.dag_simulator import DAGSimulator
from sim_dags.example_generators import (
    get_collider_simulator,
    get_dag1_simulator,
    get_fork_simulator,
    get_pipe_simulator,
)
from sim_dags.iterate_sims import (
    CompareFunction,
    build_compare_function,
    iterate_samples,
    plot_samples,
)
from sim_dags.probability import p, p_array
from sim_dags.utils import Chart, default_chart_config, to_df


# slightly more involved due to having three simple DAGs,
# but this function simple makes a SimulateFunction for the simple DAGSimulators
def get_simple_generator(gen: DAGSimulator) -> CompareFunction:
    """Makes a SimulateFunction for the chosen generator."""
    sum_ = "∑z P(y|x,z)P(z)"
    return build_compare_function(
        gen,
        intervention=lambda samples: p(samples, "y|x", name="do"),
        # calculating ∑z P(y|x,z)P(z) using p_arrays
        estimands={
            sum_: lambda samples: to_df(
                (p_array(samples, "y|x,z") * p_array(samples, "z"))
                .sum(dim="z")
                .rename(sum_)
            )
        },
    )


def compare_simple_dags(n_sizes: int = 5, n_seeds: int = 10) -> alt.VConcatChart:
    """Generate comparison for simple DAGs."""
    pipe = get_pipe_simulator()
    fork = get_fork_simulator()
    collider = get_collider_simulator()

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


# --- Comparing DAG 1, writing the SimulateFunction manually.


def compare_dag1(n_sizes: int = 5, n_seeds: int = 10) -> Chart:
    """Generate comparison for DAG1.

    This is an example of writing the SimulateFunction manually.
    """
    obs_ = "P(y|x)"
    wrong_est_ = "∑w P(y|x, w)P(w)"
    est_ = "∑w ∑z P(y|x, z, w)P(z)P(w)"

    simulator = get_dag1_simulator()

    def sim_func(size: int, seed: int) -> pl.DataFrame:
        samples = simulator.sample(size=size, seed=seed)
        do_samples = simulator.sample(size=size, seed=seed, do={"x": True})

        py_x = p(samples, "y|x")
        py_do_x = p(do_samples, "y|do(x)")

        est = to_df(
            (
                p_array(samples, "y|x, z, w")
                * p_array(samples, "z")
                * p_array(samples, "w")
            )
            .sum(dim=["z", "w"])
            .rename(est_)
        )
        wrong_est = to_df(
            (p_array(samples, "y|x,w") * p_array(samples, "w"))
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

    return (
        default_chart_config(
            plot_samples(iterate_samples(sim_func, n_sizes, n_seeds))
        )
        .properties(
            title="Comparison of DAG 1 (correct: ∑w ∑z P(y|x, z, w)P(z)P(w))"
        )
        .configure_range(category=["indigo", "orangered", "forestgreen"])
    )


def main() -> None:  # pragma: no cover
    """Main runner."""
    compare_simple_dags().save("simple_dags.png")
    compare_dag1().save("dag1.png")


if __name__ == "__main__":  # pragma: no cover
    import subprocess

    subprocess.run(["clear"], shell=True, check=True)
    main()
    print("=== Klaar ====")  # noqa: T201
