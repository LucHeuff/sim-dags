from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator
from sim_dags.iterate_sims import (
    build_simulate_function,
    iterate_samples,
    plot_samples,
)
from sim_dags.probability import p, p_array
from sim_dags.utils import default_chart_config, to_df


def test_iterate_sims() -> None:
    """Integration test of iterate_sims.py."""
    distributions = [
        Categorical("x", 3),
        Categorical("z", 4, ["x"]),
        Binomial("y", ["x", "z"]),
    ]

    dag_simulator = DAGSimulator(distributions)

    est_ = "∑z P(y|x,z)P(z)"

    sim_func = build_simulate_function(
        dag_simulator,
        intervention=lambda samples: p(samples, "y|x", name="do"),
        estimands={
            est_: lambda samples: to_df(
                (p_array(samples, "y|x,z") * p_array(samples, "z"))
                .sum(dim="z")
                .rename(est_)
            )
        },
        true_do={"x": True},
    )

    n_sizes = 3
    n_seeds = 2

    default_chart_config(plot_samples(iterate_samples(sim_func, n_sizes, n_seeds)))
