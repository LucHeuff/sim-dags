import polars as pl
from sim_dags.example_generators import SimpleDAGParams, generate_pipe
from sim_dags.iterate_sims import iterate_samples, plot_samples
from sim_dags.probability import p
from sim_dags.utils import default_chart_config


def test_iterate_sims() -> None:
    """Integration test of iterate_sims.py."""
    obs_ = "P(y|x)"
    do_ = "P(y|do(x))"

    params = SimpleDAGParams(54321)

    def sim_func(size: int, seed: int) -> pl.DataFrame:
        sim = generate_pipe(params, size, seed)
        do_sim = generate_pipe(params, size, seed, do_x=True)

        py_x = p(sim, "y|x")
        py_do_x = p(do_sim, "y|do(x)")

        return (
            py_x.join(py_do_x, left_on=["y", "x"], right_on=["y", "do(x)"])
            .with_columns(pl.col(do_) - pl.col(obs_).pow(2).sqrt().alias(obs_))
            .select(pl.col([obs_]).mean())
            .unpivot(variable_name="estimand")
        )

    n_sizes = 3
    n_seeds = 2

    default_chart_config(plot_samples(iterate_samples(sim_func, n_sizes, n_seeds)))
