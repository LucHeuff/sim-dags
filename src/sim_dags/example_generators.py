import polars as pl

from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator

# --- These functions are minimal examples of how to implement a DAG.
# They are not a great way to show when different estimands are required,
# even in the Fork model P(y|x) is a decent estimate of P(y|do(x))
# in rough simulations, despite the proper estimand being ∑z P(y|x,z)P(z)


def generate_pipe(size: int, seed: int, *, do_x: bool = False) -> pl.DataFrame:
    """Generate samples from Pipe DAG."""
    distributions = [
        Categorical("x", 3),
        Categorical("z", 4, ["x"]),
        Binomial("y", ["x", "z"]),
    ]
    dag = DAGSimulator(distributions)

    do_x_ = {"x": True} if do_x else None

    return dag.sample(size, seed, do=do_x_)  # ty:ignore[invalid-argument-type]


def generate_fork(size: int, seed: int, *, do_x: bool = False) -> pl.DataFrame:
    """Generate samples from Fork DAG."""
    distributions = [
        Categorical("z", 4),
        Categorical("x", 4, ["z"]),
        Binomial("y", ["x", "z"]),
    ]
    dag = DAGSimulator(distributions)

    do_x_ = {"x": True} if do_x else None

    return dag.sample(size, seed, do=do_x_)  # ty:ignore[invalid-argument-type]


def generate_collider(size: int, seed: int, *, do_x: bool = False) -> pl.DataFrame:
    """Generate samples from Collider DAG."""
    distributions = [
        Categorical("x", 3),
        Binomial("y", ["x"]),
        Categorical("z", 4, ["x", "y"]),
    ]
    dag = DAGSimulator(distributions)

    do_x_ = {"x": True} if do_x else None

    return dag.sample(size, seed, do=do_x_)  # ty:ignore[invalid-argument-type]


# --- Example of manually
# Naming scheme and visualisation of DAGS can be found in example_dags.pdf


def generate_dag1(size: int, seed: int, *, do_x: bool = False) -> pl.DataFrame:
    """Generate samples from DAG 1."""
    distributions = [
        Categorical("w", 4),
        Categorical("z", 4),
        Categorical("x", 3, ["w", "z"]),
        Binomial("y", ["w", "x", "z"]),
    ]
    dag = DAGSimulator(distributions)

    do_x_ = {"x": True} if do_x else None

    return dag.sample(size, seed, do=do_x_)  # ty:ignore[invalid-argument-type]
