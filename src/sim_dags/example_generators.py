from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator

# --- These functions are minimal examples of how to implement a DAG.
# They are not a great way to show when different estimands are required,
# even in the Fork model P(y|x) is a decent estimate of P(y|do(x))
# in rough simulations, despite the proper estimand being ∑z P(y|x,z)P(z)


def get_pipe_simulator() -> DAGSimulator:
    """Generate samples from Pipe DAG."""
    distributions = [
        Categorical("x", 3),
        Categorical("z", 4, ["x"]),
        Binomial("y", ["x", "z"]),
    ]
    return DAGSimulator(distributions)


def get_fork_simulator() -> DAGSimulator:
    """Generate samples from Fork DAG."""
    distributions = [
        Categorical("z", 4),
        Categorical("x", 4, ["z"]),
        Binomial("y", ["x", "z"]),
    ]
    return DAGSimulator(distributions)


def get_collider_simulator() -> DAGSimulator:
    """Generate samples from Collider DAG."""
    distributions = [
        Categorical("x", 3),
        Binomial("y", ["x"]),
        Categorical("z", 4, ["x", "y"]),
    ]
    return DAGSimulator(distributions)


# --- Slightly more complicated DAG
# Naming scheme and visualisation of DAGs can be found in example_dags.pdf


def get_dag1_simulator() -> DAGSimulator:
    """Generate samples from DAG 1."""
    distributions = [
        Categorical("w", 4),
        Categorical("z", 4),
        Categorical("x", 3, ["w", "z"]),
        Binomial("y", ["w", "x", "z"]),
    ]
    return DAGSimulator(distributions)
