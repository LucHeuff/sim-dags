from sim_dags.example_validation import compare_dag1, compare_simple_dags


def test_compare_simple_dags() -> None:
    """Test compare_simple_dags()."""
    compare_simple_dags(n_sizes=2, n_seeds=2)


def test_compare_dag1() -> None:
    """Test compare_dag1()."""
    compare_dag1(n_sizes=2, n_seeds=2)
