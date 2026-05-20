# Tests based on real world bugs

from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator
from sim_dags.graph_algorithms import backdoor_criterion, conditional_independencies
from sim_dags.probability import p, p_array


def test_p_array_duplicates() -> None:
    """Testing a bug where duplicates caused p_array to fail on MultiIndex conversion."""  # noqa: E501
    dag = DAGSimulator(
        [
            Categorical("r", 4),
            Categorical("a", 3, ["r"]),
            Categorical("o", 5, ["a", "r"]),
            Binomial("n", ["o", "a", "r"]),
        ]
    )

    sim = dag.sample(100, 12345)

    p(sim, "n|o")
    p_array(sim, "n|o")

    p(sim, "n|o,a")
    p_array(sim, "n|o,a")

    p(sim, "n|o,a,r")
    p_array(sim, "n|o,a,r")


def test_realistic_dag() -> None:
    """Bugs encountered on realistic DAG."""
    distributions = [
        Binomial("C"),
        Binomial("G", ["C"]),
        Binomial("L", ["C"]),
        Binomial("A", ["C"]),
        Binomial("T", ["G"]),
        Binomial("O", ["G", "T"]),
        Binomial("M", ["G", "L", "O", "A"], unobserved=True),
        Binomial("D", ["T", "O", "M", "G"], unobserved=True),
        Binomial("S", ["T", "D"]),
        Binomial("I", ["C", "G", "L", "T", "O", "S", "A"]),
        Binomial("N", ["I", "D"]),
    ]
    dag = DAGSimulator(distributions)

    backdoor = backdoor_criterion(dag.graph, "O", "N", [], set())
    assert len(backdoor.adjustment_sets) == 1, "Should be one valid adjustment set."
    assert sorted(backdoor.adjustment_sets[0]) == ["G", "T"], (
        "Incorrect adjustment set found."
    )

    do_backdoor = backdoor_criterion(dag.graph, "O", "N", ["I"], set())
    assert len(do_backdoor.adjustment_sets) == 1, (
        "Should be one valid adjustment set."
    )
    assert sorted(do_backdoor.adjustment_sets[0]) == ["G", "T"], (
        "Incorrect adjustment set found."
    )

    cond = conditional_independencies(dag.graph, None, None, dag.unobserved)

    testable_len = len(cond.testable)
    # confirmed via dagitty.net/dgas.html#
    correct = [
        "C ⫫ T | G",
        "C ⫫ O | G",
        "C ⫫ S | A,G,L",
        "C ⫫ N | A,G,I,L,O,S,T",
        "G ⫫ L | C",
        "G ⫫ A | C",
        "L ⫫ A | C",
        "L ⫫ T | C",
        "L ⫫ T | G",
        "L ⫫ O | C",
        "L ⫫ O | G",
        "A ⫫ T | C",
        "A ⫫ T | G",
        "A ⫫ O | C",
        "A ⫫ O | G",
    ]
    assert testable_len == 15, "Incorrect number of testable independencies."  # noqa: PLR2004
    assert cond.testable == correct
