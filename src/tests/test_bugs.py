# Tests based on real world bugs

from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator
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
        Binomial("I", ["C", "G", "L", "T", "O", "S"]),
        Binomial("N", ["I", "D"]),
    ]
    dag = DAGSimulator(distributions)

    backdoor = dag._backdoor("O", "N", [])  # noqa: SLF001
    assert len(backdoor.adjustment_sets) == 1, "Should be one valid adjustment set."
    assert sorted(backdoor.adjustment_sets[0]) == ["G", "T"], (
        "Incorrect adjustment set found."
    )

    do_backdoor = dag._backdoor("O", "N", ["I"])  # noqa: SLF001
    assert len(do_backdoor.adjustment_sets) == 1, (
        "Should be one valid adjustment set."
    )
    assert sorted(do_backdoor.adjustment_sets[0]) == ["G", "T"], (
        "Incorrect adjustment set found."
    )

    cond = dag._conditional([], [])  # noqa: SLF001

    testable_len = sum(len(value) for value in cond.testable.values())

    assert testable_len == 15, "Incorrect number of testable independencies."  # noqa: PLR2004
    assert cond.testable["A ⫫ T"] == [["C"], ["G"]], "Problem with A ⫫ T"
    assert cond.testable["A ⫫ O"] == [["C"], ["G"]], "Problem with A ⫫ O"
    assert cond.testable["L ⫫ T"] == [["C"], ["G"]], "Problem with L ⫫ T"
    assert cond.testable["L ⫫ O"] == [["C"], ["G"]], "Problem with L ⫫ O"
