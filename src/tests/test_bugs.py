# Tests based on real world bugs

from sim_dags.dag_simulator import Binomial, Categorical, DAGSimulator
from sim_dags.graphs import (
    all_simple_paths,
    backdoor_criterion,
    conditional_independencies,
    find_d_separators,
    find_existing_paths,
    get_descendants,
)
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

    simple_paths = all_simple_paths(
        "O",
        "N",
        dag.dag._neighbours,  # noqa: SLF001
        dag.dag._reachable,  # noqa: SLF001
    )

    backdoor = backdoor_criterion(
        "O",
        "N",
        dag.dag.edges,
        simple_paths,
        dag.unobserved,
        get_descendants(dag.dag.edges, dag.dag.topological_generations),
    )

    assert isinstance(repr(backdoor), str)  # mostly for repr coverage
    assert backdoor.minimal_adjustment_sets is not None, (
        "Should have adjustment sets"
    )
    assert len(backdoor.minimal_adjustment_sets) == 1, (
        "Should be one valid adjustment set."
    )
    assert sorted(backdoor.minimal_adjustment_sets) == [["G", "T"]], (
        "Incorrect adjustment set found."
    )

    do_edges = dag.dag.mutilate(["I"], None)
    do_paths = find_existing_paths(do_edges, simple_paths)
    do_backdoor = backdoor_criterion(
        "O",
        "N",
        do_edges,
        do_paths,
        dag.unobserved,
        get_descendants(do_edges, dag.dag.topological_generations),
    )

    assert do_backdoor.minimal_adjustment_sets is not None, (
        "Should have adjustment sets"
    )
    assert len(do_backdoor.minimal_adjustment_sets) == 1, (
        "Should be one valid minimal_adjustment set."
    )
    assert sorted(do_backdoor.minimal_adjustment_sets) == [["G", "T"]], (
        "Incorrect minimal_adjustment set found."
    )

    cond = conditional_independencies(
        dag.dag.topological_sort,
        dag.dag.edges,
        set(),
        dag.unobserved,
        dag.dag._neighbours,  # noqa: SLF001
        get_descendants(dag.dag.edges, dag.dag.topological_generations),
        dag.dag._reachable,  # noqa: SLF001
        testable_only=True,
    )

    testable_len = len(cond.testable)
    # confirmed via dagitty.net/dgas.html#
    correct = [
        "C ⫫ T | G",
        "C ⫫ O | G",
        "C ⫫ S | A,G,L",
        "C ⫫ N | A,G,I,L,O,S,T",
        "A ⫫ G | C",
        "A ⫫ L | C",
        "A ⫫ T | C",
        "A ⫫ T | G",
        "A ⫫ O | C",
        "A ⫫ O | G",
        "G ⫫ L | C",
        "L ⫫ T | C",
        "L ⫫ T | G",
        "L ⫫ O | C",
        "L ⫫ O | G",
    ]
    assert testable_len == len(correct), (
        "Incorrect number of testable independencies."
    )
    assert cond.testable == correct, "Incorrect independencies"
    assert cond.untestable == [], "Should be no untestable dependencies"


def test_d_sep_bug() -> None:
    """Test a bug with is_d_separator()."""
    distributions = [
        Binomial("Va"),
        Binomial("Vg"),
        Binomial("Vi"),
        Binomial("Vl"),
        Binomial("Vo"),
        Binomial("Vs"),
        Binomial("Vt"),
        Binomial("C"),
        Binomial("G", ["C", "Vg"]),
        Binomial("L", ["C", "Vl"]),
        Binomial("A", ["C", "Va"]),
        Binomial("T", ["G", "Vt"]),
        Binomial("O", ["G", "T", "Vo"]),
        Binomial("M", ["G", "L", "O", "A"], unobserved=True),
        Binomial("D", ["T", "O", "M", "G"], unobserved=True),
        Binomial("S", ["T", "D", "Vs"]),
        Binomial("I", ["C", "G", "L", "T", "O", "S", "A", "Vi"]),
        Binomial("N", ["I", "D"]),
    ]
    dag = DAGSimulator(distributions)

    assert dag.is_d_separator("D", "Vg", {"A", "G", "L"}), (
        "Incorrect d-separator judgment"
    )
    v = {"Va", "Vg", "Vi", "Vl", "Vo", "Vs", "Vt"}
    correct_set = {"A", "G", "L", "T"}
    assert dag.is_d_separator("D", v, correct_set, over=["O"]), (
        "Incorrect d-separator judgement for all V"
    )

    edges = dag.dag.mutilate(["O"], None)
    descendants = get_descendants(edges, dag.dag.topological_generations)
    d_separators = find_d_separators(
        {"D"},
        v,
        edges,
        dag.dag._neighbours,  # noqa: SLF001
        descendants,
        dag.dag._reachable,  # noqa: SLF001
        dag.unobserved,
    )
    assert sorted(correct_set) in d_separators.separators, (
        "Correct d-separator doesn't appear in separators"
    )
    assert sorted(correct_set) in d_separators.minimal, (
        "Correct d-separator doesn't appear in minimal set"
    )
