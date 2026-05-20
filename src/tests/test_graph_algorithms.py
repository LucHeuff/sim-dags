import networkx as nx
import polars as pl
import pytest
from sim_dags.graph_algorithms import (
    backdoor_criterion,
    calculate_node_positions,
    conditional_independencies,
    find_minimal_adjustment_set,
    find_minimal_d_separators,
    mutilate,
    path_has_collider,
    render_path,
)


@pytest.fixture
def graph() -> nx.DiGraph:
    """Basic network graph."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("a", "c"), ("b", "c")]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def test_mutilate(graph: nx.DiGraph) -> None:
    """Test mutilate()."""
    same_graph = mutilate(graph, None, None)
    assert same_graph.nodes == graph.nodes, "nodes shoudn't change"
    assert same_graph.edges == graph.edges, "edges shoudn't change"

    over_b = mutilate(graph, over=["b"], under=None)
    assert over_b.nodes == graph.nodes, "nodes shouldn't change"
    assert list(over_b.edges) == [("a", "c"), ("b", "c")], (
        "only (a, b) should be removed"
    )

    over_bc = mutilate(graph, over=["b", "c"], under=None)
    assert over_bc.nodes == graph.nodes, "nodes shouldn't change"
    assert list(over_bc.edges) == [], "no edges should remain"

    under_a = mutilate(graph, under=["a"], over=None)
    assert under_a.nodes == graph.nodes, "nodes shouldn't change"
    assert list(under_a.edges) == [("b", "c")], "only (b, c) should remain"

    over_c_under_a = mutilate(graph, over=["c"], under=["a"])
    assert over_c_under_a.nodes == graph.nodes, "nodes shouldn't change"
    assert list(over_c_under_a.edges) == [], "no edges should remain"


def test_path_has_collider(graph: nx.DiGraph) -> None:
    """Test path_has_collider()."""
    assert not path_has_collider(graph, ["a", "b"]), "No collider on a -> b"
    assert not path_has_collider(graph, ["b", "a"]), "No collider on b <- a"
    assert path_has_collider(graph, ["a", "c", "b"]), "Collider on a -> c <- b"
    assert path_has_collider(graph, ["b", "c", "a"]), "Collider on b -> c <- a"
    assert not path_has_collider(graph, ["a", "b", "c"]), (
        "No collider on a -> b -> c"
    )
    assert not path_has_collider(graph, ["b", "a", "c"]), (
        "No collider on b <- a -> c"
    )


def test_find_minimal_adjustment_set() -> None:
    """Test find_minimal_adjustment_set()."""
    assert find_minimal_adjustment_set(["b"], [["a", "b", "c"]]) == [["b"]], (
        "incorrect adjustment set."
    )
    assert find_minimal_adjustment_set([], [["a", "b", "c"]]) is None, (
        "incorrect adjustment set."
    )
    assert find_minimal_adjustment_set(["d"], [["a", "b", "c"]]) is None, (
        "incorrect adjustment set."
    )
    assert find_minimal_adjustment_set(
        ["b", "c", "d"],
        [["a", "c", "e"], ["a", "b", "c", "d", "e"], ["a", "b", "d", "e"]],
    ) == [["b", "c"], ["c", "d"], ["b", "c", "d"]], "Incorrect adjustment set"

    # making a test that hits the size break
    set_ = find_minimal_adjustment_set(
        list("abcdefgh"),
        [
            ["a", "b", "e"],
            ["a", "c", "e"],
            ["b", "d", "f"],
        ],
    )
    if set_ is not None:
        assert len(set_) == 24, "Incorrect adjustment set"  # noqa: PLR2004


def test_find_minimal_d_separators(graph: nx.DiGraph) -> None:
    """Test find_minimal_d_separators()."""
    # test graph should have no d-separators.
    assert find_minimal_d_separators(graph, "a", "b") is None, (
        "test graph shouldn't have d-separators"
    )
    assert find_minimal_d_separators(graph, "b", "c") is None, (
        "test graph shouldn't have d-separators"
    )
    assert find_minimal_d_separators(graph, "a", "c") is None, (
        "test graph shouldn't have d-separators"
    )

    # Somewhat more complicated model
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("v", "r"),
            ("w", "r"),
            ("w", "y"),
            ("x", "v"),
            ("x", "y"),
            ("z", "x"),
            ("z", "y"),
        ]
    )
    assert find_minimal_d_separators(graph, "x", "r") == [["v"]], (
        "wrong d-separators for x ⫫ r"
    )
    assert find_minimal_d_separators(graph, "x", "w") == [[]], (
        "wrong d-separators for x ⫫ w"
    )
    assert find_minimal_d_separators(graph, "y", "v") == [["x"]], (
        "wrong d-separators for y ⫫ v"
    )
    assert find_minimal_d_separators(graph, "v", "w") == [[]], (
        "wrong d-separators for v ⫫ w"
    )
    d_sep = find_minimal_d_separators(graph, "y", "r")
    # I don't care about the order of the results per se
    assert d_sep in ([["w", "x"], ["v", "w"]], [["v", "w"], ["w", "x"]]), (
        "wrong d-separators for y ⫫ r"
    )


def test_render_path(graph: nx.DiGraph) -> None:
    """Test render_path()."""
    assert render_path(graph, ["a", "b"]) == "a -> b"
    assert render_path(graph, ["b", "a"]) == "b <- a"
    assert render_path(graph, ["b", "c"]) == "b -> c"
    assert render_path(graph, ["c", "a"]) == "c <- a"
    assert render_path(graph, ["c", "b"]) == "c <- b"
    assert render_path(graph, ["a", "b", "c"]) == "a -> b -> c"
    assert render_path(graph, ["c", "a", "b"]) == "c <- a -> b"
    assert render_path(graph, ["b", "c", "a"]) == "b -> c <- a"

    with pytest.raises(AssertionError):
        render_path(graph, ["a", "d"])


@pytest.fixture
def pipe() -> nx.DiGraph:
    """Pipe model."""
    edges = [("x", "y"), ("x", "z"), ("z", "y")]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def fork() -> nx.DiGraph:
    """Fork model."""
    edges = [("x", "y"), ("z", "x"), ("z", "y")]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def m_model() -> nx.DiGraph:
    """M model."""
    edges = [("x", "y"), ("w", "x"), ("w", "z"), ("v", "z"), ("v", "y")]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def complex_model() -> nx.DiGraph:
    """Complex model."""
    edges = [
        ("x", "y"),
        ("w", "x"),
        ("w", "r"),
        ("v", "r"),
        ("v", "y"),
        ("z", "x"),
        ("z", "y"),
    ]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def test_backdoor_criterion(
    pipe: nx.DiGraph,
    fork: nx.DiGraph,
    m_model: nx.DiGraph,
    complex_model: nx.DiGraph,
) -> None:
    """Test backdoor_criterion()."""
    # testing pipe
    pipe_back = backdoor_criterion(pipe, "x", "y", [], set())
    assert pipe_back.backdoor_paths == [], "Pipe should have no backdoor paths."
    assert pipe_back.open_paths == [], "Pipe should have no open paths"
    assert pipe_back.adjustment_sets == [], "Pipe doesn't need adjustment"

    # testing fork
    fork_back = backdoor_criterion(fork, "x", "y", [], set())
    assert fork_back.backdoor_paths == [["x", "z", "y"]], (
        "Fork should have x -> z -> y backdoor path"
    )
    assert fork_back.open_paths == ["x <- z -> y"], (
        "Fork should have x -> z -> y open path"
    )
    assert fork_back.adjustment_sets == [["z"]], "Fork adjustment set should be {z}"

    # testing fork with unobserved z
    fork_back_unobs = backdoor_criterion(fork, "x", "y", [], {"z"})
    assert fork_back_unobs.backdoor_paths == [["x", "z", "y"]], (
        "Fork unobs should have x -> z -> y backdoor path"
    )
    assert fork_back_unobs.open_paths == ["x <- z -> y"], (
        "Fork unobs should have x -> z -> y open path"
    )
    assert fork_back_unobs.adjustment_sets == [], (
        "Fork unobs adjustment set should be {}"
    )

    # testing fork with do(y)
    fork_back_do = backdoor_criterion(fork, "x", "y", ["y"], set())
    assert fork_back_do.backdoor_paths == [], (
        "fork do(y) should have no backdoor paths"
    )
    assert fork_back_do.open_paths == [], "fork do(y) should have no open paths"
    assert fork_back_do.adjustment_sets == [], "fork do(y) needs no adjustment"

    # testing M model
    m_back = backdoor_criterion(m_model, "x", "y", [], set())
    assert m_back.backdoor_paths == [["x", "w", "z", "v", "y"]], (
        "M model should have one backdoor path"
    )
    assert m_back.open_paths == [], (
        "M model backdoor path should be closed by collider z"
    )
    assert m_back.adjustment_sets == [], "M model needs no adjustment"

    # testing complex model, has backdoor path but no adjustment set
    complex_back = backdoor_criterion(complex_model, "x", "y", [], {"z"})
    assert complex_back.backdoor_paths == [["x", "z", "y"]], (
        "complex model should have one backdoor path"
    )
    assert complex_back.open_paths == ["x <- z -> y"], (
        "complex model should have one open path"
    )
    assert complex_back.adjustment_sets == [], (
        "complex model should have no adjustment set"
    )


def test_conditional_independencies(
    complex_model: nx.DiGraph, pipe: nx.DiGraph
) -> None:
    """Test conditional_independencies()."""
    cond = conditional_independencies(complex_model, None, None, None)
    assert isinstance(cond.render_testable, str), "render isn't a string"
    assert isinstance(cond.render_untestable, str), "render isn't a string"
    assert isinstance(repr(cond), str), "repr isn't a string"

    # validated against dagitty
    correct = [
        "x ⫫ r | w",
        "x ⫫ v",
        "y ⫫ w | x,z",
        "y ⫫ r | v,w",
        "y ⫫ r | v,x,z",
        "w ⫫ v",
        "w ⫫ z",
        "r ⫫ z",
        "v ⫫ z",
    ]
    assert cond.testable == correct
    assert cond.untestable == []

    cond_unobs = conditional_independencies(complex_model, None, None, {"z"})
    assert isinstance(cond_unobs.render_testable, str), "render isn't a string"
    assert isinstance(cond_unobs.render_untestable, str), "render isn't a string"
    assert isinstance(repr(cond_unobs), str), "repr isn't a string"
    # Testing if z is unobserved
    correct_testable = [
        "x ⫫ r | w",
        "x ⫫ v",
        "y ⫫ r | v,w",
        "w ⫫ v",
    ]
    correct_untestable = [
        "y ⫫ w | x,(z)",
        "y ⫫ r | v,x,(z)",
        "w ⫫ (z)",
        "r ⫫ (z)",
        "v ⫫ (z)",
    ]
    assert cond_unobs.testable == correct_testable
    assert cond_unobs.untestable == correct_untestable

    # Ignoring z should remove a few untestable implications
    cond_unobs_ignore_z = conditional_independencies(
        complex_model, None, ["z"], {"z"}
    )
    correct_untestable_ignore_z = [
        "y ⫫ w | x,(z)",
        "y ⫫ r | v,x,(z)",
    ]
    assert cond_unobs_ignore_z.testable == correct_testable
    assert cond_unobs_ignore_z.untestable == correct_untestable_ignore_z

    # intervening on Z should not do anything to implications
    do_cond = conditional_independencies(complex_model, ["z"], None, None)
    assert do_cond.testable == correct
    assert do_cond.untestable == []

    # pipe model shouldn't have any conditional independencies
    pipe_cond = conditional_independencies(pipe, None, None, None)
    assert pipe_cond.testable == []
    assert pipe_cond.untestable == []

    assert isinstance(pipe_cond.render_testable, str), "render isn't a string"
    assert isinstance(pipe_cond.render_untestable, str), "render isn't a string"
    assert isinstance(repr(pipe_cond), str), "repr isn't a string"


def test_calculate_node_positions(
    pipe: nx.DiGraph, complex_model: nx.DiGraph
) -> None:
    """Test calculate_node_positions()."""
    pos = calculate_node_positions(pipe).sort("node")

    assert isinstance(pos, pl.DataFrame), "wrong datatype"
    assert pos.columns == ["node", "x", "y"], "wrong columns"
    assert pos["node"].to_list() == ["x", "y", "z"], "wrong nodes"
    assert pos["x"].to_list() == [0.0, 1.0, 0.5], "wrong layers"

    pos = calculate_node_positions(complex_model).sort("node")
    assert isinstance(pos, pl.DataFrame), "wrong datatype"
    assert pos.columns == ["node", "x", "y"], "wrong columns"
    assert pos["node"].to_list() == sorted(complex_model.nodes), "wrong nodes"
