from collections.abc import Sized
from dataclasses import dataclass

import hypothesis.strategies as st
import networkx as nx
import pytest
from hypothesis import given
from sim_dags.exceptions import (
    MissingNodeError,
    NoDisjointSetsError,
    NotADAGError,
)
from sim_dags.graphs import (
    DirectedAcyclicGraph,
    Edges,
    Node,
    NodeMap,
    NodeSequence,
    all_simple_paths,
    backdoor_criterion,
    conditional_independencies,
    edges_to_nodes,
    find_colliders,
    find_d_separators,
    find_existing_paths,
    find_minimal_separators,
    find_open_paths,
    find_separators,
    get_descendants,
    get_neighbours,
    get_parents,
    get_reachable,
    get_topological_sortings,
    is_collider,
    is_d_separator,
    mutilate,
    path_exists,
    path_has_collider,
    path_is_closed,
    undirected_path_exists,
)

# Controleren of alles wel gebruikt wordt

# Testing DAG
#     A
#   ↙   ↘
# B       C
# ↓       ↓
# D       E
# ↓   ↘   ↓
# F       G


@pytest.fixture
def nodes() -> Sized[Node]:
    """Nodes of the DAG above."""
    return list("abcdefg")


@pytest.fixture
def topological_sort() -> NodeSequence:
    """Topological sort of the DAG above."""
    return ["a", "b", "c", "d", "e", "f", "g"]


@pytest.fixture
def topological_generations() -> list[NodeSequence]:
    """Topological sort of the DAG above."""
    return [["a"], ["b", "c"], ["d", "e"], ["f", "g"]]


@pytest.fixture
def edges() -> Edges:
    """Edges of the DAG above."""
    return {
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "e"),
        ("d", "f"),
        ("d", "g"),
        ("e", "g"),
    }


@pytest.fixture
def parents() -> NodeMap:
    """Parents of the DAG above."""
    return {
        "a": [],
        "b": ["a"],
        "c": ["a"],
        "d": ["b"],
        "e": ["c"],
        "f": ["d"],
        "g": ["d", "e"],
    }


@pytest.fixture
def neighbours() -> NodeMap:
    """Neighbours of the DAG above."""
    return {
        "a": ["b", "c"],
        "b": ["a", "d"],
        "c": ["a", "e"],
        "d": ["b", "f", "g"],
        "e": ["c", "g"],
        "f": ["d"],
        "g": ["d", "e"],
    }


@pytest.fixture
def descendants() -> NodeMap:
    """Descendants of the DAG above."""
    return {
        "a": ["b", "c", "d", "e", "f", "g"],
        "b": ["d", "f", "g"],
        "c": ["e", "g"],
        "d": ["f", "g"],
        "e": ["g"],
        "f": [],
        "g": [],
    }


@pytest.fixture
def reachable() -> NodeMap:
    """Reachable nodes of the DAG above."""
    return {
        "a": ["b", "c", "d", "e", "f", "g"],
        "b": ["a", "c", "d", "e", "f", "g"],
        "c": ["a", "b", "d", "e", "f", "g"],
        "d": ["a", "b", "c", "e", "f", "g"],
        "e": ["a", "b", "c", "d", "f", "g"],
        "f": ["a", "b", "c", "d", "e", "g"],
        "g": ["a", "b", "c", "d", "e", "f"],
    }


# Alternative DAG (more cycle)


# E → F ← G
# ↑   B   ↑
# | ↗ ↑ ↘ |
# A   |   D
#   ↘ | ↗
#     C


@pytest.fixture
def alt_nodes() -> Sized[Node]:
    """Nodes of alternative DAG above."""
    return list("abcdefghi")


@pytest.fixture
def alt_topological_sort() -> NodeSequence:
    """Topological sort for alternative DAG above."""
    return ["a", "h", "c", "e", "i", "b", "d", "g", "f"]


@pytest.fixture
def alt_topological_generations() -> list[NodeSequence]:
    """Topological sort of the alternative DAG above."""
    return [["a", "h"], ["c", "e", "i"], ["b"], ["d"], ["g"], ["f"]]


@pytest.fixture
def alt_edges() -> Edges:
    """Edges of the alternative DAG above."""
    return {
        ("a", "b"),
        ("a", "c"),
        ("a", "e"),
        ("b", "d"),
        ("c", "b"),
        ("c", "d"),
        ("d", "g"),
        ("e", "f"),
        ("g", "f"),
        ("h", "i"),  # separate pair of nodes
    }


@pytest.fixture
def alt_neighbours() -> NodeMap:
    """Neighbours of the alternative DAG above."""
    return {
        "a": ["b", "c", "e"],
        "b": ["a", "c", "d"],
        "c": ["a", "b", "d"],
        "d": ["b", "c", "g"],
        "e": ["a", "f"],
        "f": ["e", "g"],
        "g": ["d", "f"],
        "h": ["i"],
        "i": ["h"],
    }


@pytest.fixture
def alt_descendants() -> NodeMap:
    """Descendants of the DAG above."""
    return {
        "a": ["b", "c", "d", "e", "f", "g"],
        "b": ["d", "f", "g"],
        "c": ["b", "d", "f", "g"],
        "d": ["f", "g"],
        "e": ["f"],
        "f": [],
        "g": ["f"],
        "h": ["i"],
        "i": [],
    }


@pytest.fixture
def alt_reachable() -> NodeMap:
    """Reachable map of the DAG above."""
    return {
        "a": ["b", "c", "d", "e", "f", "g"],
        "b": ["a", "c", "d", "e", "f", "g"],
        "c": ["a", "b", "d", "e", "f", "g"],
        "d": ["a", "b", "c", "e", "f", "g"],
        "e": ["a", "b", "c", "d", "f", "g"],
        "f": ["a", "b", "c", "d", "e", "g"],
        "g": ["a", "b", "c", "d", "e", "f"],
        "h": ["i"],
        "i": ["h"],
    }


# --- Testing helper functions


def test_edges_to_nodes(nodes: Sized[Node], edges: Edges) -> None:
    """Test edges_to_nodes()."""
    assert edges_to_nodes(edges) == set(nodes), "Not all nodes recovered."


def test_mutilate(edges: Edges) -> None:
    """Test mutilate()."""
    over_g = {("d", "g"), ("e", "g")}
    under_a = {("a", "b"), ("a", "c")}
    over_b = {("a", "b")}
    assert mutilate(edges, ["g"], []) == (edges - over_g)
    assert mutilate(edges, [], ["a"]) == (edges - under_a)
    assert mutilate(edges, ["g"], ["a"]) == (edges - over_g - under_a)
    assert mutilate(edges, ["g", "b"], ["a", "f"]) == (
        edges - over_g - over_b - under_a
    )
    # removing all arrows pointing into something should leave nothing
    assert mutilate(edges, ["b", "c", "d", "e", "f", "g"], []) == set()
    # removing all arrows coming out of something should leave nothing
    assert mutilate(edges, [], ["a", "b", "c", "d", "e"]) == set()


def test_path_exists(edges: Edges) -> None:
    """Test path_exists()."""
    assert path_exists(["a", "c", "e", "g"], edges), "path should exist"
    assert not path_exists(["e", "g", "d", "f"], edges), "path shouldn't exist"


def test_undirected_path_exists(edges: Edges) -> None:
    """Test undirected_path_exists()."""
    assert undirected_path_exists(["a", "c", "e"], edges), "path should exist"
    assert undirected_path_exists(["e", "c", "a"], edges), "path should exist"
    assert undirected_path_exists(["a", "b", "d", "g"], edges), "path should exist"
    assert undirected_path_exists(["g", "d", "b", "a"], edges), "path should exist"
    assert not undirected_path_exists(["f", "g"], edges), "path should not exist"
    assert not undirected_path_exists(["g", "f"], edges), "path should not exist"
    assert not undirected_path_exists(["f", "g"], edges), "path should not exist"
    assert not undirected_path_exists(["g", "d", "e", "c"], edges), (
        "path should not exist"
    )
    assert not undirected_path_exists(["c", "e", "d", "g"], edges), (
        "path should not exist"
    )


def test_path_has_collider(edges: Edges) -> None:
    """Test path_has_collider()."""
    assert not path_has_collider(["a", "c"], edges)
    assert not path_has_collider(["a", "c", "e"], edges)
    assert path_has_collider(["d", "g", "e"], edges)
    assert path_has_collider(["e", "g", "d"], edges)


def test_is_collider(edges: Edges) -> None:
    """Test is_collider()."""
    assert not is_collider("a", ["a", "c"], edges)
    assert not is_collider("c", ["a", "c"], edges)
    assert not is_collider("c", ["a", "c", "e"], edges)
    assert is_collider("g", ["d", "g", "e"], edges)
    assert is_collider("g", ["b", "d", "g", "e", "c"], edges)
    # testing if node doesn't even appear in the path
    assert not is_collider("g", ["a", "c", "e"], edges)


def test_get_parents(nodes: set[Node], edges: Edges, parents: NodeMap) -> None:
    """Test get_parents()."""
    assert get_parents(nodes, edges) == parents


def test_get_neighbours(nodes: set[Node], edges: Edges, neighbours: NodeMap) -> None:
    """Test get_neighbours()."""
    assert get_neighbours(nodes, edges) == neighbours


def test_get_topological_sortings_raises(nodes: Sized[Node], edges: Edges) -> None:
    """Test if an exception is raised if the graph is not a DAG."""
    with pytest.raises(NotADAGError):
        get_topological_sortings(nodes, edges | {("g", "a")})


def test_get_topological_sortings(
    nodes: Sized[Node],
    edges: Edges,
    topological_sort: NodeSequence,
    topological_generations: list[NodeSequence],
) -> None:
    """Test get_topological_sort()."""
    ts = get_topological_sortings(nodes, edges)

    assert ts.topological_sort == topological_sort, "incorrect order."
    assert ts.topological_generations == topological_generations, (
        "incorrect generations"
    )


def test_alt_get_topological_sortings(
    alt_nodes: Sized[Node],
    alt_edges: Edges,
    alt_topological_sort: NodeSequence,
    alt_topological_generations: list[NodeSequence],
) -> None:
    """Test get_topological_sort()."""
    ts = get_topological_sortings(alt_nodes, alt_edges)

    assert ts.topological_sort == alt_topological_sort, "incorrect order."
    assert ts.topological_generations == alt_topological_generations, (
        "incorrect generations"
    )


def test_get_descendants(
    topological_generations: list[NodeSequence], edges: Edges, descendants: NodeMap
) -> None:
    """Test get_descendants()."""
    assert get_descendants(edges, topological_generations) == descendants


def test_alt_get_descendants(
    alt_topological_generations: list[NodeSequence],
    alt_edges: Edges,
    alt_descendants: NodeMap,
) -> None:
    """Test get_descendants()."""
    assert get_descendants(alt_edges, alt_topological_generations) == alt_descendants


def test_get_reachable(neighbours: NodeMap, reachable: NodeMap) -> None:
    """Test get_reachable()."""
    assert get_reachable(neighbours) == reachable


def test_alt_get_reachable(alt_neighbours: NodeMap, alt_reachable: NodeMap) -> None:
    """Test get_reachable()."""
    assert get_reachable(alt_neighbours) == alt_reachable


def test_basic_all_simple_paths(neighbours: NodeMap, reachable: NodeMap) -> None:
    """Test all_simple_paths()."""

    # Helper function for comparing
    def compare_paths(
        source: Node, target: Node, true_paths: list[NodeSequence]
    ) -> None:
        p = all_simple_paths(source, target, neighbours, reachable)
        assert len(p) == len(true_paths), (
            f"Incorrect number of paths for {source}, {target}"
        )
        assert all(path in true_paths for path in p), (
            f"Incorrect paths found for {source}, {target}"
        )

    ab = [["a", "b"], ["a", "c", "e", "g", "d", "b"]]
    ba = [["b", "a"], ["b", "d", "g", "e", "c", "a"]]

    compare_paths("a", "b", ab)
    compare_paths("b", "a", ba)
    ae = [
        ["a", "b", "d", "g", "e"],
        ["a", "c", "e"],
    ]
    ea = [
        ["e", "c", "a"],
        ["e", "g", "d", "b", "a"],
    ]
    compare_paths("a", "e", ae)
    compare_paths("e", "a", ea)

    bg = [
        ["b", "a", "c", "e", "g"],
        ["b", "d", "g"],
    ]
    gb = [
        ["g", "d", "b"],
        ["g", "e", "c", "a", "b"],
    ]
    compare_paths("b", "g", bg)
    compare_paths("g", "b", gb)

    gf = [
        ["g", "d", "f"],
        ["g", "e", "c", "a", "b", "d", "f"],
    ]
    fg = [
        ["f", "d", "b", "a", "c", "e", "g"],
        ["f", "d", "g"],
    ]
    compare_paths("g", "f", gf)
    compare_paths("f", "g", fg)

    ef = [
        ["e", "c", "a", "b", "d", "f"],
        ["e", "g", "d", "f"],
    ]
    fe = [
        ["f", "d", "b", "a", "c", "e"],
        ["f", "d", "g", "e"],
    ]
    compare_paths("e", "f", ef)
    compare_paths("f", "e", fe)


def test_basic_alt_all_simple_paths(
    alt_neighbours: NodeMap, alt_reachable: NodeMap
) -> None:
    """Test all_simple_paths()."""

    # Helper function for comparing
    def compare_paths(
        source: Node, target: Node, true_paths: list[NodeSequence]
    ) -> None:
        p = all_simple_paths(source, target, alt_neighbours, alt_reachable)
        assert len(p) == len(true_paths), (
            f"Incorrect number of paths for {source}, {target}"
        )
        assert all(path in true_paths for path in p), (
            f"Incorrect paths found for {source}, {target}"
        )

    # Should return empty list if no path exists
    assert all_simple_paths("h", "e", alt_neighbours, alt_reachable) == []

    compare_paths(
        "a",
        "b",
        [
            ["a", "b"],
            ["a", "c", "b"],
            ["a", "c", "d", "b"],
            ["a", "e", "f", "g", "d", "b"],
            ["a", "e", "f", "g", "d", "c", "b"],
        ],
    )
    compare_paths(
        "b",
        "a",
        [
            ["b", "a"],
            ["b", "c", "a"],
            ["b", "d", "c", "a"],
            ["b", "d", "g", "f", "e", "a"],
            ["b", "c", "d", "g", "f", "e", "a"],
        ],
    )

    compare_paths(
        "a",
        "d",
        [
            ["a", "b", "d"],
            ["a", "c", "d"],
            ["a", "c", "b", "d"],
            ["a", "b", "c", "d"],
            ["a", "e", "f", "g", "d"],
        ],
    )

    compare_paths("i", "h", [["i", "h"]])
    compare_paths("h", "i", [["h", "i"]])


@dataclass
class ComplexGraph:
    """Container for complex graph strategy."""

    nodes: Sized[Node]
    edges: Edges
    graph: nx.Graph
    neighbours: NodeMap
    reachable: NodeMap


# Splitting this off so generation doesn't keep repeating the same calculations
@st.composite
def complex_graph(draw: st.DrawFn) -> ComplexGraph:
    """Complex graph for randomised testing against networkx."""
    nodes = sorted(["D", "C", "L", "S", "T", "N", "O", "A", "M", "I", "G"])
    edges = {
        ("O", "D"),
        ("D", "N"),
        ("A", "I"),
        ("T", "S"),
        ("T", "O"),
        ("T", "I"),
        ("I", "N"),
        ("A", "M"),
        ("S", "I"),
        ("C", "I"),
        ("G", "D"),
        ("C", "L"),
        ("O", "I"),
        ("O", "M"),
        ("L", "I"),
        ("G", "T"),
        ("D", "S"),
        ("L", "M"),
        ("C", "A"),
        ("M", "D"),
        ("T", "D"),
        ("C", "G"),
        ("G", "I"),
        ("G", "O"),
        ("G", "M"),
    }
    # Setting up NetworkX stuff
    g = nx.Graph()
    g.add_edges_from(edges)

    # Setting up stuff for all_simple_paths
    neighbours = get_neighbours(nodes, edges)
    reachable = get_reachable(neighbours)

    return draw(st.just(ComplexGraph(nodes, edges, g, neighbours, reachable)))


@given(st.data(), complex_graph())
def test_all_simple_paths(data: st.DataObject, complex_graph: ComplexGraph) -> None:
    """Test whether all_simple_paths returns the same as nx.all_simple_paths."""
    # randomly selecting a pair of nodes from the
    source, target = data.draw(
        st.lists(
            st.sampled_from(complex_graph.nodes), min_size=2, max_size=2, unique=True
        )
    )

    nx_paths = list(nx.all_simple_paths(complex_graph.graph, source, target))
    paths = all_simple_paths(
        source,
        target,
        complex_graph.neighbours,
        complex_graph.reachable,
    )

    assert len(paths) == len(nx_paths), (
        f"Didn't find the same number of paths for {source} to {target}"
    )
    assert all(path in nx_paths for path in paths), (
        f"all paths in nx_paths for {source} to {target}"
    )
    assert all(path in paths for path in nx_paths), (
        f"all nx_paths in paths for {source} to {target}"
    )


# ---- Testing helper functions for d-separators


def test_find_existing_paths(edges: Edges) -> None:
    """Test find_existing_paths()."""
    existing = [["a", "b", "d"], ["d", "f"], ["c", "e", "g"]]
    # should filter out paths that are impossible according to nodes
    non_existing = [["a", "d", "g"], ["e", "f", "d"], ["c", "f", "g"]]
    assert find_existing_paths(edges, existing + non_existing) == existing


def test_find_open_paths(edges: Edges) -> None:
    """Test find_open_paths()."""
    open_paths = [["a", "b", "d"], ["g", "f", "d"]]
    closed_paths = [["d", "g", "e"]]
    assert find_open_paths(edges, open_paths + closed_paths) == open_paths


def test_find_colliders() -> None:
    """Test find_colliders()."""
    # a -> b <- c -> d <- e <- f
    edges = [("a", "b"), ("c", "b"), ("c", "d"), ("e", "d"), ("f", "e")]
    assert find_colliders(edges, ["a", "b", "c"]) == {"b"}  # ty:ignore[invalid-argument-type]
    assert find_colliders(edges, ["a", "b", "c", "d"]) == {"b"}  # ty:ignore[invalid-argument-type]
    assert find_colliders(edges, ["a", "b", "c", "d", "e"]) == {"b", "d"}  # ty:ignore[invalid-argument-type]
    assert find_colliders(edges, ["a", "b", "c", "d", "e", "f"]) == {"b", "d"}  # ty:ignore[invalid-argument-type]
    assert find_colliders(edges, ["d", "e", "f"]) == set()  # ty:ignore[invalid-argument-type]


def test_path_is_closed() -> None:
    """Test path_is_closed()."""
    # a <- b -> c <- d -> e
    # c -> f
    path = ["a", "b", "c", "d", "e"]
    colliders = {"c"}
    descendants = {"c": ["f"]}

    assert path_is_closed(path, set(), colliders, descendants)
    assert path_is_closed(path, {"b"}, colliders, descendants)
    assert path_is_closed(path, {"d"}, colliders, descendants)
    assert path_is_closed(path, {"b", "d"}, colliders, descendants)
    assert not path_is_closed(path, {"c"}, colliders, descendants)
    assert not path_is_closed(path, {"f"}, colliders, descendants)
    assert not path_is_closed(path, {"c", "f"}, colliders, descendants)

    # open path should not be closed by empty set
    assert not path_is_closed(["a", "b", "c"], set(), set(), {})


def test_find_separators(edges: Edges, descendants: NodeMap) -> None:
    """Test find_separators()."""
    # Testing if no paths leads to all permutations of available
    all_combinations = [
        [],
        ["a"],
        ["b"],
        ["c"],
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
        ["a", "b", "c"],
    ]
    test_sep = find_separators({"a", "b", "c"}, edges, [], descendants)
    assert all(c in test_sep for c in all_combinations), (
        "Empty paths should lead to all combinations of {a, b, c}"
    )
    # Testing if empty list is return when one of the paths is an edge
    assert (
        find_separators(
            {"a", "b", "c"}, edges, [["a", "c", "e"], ["a", "b"]], descendants
        )
        == []
    ), "should be no d-separators if any path is an edge"

    # testing d-separating sets
    assert find_separators(
        {"b", "c"}, edges, [["a", "c", "e"], ["a", "b", "d"]], descendants
    ) == [["b", "c"]]

    true_set = [["a"], ["b"], ["a", "b"]]
    test_set = find_separators(
        {"a", "b"}, edges, [["c", "e", "g", "d"], ["c", "a", "b", "d"]], descendants
    )
    assert len(true_set) == len(test_set), "not enough d-separators"
    assert all(s in test_set for s in true_set), "not all true sets appear"


def test_find_minimal_separators() -> None:
    """Test find_minimal_separators()."""
    separators = [["a"], ["a", "b"], ["c", "d"], ["c", "d", "e"]]
    minimal_separators = [["a"], ["c", "d"]]
    assert find_minimal_separators(separators) == minimal_separators
    assert find_minimal_separators([[], ["a"]]) == [[]]


def test_is_d_separator_raises(
    edges: Edges,
    neighbours: NodeMap,
    descendants: NodeMap,
    reachable: NodeMap,
) -> None:
    """Test if is_d_separator() raises the correct exception."""
    with pytest.raises(NoDisjointSetsError):
        is_d_separator(
            {"a"}, {"a"}, {"b"}, edges, neighbours, descendants, reachable
        )
    with pytest.raises(NoDisjointSetsError):
        is_d_separator(
            {"a"}, {"a"}, {"a"}, edges, neighbours, descendants, reachable
        )
    with pytest.raises(NoDisjointSetsError):
        is_d_separator(
            {"b"}, {"a"}, {"a"}, edges, neighbours, descendants, reachable
        )
    with pytest.raises(NoDisjointSetsError):
        is_d_separator(
            {"b"}, {"a"}, {"b"}, edges, neighbours, descendants, reachable
        )


def test_is_d_separator(
    edges: Edges,
    neighbours: NodeMap,
    descendants: NodeMap,
    reachable: NodeMap,
) -> None:
    """Test is_d_separator()."""
    # if X and Y contain neighbours, Z is not a d-separator
    assert not is_d_separator(
        {"a"}, {"c"}, {"b"}, edges, neighbours, descendants, reachable
    )
    assert not is_d_separator(
        {"a"}, {"b", "c"}, {"f"}, edges, neighbours, descendants, reachable
    )
    # if Z doesn't contain any nodes on the paths from X to Y, no d-separator.
    assert not is_d_separator(
        {"a", "b"},
        {"d", "e"},
        {"f"},
        edges,
        neighbours,
        descendants,
        reachable,
    )
    # if Z contains a collider, cannot be a d-separator
    # (removing path through a for this test)
    assert not is_d_separator(
        {"c"},
        {"b"},
        {"g"},
        mutilate(edges, [], ["a"]),
        neighbours,
        descendants,
        reachable,
    )
    # if Z doesn't close all paths, cannot be a d-separtor
    # (path through a remains open)
    assert not is_d_separator(
        {"b"}, {"c"}, {"e"}, edges, neighbours, descendants, reachable
    )

    # e and a should be a d-separator between b and c
    assert is_d_separator(
        {"b"}, {"c"}, {"a", "e"}, edges, neighbours, descendants, reachable
    )

    # if there are no paths, any Z should be a d-separator
    assert is_d_separator(
        {"a"},
        {"g"},
        {"d", "e"},
        mutilate(edges, [], ["b", "c"]),
        neighbours,
        descendants,
        reachable,
    )


def test_find_d_separators_raises(
    edges: Edges, neighbours: NodeMap, descendants: NodeMap, reachable: NodeMap
) -> None:
    """Test if find_d_separators raises the correct exception."""
    with pytest.raises(NoDisjointSetsError):
        find_d_separators(
            {"a", "b"}, {"b", "c"}, edges, neighbours, descendants, reachable, set()
        )


def test_find_d_separators(
    edges: Edges, neighbours: NodeMap, descendants: NodeMap, reachable: NodeMap
) -> None:
    """Test find_d_separators()."""
    correct = [["b", "c"], ["b", "e"], ["c", "d"], ["d", "e"]]
    x = {"a"}
    y = {"g"}
    d_sep = find_d_separators(x, y, edges, neighbours, descendants, reachable, set())
    assert all(c in d_sep.separators for c in correct)
    assert d_sep.separators != d_sep.minimal
    assert isinstance(repr(d_sep), str)
    assert all(
        is_d_separator(x, y, set(z), edges, neighbours, descendants, reachable)
        for z in d_sep.separators
    )

    x = {"a"}
    y = {"f", "g"}
    d_sep = find_d_separators(x, y, edges, neighbours, descendants, reachable, set())
    assert all(c in d_sep.separators for c in correct)
    assert d_sep.separators != d_sep.minimal
    assert isinstance(repr(d_sep), str)
    assert all(
        is_d_separator(x, y, set(z), edges, neighbours, descendants, reachable)
        for z in d_sep.separators
    )

    # removing nodes through unobserved
    correct = [["c"]]
    x = {"a"}
    y = {"e"}
    d_sep = find_d_separators(
        x, y, edges, neighbours, descendants, reachable, {"b", "d", "g"}
    )
    assert d_sep.minimal == correct
    assert d_sep.minimal == d_sep.separators
    assert isinstance(repr(d_sep), str)
    assert all(
        is_d_separator(x, y, set(z), edges, neighbours, descendants, reachable)
        for z in d_sep.separators
    )

    # removing edges through mutilation
    e = mutilate(edges, over=["b"], under=[])
    x = {"a"}
    y = {"e"}
    d_sep = find_d_separators(x, y, e, neighbours, descendants, reachable, set())
    assert d_sep.minimal == correct
    assert d_sep.minimal == d_sep.separators
    assert isinstance(repr(d_sep), str)
    assert all(
        is_d_separator(x, y, set(z), e, neighbours, descendants, reachable)
        for z in d_sep.separators
    )

    # empty if a pair is an edge
    d_sep = find_d_separators(
        {"a"}, {"b", "c"}, edges, neighbours, descendants, reachable, set()
    )
    assert d_sep.minimal == []
    assert d_sep.separators == []
    assert isinstance(repr(d_sep), str)
    assert all(
        is_d_separator(x, y, set(z), edges, neighbours, descendants, reachable)
        for z in d_sep.separators
    )


# --- Tests for backdoor criterion


@dataclass
class BC:
    """Container for backdoor criterion fixtures."""

    edges: Edges
    paths: list[NodeSequence]
    descendants: NodeMap


@pytest.fixture
def pipe() -> BC:
    """Pipe and paths from x to y."""
    edges = {("x", "y"), ("x", "z"), ("z", "y")}
    paths = [["x", "y"], ["x", "z", "y"]]
    descendants = {
        "x": ["y", "z"],
        "y": [],
        "z": ["y"],
    }
    return BC(edges, paths, descendants)


@pytest.fixture
def fork() -> BC:
    """Fork and paths from x to y."""
    edges = {("x", "y"), ("z", "x"), ("z", "y")}
    paths = [["x", "y"], ["x", "z", "y"]]
    descendants = {
        "x": ["y"],
        "y": [],
        "z": ["x", "y"],
    }
    return BC(edges, paths, descendants)


@pytest.fixture
def collider() -> BC:
    """Fork and paths from x to y."""
    edges = {("x", "y"), ("x", "z"), ("y", "z")}
    paths = [["x", "y"], ["x", "z", "y"]]
    descendants = {
        "x": ["y", "z"],
        "y": ["z"],
        "z": [],
    }
    return BC(edges, paths, descendants)


def test_backdoor_criterion(pipe: BC, fork: BC, collider: BC) -> None:
    """Test backdoor_criterion()."""
    # Pipe has no backdoor paths
    pipe_bc = backdoor_criterion(
        "x", "y", pipe.edges, pipe.paths, set(), pipe.descendants
    )
    assert pipe_bc.backdoor_paths is None, "Pipe has no backdoor paths"
    assert pipe_bc.open_paths is None, "Pipe has no open paths"
    assert pipe_bc.adjustment_sets is None, "Pipe requires no adjustment"
    assert isinstance(repr(pipe_bc), str)

    # Fork has a backdoor path, and needs adjustment
    fork_bc = backdoor_criterion(
        "x", "y", fork.edges, fork.paths, set(), fork.descendants
    )
    assert fork_bc.backdoor_paths == [["x", "z", "y"]], "Fork one backdoor path"
    assert fork_bc.open_paths == [["x", "z", "y"]], "Fork one open path"
    assert fork_bc.adjustment_sets == [["z"]], "Fork requires adjustment"
    assert isinstance(repr(fork_bc), str)

    # Fork is not adjustable if z is unobserved
    fork_u_desc = {"x": ["y"], "y": [], "z": []}
    fork_u_bc = backdoor_criterion(
        "x", "y", fork.edges, fork.paths, {"z"}, fork_u_desc
    )
    assert fork_u_bc.backdoor_paths == [["x", "z", "y"]], "Fork one backdoor path"
    assert fork_u_bc.open_paths == [["x", "z", "y"]], "Fork one open path"
    assert fork_u_bc.adjustment_sets is None, "Fork u(z) has no adjustment"
    assert isinstance(repr(fork_u_bc), str)

    # Collider has a backdoor path, but doesn't require adjustment
    collider_bc = backdoor_criterion(
        "x", "y", collider.edges, collider.paths, set(), collider.descendants
    )
    assert collider_bc.backdoor_paths == [["x", "z", "y"]], (
        "Collider one backdoor path"
    )
    assert collider_bc.open_paths is None, "Pipe has no open paths"
    assert collider_bc.adjustment_sets is None, "Pipe requires no adjustment"
    assert isinstance(repr(collider_bc), str)


# ---- Test for conditional independencies


def test_conditional_independencies() -> None:
    """Test conditional_independencies()."""
    nodes = list("xwyz")
    edges = {("x", "y"), ("y", "z"), ("w", "z")}

    sortings = get_topological_sortings(nodes, edges)

    neighbours = get_neighbours(nodes, edges)
    reachable = get_reachable(neighbours)
    descendants = get_descendants(edges, sortings.topological_generations)

    ci = conditional_independencies(
        nodes,
        edges,
        set(),
        set(),
        neighbours,
        descendants,
        reachable,
        testable_only=True,
    )
    all_indep = ["x ⫫ w", "x ⫫ z | y", "w ⫫ y"]
    assert ci.testable == all_indep, "incorrect testable independencies"
    assert ci.untestable == [], "should be no untestable independencies"

    ci = conditional_independencies(
        nodes,
        edges,
        set(),
        {"y"},
        neighbours,
        descendants,
        reachable,
        testable_only=False,
    )
    assert ci.testable == ["x ⫫ w"], "incorrect testable independencies"
    assert ci.untestable == ["x ⫫ z | (y)", "w ⫫ (y)"], (
        "incorrect untestable independencies"
    )

    ci = conditional_independencies(
        nodes,
        mutilate(edges, [], ["x"]),
        {"y"},
        {"z"},
        neighbours,
        descendants,
        reachable,
        testable_only=False,
    )
    assert ci.testable == ["x ⫫ w"], "incorrect testable independencies"
    assert ci.untestable == ["x ⫫ (z)"], "incorrect untestable independencies"


def test_no_conditional_independencies(pipe: BC) -> None:
    """Test if rendering still works if there are no conditional independencies."""
    nodes = edges_to_nodes(pipe.edges)
    sortings = get_topological_sortings(nodes, pipe.edges)
    neighbours = get_neighbours(nodes, pipe.edges)
    reachable = get_reachable(neighbours)
    descendants = get_descendants(pipe.edges, sortings.topological_generations)

    ci = conditional_independencies(
        sortings.topological_sort,
        pipe.edges,
        set(),
        set(),
        neighbours,
        descendants,
        reachable,
        testable_only=False,
    )

    assert isinstance(ci.render(""), str)


# ---- Test for DirectedAcyclicGraph


def test_directed_acyclic_graph_raises(edges: Edges) -> None:
    """Test if DirectedAcyclicGraph raises the correct exceptions."""
    with pytest.raises(MissingNodeError):
        DirectedAcyclicGraph([], edges)


def test_directed_acyclic_graph(nodes: set[Node], edges: Edges) -> None:
    """Test DirectedAcyclicGraph()."""
    dag = DirectedAcyclicGraph(nodes, edges)
    dag.backdoor_criterion("a", "g", [], set())
    dag.backdoor_criterion("a", "g", do=["c", "b"], unobserved=set())

    dag.conditional_independencies(None, None, set(), set())
    dag.conditional_independencies(None, None, set(), set(), show="untestable")
    dag.conditional_independencies(None, None, set(), set(), show="both")

    dag.is_d_separator({"a"}, {"g"}, set(), None, None)
