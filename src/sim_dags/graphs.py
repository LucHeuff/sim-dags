from collections import deque
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from typing import Literal

from more_itertools import sliding_window

from sim_dags.exceptions import (
    MissingNodeError,
    NoDisjointSetsError,
    NotADAGError,
)

type Edge = tuple[str, str]
type Edges = set[Edge]
type Node = str
type NodeMap = dict[Node, list[Node]]
type NodeSequence = list[Node]

type CIOptions = Literal["testable", "untestable", "both"]


def edges_to_nodes(edges: Iterable[Edge]) -> set[Node]:
    """Convert an edge list to a list of nodes."""
    return {node for edge in edges for node in edge}


def mutilate(edges: Edges, over: NodeSequence, under: NodeSequence) -> Edges:
    """Mutilate a list of edges.

    Removes all edges pointing into nodes in over,
    and all edges pointing out of under.
    """
    over_edges = {(u, v) for u, v in edges if v in over}
    under_edges = {(u, v) for u, v in edges if u in under}
    return edges - over_edges - under_edges


def path_exists(path: NodeSequence, edges: Edges) -> bool:
    """Check if the provided path exists in edges."""
    path_edges = sliding_window(path, 2)
    return all(edge in edges for edge in path_edges)


def undirected_path_exists(path: NodeSequence, edges: Edges) -> bool:
    """Check if the provided path exists in edges."""
    path_edges = sliding_window(path, 2)
    return all(edge in edges or edge[::-1] in edges for edge in path_edges)


def path_has_collider(path: NodeSequence, edges: Edges) -> bool:
    """Check if this path contains a collider."""
    if len(path) < 3:  # noqa: PLR2004
        return False
    for u, c, v in sliding_window(path, 3):
        if (u, c) in edges and (v, c) in edges:
            return True
    return False


def is_collider(node: Node, path: NodeSequence, edges: Edges) -> bool:
    """Check if this node is a collider on this path."""
    if node not in path:
        return False

    if len(path) < 3:  # noqa: PLR2004
        return False  # can't be a collider if path only has 2 nodes.
    node_index = path.index(node)
    if node_index == 0 or node_index == len(path) - 1:
        return False
    u = path[node_index - 1]
    v = path[node_index + 1]

    return (u, node) in edges and (v, node) in edges


def get_parents(nodes: set[Node], edges: Edges) -> NodeMap:
    """Get a mapping from nodes to their direct parents."""
    return {node: sorted(u for u, v in edges if v == node) for node in sorted(nodes)}


def get_neighbours(nodes: Sized[Node], edges: Edges) -> NodeMap:
    """Get a mapping from nodes to their undirected neighbours."""
    neighbours = {node: set() for node in nodes}
    for node in nodes:
        for u, v in edges:
            # if one part of the edges is our node, add the other one to the mapping.
            if u == node:
                neighbours[node] |= {v}
            elif v == node:
                neighbours[node] |= {u}

    return {node: sorted(neighbours[node]) for node in nodes}


@dataclass
class TopologicalSorting:
    """Contains both topological sort and topological generations."""

    topological_sort: NodeSequence
    topological_generations: list[NodeSequence]


def get_topological_sortings(nodes: Sized[Node], edges: Edges) -> TopologicalSorting:
    """Attempt to sort nodes into topological generations. Returns an error if not a DAG."""  # noqa: E501
    # inspired by Kahn's Algorithm: https://en.wikipedia.org/wiki/Topological_sorting
    in_degree = {node: sum([node == end for _, end in edges]) for node in nodes}
    children = {
        node: [end for start, end in edges if start == node] for node in nodes
    }

    node_gen = {node: 0 for node, in_ in in_degree.items() if in_ == 0}

    queue = deque(list(node_gen))
    order = []

    # convention, edge (u, v) means u -> v
    while queue:
        # remove leftmost item from the queue and find its neighbours
        u = queue.popleft()
        order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            node_gen[v] = node_gen[u] + 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(nodes):
        msg = "Provided graph is not a DAG."
        raise NotADAGError(msg)

    max_gen = max(node_gen.values())

    topological_generations = [
        sorted(node for node, gen in node_gen.items() if gen == g)
        for g in range(max_gen + 1)
    ]

    topological_sort = [
        node for nodes in topological_generations for node in sorted(nodes)
    ]

    return TopologicalSorting(topological_sort, topological_generations)


def get_descendants(
    edges: Edges, topological_generations: list[NodeSequence]
) -> NodeMap:
    """Create a mapping from every node to their descendants."""
    desc = {}

    def find_descendants(node: Node) -> set[Node]:
        """Recursively retrieve nodes from descendants map."""
        # if this node doesn't appear yet, find its children
        children = desc.get(node, {v for u, v in edges if u == node})

        return children | {
            node for child in children for node in find_descendants(child)
        }

    # starting with reversed topological generations, since last generation
    # doesnn't have any descendants
    for i, layer in enumerate(topological_generations[::-1]):
        if i == 0:
            desc.update({node: set() for node in layer})
        else:
            for node in layer:
                desc[node] = find_descendants(node)

    return {node: sorted(d) for node, d in desc.items()}


def get_reachable(neighbours: NodeMap) -> NodeMap:
    """Create mapping from every node to every other node it can reach."""

    def find_reachable(node: Node, seen: set[Node]) -> set[Node]:
        """Recursively search across neighbours for reachable nodes."""
        seen = seen | {node}
        neigh = set(neighbours[node]) - seen
        return neigh | {r for n in neigh for r in find_reachable(n, seen)}

    return {node: sorted(find_reachable(node, set())) for node in neighbours}


def all_simple_paths(
    source: Node,
    target: Node,
    neighbours: NodeMap,
    reachable: NodeMap,
) -> list[NodeSequence]:
    """Generate all simple paths from a list of edges."""

    def find_paths(node: Node, path: NodeSequence) -> list[NodeSequence]:
        """Recursively search along neighbours for a path to target."""
        path = [*path, node]

        if node == target:
            return [path]

        paths = []
        for neighbour in neighbours[node]:
            if neighbour in path:
                continue
            paths.extend(find_paths(neighbour, path))

        return paths

    # If the source and target cannot reach each other,
    # then return the empty list immediately
    if target not in reachable[source]:
        return []

    # Otherwise, search for unknown paths.
    return find_paths(source, [])


# --- Helper functions for finding d-separators.


def find_existing_paths(
    edges: Edges, paths: list[NodeSequence]
) -> list[NodeSequence]:
    """Get a list of paths that exist according to edges."""
    return [path for path in paths if undirected_path_exists(path, edges)]


def find_open_paths(edges: Edges, paths: list[NodeSequence]) -> list[NodeSequence]:
    """Get a list of open paths based on given edges."""
    return [path for path in paths if not path_has_collider(path, edges)]


def find_colliders(edges: Edges, path: NodeSequence) -> set[Node]:
    """Find colliders on a path."""
    return {node for node in path if is_collider(node, path, edges)}


def path_is_closed(
    path: NodeSequence,
    z: set[Node],
    colliders: set[Node],
    descendants: NodeMap,
) -> bool:
    """Check if the path is closed by Z.

    A path is closed (or equivalently, Z is a d-separator) if:
    - path contains at least one arrow emitting node that is in Z
    OR
    - path contains a collider that is not in Z and doesn't have any descendants in Z

    Args:
        path: a sequence of nodes forming a valid path
        z: Z set to be tested for d-separation
        colliders: set of colliders on this path
        descendants: mapping of nodes to their descendants

    Returns:
        boolean whether the path is closed

    """
    emitting_nodes = set(path) - colliders

    # Checking the first condition: arrow emitting node in Z
    if z & emitting_nodes:
        return True

    # Checking the second condition:
    # any collider that isn't in z and also doesn't have descendants in z
    return any((c not in z and not z & set(descendants[c])) for c in colliders)


def find_separators(
    available: set[Node],
    edges: Edges,
    paths: list[NodeSequence],
    descendants: NodeMap,
) -> list[NodeSequence]:
    """Find d-separating sets: subsets of nodes that close all paths."""
    # if there are no paths, return the empty set
    if not bool(paths):
        return [[]]

    # if any of the paths is a direct edge, then there are no d-eseparators
    if any(len(path) == 2 for path in paths):  # noqa: PLR2004
        return []

    # Storing a map of colliders for each path, so I only need to calculate once
    collider_map = {tuple(path): find_colliders(edges, path) for path in paths}

    d_sep = []
    # Finding a list of nodes that never appear as a collider on any path
    # These can trivialy be added to a valid d-separating set, which speeds
    # up the search a bit

    # iterating over all possible combinations of nodes
    for i in range(len(available) + 1):
        # i = 0 also tests the empty set
        for c in combinations(available, i):
            z = set(c)
            # Can skip this combination if it already appears in d-separators
            if all(
                path_is_closed(path, z, collider_map[tuple(path)], descendants)
                for path in paths
            ):
                d_sep.append(sorted(z))

    # converting sets to sorted lists for more consistent output
    return d_sep


def find_minimal_separators(d_sep: list[NodeSequence]) -> list[NodeSequence]:
    """Find all d-separating sets that are not subsets of another d-separating set."""  # noqa: E501
    # A d-separating set is minimal if there are not other d-separating sets
    # that are a subset of this set.
    # Since any set is a subset of itself, ignore comparing sets to themselves.

    # If the empty set appears in d_sep, there is no smaller possible set and
    # we can return it immediately
    if [] in d_sep:
        return [[]]
    return [
        z for z in d_sep if not any(set(w).issubset(set(z)) for w in d_sep if z != w)
    ]


def is_d_separator(
    x: set[Node],
    y: set[Node],
    z: set[Node],
    edges: Edges,
    neighbours: NodeMap,
    descendants: NodeMap,
    reachable: NodeMap,
) -> bool:
    """Check if z d-separates x and y.

    a set Z d-separates a set X and a set Y if for every x ∈ X and every y ∈ Y,
    all paths between x and y are blocked by Z.

    Equivalently, there should be no d-connected pair x ∈ X, y ∈ Y given Z.
    """
    if bool(x & y & z):
        msg = "X, Y and Z are not disjoint."
        raise NoDisjointSetsError(msg)
    if bool(x & y):
        msg = "X and Y are not disjoint."
        raise NoDisjointSetsError(msg)
    if bool(x & z):
        msg = "X and Z are not disjoint."
        raise NoDisjointSetsError(msg)
    if bool(y & z):
        msg = "Y and Z are not disjoint."
        raise NoDisjointSetsError(msg)

    pairs = list(product(x, y))

    # if any x ∈ X is a neighbour of any y ∈ Y, then Z cannot be a d-separator.
    if any(pair in edges or pair[::-1] in edges for pair in pairs):
        return False

    # Evaluating if Z d-separates for every pair
    for u, v in pairs:
        paths = [
            path
            for path in all_simple_paths(u, v, neighbours, reachable)
            if undirected_path_exists(path, edges)
        ]
        # If there are no paths between these two nodes, then any Z is a d-separator
        if len(paths) == 0:
            continue

        if not all(
            path_is_closed(path, z, find_colliders(edges, path), descendants)
            for path in paths
        ):
            return False

    return True


@dataclass(slots=True, frozen=True)
class DSeparators:
    """Container for d-separators."""

    minimal: list[NodeSequence]
    separators: list[NodeSequence]

    def render_set(self, s: NodeSequence) -> str:
        """Render a separating set."""
        return f"{{{', '.join(s)}}}"

    def render_sets(self, s: list[NodeSequence]) -> str:
        """Render a list of sets."""
        return "\n\t" + f"{'\n\t'.join(self.render_set(s_) for s_ in s)}"

    def __repr__(self) -> str:
        """List d-separators, if any."""
        if len(self.minimal) == 0:
            return "No d-separating sets found."
        if len(self.minimal) == len(self.separators):
            return "D-separating sets:" + self.render_sets(self.minimal)
        min_ = "Minimal d-separating sets:" + self.render_sets(self.minimal)
        all_ = "All d-separating sets:" + self.render_sets(self.separators)

        return f"D-separating sets:\n{min_}\n{all_}"


def find_d_separators(
    x: set[Node],
    y: set[Node],
    edges: Edges,
    neighbours: NodeMap,
    descendants: NodeMap,
    reachable: NodeMap,
    unobserved: set[Node],
) -> DSeparators:
    """Find d-separating sets between X and Y."""
    if bool(x & y):
        msg = "X and Y are not disjoint."
        raise NoDisjointSetsError(msg)

    pairs = list(product(x, y))

    # If any of the pairs appears in the edges, then no d-separators are possible
    if any(pair in edges for pair in pairs):
        return DSeparators([], [])

    # keeping a list of set of d-separating sets.
    all_separators = []

    for u, v in pairs:
        all_paths = all_simple_paths(u, v, neighbours, reachable)
        paths = find_existing_paths(edges, all_paths)
        available = {
            node
            for path in paths
            for node in path
            if node not in unobserved | {u, v}
        }
        separators = {
            tuple(sep)
            for sep in find_separators(available, edges, paths, descendants)
        }
        all_separators.append(separators)

    # We now have a list of sets of d-separators for each pair
    # Now we need to find out which d-separators all of these sets have in common.

    d_sep = [list(sep) for sep in reduce(lambda u, v: u & v, all_separators)]

    minimal = find_minimal_separators(d_sep)

    return DSeparators(minimal, d_sep)


# --- Backdoor criterion
@dataclass(slots=True, frozen=True)
class BackdoorCriterion:
    """Container for partial outputs of backdoor criterion."""

    edges: set[Edge]
    backdoor_paths: list[NodeSequence] | None = None
    open_paths: list[NodeSequence] | None = None
    adjustment_sets: list[NodeSequence] | None = None
    minimal_adjustment_sets: list[NodeSequence] | None = None

    def render_path(self, path: NodeSequence) -> str:
        """Convert path into readable format, referring to the DAG."""
        r = f"{path[0]}"
        for u, v in sliding_window(path, 2):
            if (u, v) in self.edges:
                r += " ->"
            elif (v, u) in self.edges:
                r += " <-"
            r += f" {v}"

        return r.strip()

    def render_set(self, adjustment_sets: list[NodeSequence]) -> str:
        """Render a list of adjustment sets into a legigble format."""
        adj = [f"{{{', '.join(set_)}}}" for set_ in adjustment_sets]
        return "\n  ".join(adj)

    def __repr__(self) -> str:
        """Legible string with backdoor criterion output."""
        if self.backdoor_paths is None:
            return "No backdoor paths found, no adjustment is necessary."
        if self.open_paths is None:
            return "No open backdoor paths found, no adjustment is necessary."

        # Formatting the backdoor paths
        rendered_open = [self.render_path(path) for path in self.open_paths]
        msg = f"Found {len(self.open_paths)} open paths:\n  {'\n  '.join(rendered_open)}\n"  # noqa: E501

        if self.adjustment_sets is None or self.minimal_adjustment_sets is None:
            msg += "No adjustment sets found."
        else:
            min_adj = self.render_set(self.minimal_adjustment_sets)
            msg += f"Minimal adjustment sets: \n  {min_adj}"
            if len(self.adjustment_sets) > len(self.minimal_adjustment_sets):
                adj = self.render_set(self.adjustment_sets)
                msg += f"\nAll available adjustment sets:\n  {adj}"

        return msg


def backdoor_criterion(
    exposure: Node,
    outcome: Node,
    edges: set[Edge],
    simple_paths: list[NodeSequence],
    unobserved: set[Node],
    descendants: NodeMap,
) -> BackdoorCriterion:
    """Perform backdoor criterion for a given list of paths."""
    # Finding existing paths based on edges.
    # These might be different from simple_paths due to mutilation.

    # --- Finding backdoor paths
    # if the path appears in the DAG, it is not a backdoor path.
    backdoor_paths = [path for path in simple_paths if not path_exists(path, edges)]

    # if there are no backdoor paths, return immediately.
    if not bool(backdoor_paths):
        return BackdoorCriterion(edges)

    # --- Finding open paths
    # A backdoor path is closed if there is a collider on it
    open_paths = find_open_paths(edges, backdoor_paths)

    # if there are no open paths, return immediately
    if not bool(open_paths):
        return BackdoorCriterion(edges, backdoor_paths)

    # --- Finding adjustment sets.
    # Candidates are observed nodes that are not the exposure or the outcome.

    available = {
        node
        for path in backdoor_paths
        for node in path
        if node not in unobserved | {exposure, outcome}
    }

    # adjustment sets consist of combinations of nodes that close all open paths
    adjustment_sets = find_separators(available, edges, backdoor_paths, descendants)

    # if no adjustment sets were found, return backdoor and open paths
    if not bool(adjustment_sets):
        return BackdoorCriterion(edges, backdoor_paths, open_paths)

    minimal_adjustment_sets = find_minimal_separators(adjustment_sets)

    # otherwise, return everything
    return BackdoorCriterion(
        edges, backdoor_paths, open_paths, adjustment_sets, minimal_adjustment_sets
    )


# --- Conditional independencies
@dataclass(slots=True, frozen=True)
class ConditionalIndependecies:
    """Container for partial outputs of conditional independencies."""

    testable: list[str]
    untestable: list[str]

    def render_list(self, list_: list[str], title: str, do_msg: str) -> str:
        """Render a list into a string."""
        if len(list_) == 0:
            return f"No {title} conditional independencies{do_msg}."
        return (
            f"{title.capitalize()} independencies{do_msg}:\n  {'\n  '.join(list_)}"
        )

    def render_testable(self, do_msg: str) -> str:
        """Render testable independencies."""
        return self.render_list(self.testable, "testable", do_msg)

    def render_untestable(self, do_msg: str) -> str:
        """Render untestable independencies."""
        return self.render_list(self.untestable, "untestable", do_msg)

    def render(self, do_msg: str) -> str:
        """Render all conditional independencies."""
        if len(self.testable) == 0 and len(self.untestable) == 0:
            return (
                f"The graph does not imply any conditional independencies{do_msg}."
            )
        return f"{self.render_testable(do_msg)}\n{self.render_untestable(do_msg)}"


def conditional_independencies(
    topological_sort: NodeSequence,
    edges: Edges,
    ignore: set[Node],
    unobserved: set[Node],
    neighbours: NodeMap,
    descendants: NodeMap,
    reachable: NodeMap,
    *,
    testable_only: bool,
) -> ConditionalIndependecies:
    """Find all conditional independencies implied by this graph."""
    # Depending on the show option, not all nodes are relevant
    testable = []
    untestable = []

    # functions for rendering unobserved variables surrounded by brackets
    def u(var: str) -> str:
        return f"({var})" if var in unobserved else var

    def cond(left: str, right: str) -> str:
        return f"{u(left)} ⫫ {u(right)}"

    def r(left: str, right: str, d_sep: list[str]) -> str:
        if not bool(d_sep):
            return cond(left, right)

        return f"{cond(left, right)} | {','.join(list(map(u, d_sep)))}"

    def update(left: str, right: str, d_sep: list[str]) -> None:
        if bool({left, right, *d_sep} & unobserved):
            untestable.append(r(left, right, d_sep))
        else:
            testable.append(r(left, right, d_sep))

    nodes = set(topological_sort)
    relevant = nodes - ignore - unobserved if testable_only else nodes - ignore

    # isolated nodes are always independent of any other node.
    # nodes may have become isolated after mutilation
    isolated = nodes - edges_to_nodes(edges)

    order = [node for node in topological_sort if node in relevant]

    for left, right in combinations(order, 2):
        # skipping if the nodes are direct neighbours
        if left in neighbours[right]:
            continue
        # if any of the two nodes is isolated, immediately add the pair
        if bool({left, right} & isolated):
            update(left, right, [])
            continue

        # figuring out d-separators for this pair
        simple_paths = all_simple_paths(left, right, neighbours, reachable)
        existing_paths = find_existing_paths(edges, simple_paths)
        available = {
            node
            for path in existing_paths
            for node in path
            if node not in {left, right}
        }
        # When only looking at testable independencies, can remove unobserved
        # from available.
        # This can speed up find_separators as it has to check fewer combinations
        if testable_only:
            available -= unobserved

        d_separators = find_separators(available, edges, existing_paths, descendants)
        minimal_d_separators = find_minimal_separators(d_separators)

        # skipping if d_separators is the empty list -> no conditional independencies
        if bool(minimal_d_separators):
            for d_sep in sorted(minimal_d_separators):
                update(left, right, sorted(d_sep))

    return ConditionalIndependecies(testable, untestable)


class DirectedAcyclicGraph:
    """Directed Acyclic Graph and associated methods."""

    nodes: set[Node]
    edges: set[Edge]
    topological_sort: NodeSequence
    topological_generations: list[NodeSequence]
    parents: NodeMap
    _neighbours: NodeMap
    _reachable: NodeMap

    def __init__(
        self,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
    ) -> None:
        """Construct a DAG from a list of nodes and a list of edges."""
        # Using sets since that simplifies comparisons and makes sure all are unique
        self.nodes = set(nodes)
        self.edges = set(edges)

        if s := edges_to_nodes(edges) - self.nodes:
            msg = f"Nodes {s} appear in edges, but do not appear in nodes."
            raise MissingNodeError(msg)

        # Building the topological sort also checks whether this is a DAG.
        ts = get_topological_sortings(self.nodes, self.edges)
        self.topological_sort = ts.topological_sort
        self.topological_generations = ts.topological_generations

        # Storing often used properties of the DAG in lieu of caching
        self.parents = get_parents(self.nodes, self.edges)
        self._neighbours = get_neighbours(self.topological_sort, self.edges)
        self._reachable = get_reachable(self._neighbours)
        # Setting up paths, starting with the known edges

    def mutilate(
        self, over: NodeSequence | None, under: NodeSequence | None
    ) -> set[Edge]:
        """Mutilate the graph by cutting edges into (over) and out of (under) nodes."""  # noqa: E501
        if over is None and under is None:
            return self.edges

        over = [] if over is None else over
        under = [] if under is None else under
        return mutilate(self.edges, over, under)

    def _parse_do(self, do: NodeSequence | None) -> str:
        """Create added message for interventions."""
        return "" if do is None else " under " + ", ".join(f"do({n})" for n in do)

    def backdoor_criterion(
        self,
        exposure: Node,
        outcome: Node,
        do: NodeSequence,
        unobserved: set[Node],
    ) -> None:
        """Display adjustment sets for exposure -> outcome."""
        edges = self.mutilate(over=do, under=None)
        # Finding all paths in the base DAG.
        paths = all_simple_paths(
            exposure, outcome, self._neighbours, self._reachable
        )

        do_msg = self._parse_do(do)
        if do is not None:
            # If a mutilation was applied,
            # remove the paths that no longer appear in the mutilated DAG.
            paths = find_existing_paths(edges, paths)

        # If this means there are no more paths available, then the causal effect no
        # longer appears in the DAG.
        if not bool(paths):
            return print(  # noqa: T201
                f"Causal effect {exposure} -> {outcome} does not appear in the DAG {do_msg}."  # noqa: E501
            )

        # Adding the interventions
        msg = f"Causal effect of {exposure} -> {outcome} {do_msg}:\n"

        descendants = get_descendants(edges, self.topological_generations)

        backdoor = backdoor_criterion(
            exposure, outcome, edges, paths, unobserved, descendants
        )
        return print(msg + repr(backdoor))  # noqa: T201

    def conditional_independencies(
        self,
        over: NodeSequence | None,
        under: NodeSequence | None,
        ignore: set[str],
        unobserved: set[str],
        show: CIOptions = "testable",
    ) -> None:
        """Display conditional independencies."""
        edges = self.mutilate(over, under)
        testable_only = show == "testable"
        descendants = get_descendants(edges, self.topological_generations)
        ci = conditional_independencies(
            self.topological_sort,
            edges,
            ignore,
            unobserved,
            self._neighbours,
            descendants,
            self._reachable,
            testable_only=testable_only,
        )

        do_msg = self._parse_do(over)

        match show:
            case "testable":
                print(ci.render_testable(do_msg))  # noqa: T201
            case "untestable":
                print(ci.render_untestable(do_msg))  # noqa: T201
            case "both":
                print(ci.render(do_msg))  # noqa: T201

    def is_d_separator(
        self,
        x: set[Node],
        y: set[Node],
        z: set[Node],
        over: NodeSequence | None,
        under: NodeSequence | None,
    ) -> bool:
        """Check if Z d-separates X and Y, optionally under mutilation."""
        edges = self.mutilate(over, under)
        descendants = get_descendants(edges, self.topological_generations)
        return is_d_separator(
            x,
            y,
            z,
            edges,
            self._neighbours,
            descendants,
            self._reachable,
        )

    def find_d_separators(
        self,
        x: set[Node],
        y: set[Node],
        unobserved: set[Node],
        over: NodeSequence | None,
        under: NodeSequence | None,
    ) -> None:
        """Find sets that d-separate X and Y."""
        edges = self.mutilate(over, under)
        descendants = get_descendants(edges, self.topological_generations)
        d_sep = find_d_separators(
            x, y, edges, self._neighbours, descendants, self._reachable, unobserved
        )
        print(d_sep)  # noqa: T201
