from collections import deque
from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal, Self

from more_itertools import sliding_window

from sim_dags.exceptions import (
    MissingNodeError,
    NoDisjointSetsError,
    NotADAGError,
)
from sim_dags.utils import all_combinations

type Edge = tuple[str, str]
type Edges = set[Edge]
type Node = str
type NodeMap = dict[Node, list[Node]]
type NodeSequence = list[Node]

type CIOptions = Literal["testable", "untestable", "both"]


@dataclass(slots=True)
class NodePaths:
    """Container for undirected node paths."""

    paths: list[NodeSequence] = field(default_factory=list)
    complete: bool = False

    def add_paths(self, paths: list[NodeSequence]) -> Self:
        """Add paths."""
        assert len(paths) > 0, "Don't add empty paths."
        new_paths = [path for path in paths if path not in self.paths]
        self.paths += new_paths
        return self

    def set_complete(self, complete: bool) -> Self:  # noqa: FBT001
        """Set complete status."""
        self.complete = complete
        return self

    @property
    def is_complete(self) -> bool:
        """Check if NodePaths is complete."""
        return self.complete


class PathMap:
    """Provides a mapping of paths between two nodes."""

    mapping: dict[Edge, NodePaths]

    def __init__(self) -> None:
        self.mapping = {}

    @classmethod
    def from_edges(cls, edges: Edges) -> Self:
        """Construct a PathMapping from an initial list of edges."""
        self = cls()
        for u, v in edges:
            self._add_paths(u, v, [[u, v]], complete=True)
        return self

    def _is_reversed(self, source: Node, target: Node) -> bool:
        """Check if the mapping is reversed."""
        return (target, source) in self.mapping

    @staticmethod
    def _reverse(paths: list[NodeSequence]) -> list[NodeSequence]:
        """Reverse each path in a list of paths."""
        return [path[::-1] for path in paths]

    def _add_paths(
        self,
        source: Node,
        target: Node,
        paths: list[NodeSequence],
        *,
        complete: bool = False,
    ) -> None:
        """Add a NodePath to the PathMapping."""
        self.mapping[(source, target)] = NodePaths(paths, complete)

    def update_paths(
        self,
        source: Node,
        target: Node,
        paths: list[NodeSequence],
        *,
        complete: bool = False,
    ) -> None:
        """Update an existing path."""
        if not self.has_path(source, target):  # add new path if it didn't exist yet
            return self._add_paths(source, target, paths, complete=complete)

        # adding reversed paths if original was stored in reverse order.
        if self._is_reversed(source, target):
            self.mapping[(target, source)].add_paths(
                self._reverse(paths)
            ).set_complete(complete)
        else:
            self.mapping[(source, target)].add_paths(paths).set_complete(complete)

        return None

    def has_path(self, source: Node, target: Node) -> bool:
        """Check if the path from source to target appears in the mapping."""
        return (source, target) in self.mapping or self._is_reversed(source, target)

    def get_paths(self, source: Node, target: Node) -> list[NodeSequence]:
        """Get NodePaths from source to target."""
        if self._is_reversed(source, target):
            return self._reverse(self.mapping[(target, source)].paths)
        return self.mapping[(source, target)].paths

    def path_is_complete(self, source: Node, target: Node) -> bool:
        """Return whether the path is complete."""
        if self._is_reversed(source, target):
            return self.mapping[(target, source)].is_complete
        return self.mapping[(source, target)].is_complete


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


def all_simple_paths(
    source: Node,
    target: Node,
    paths: PathMap,
    neighbours: NodeMap,
) -> list[NodeSequence]:
    """Generate all simple paths from a list of edges."""
    # keeping track of nodes already visited -> we're looking for simple paths
    print(f"Looking for all paths between {source} and {target}")

    def find_paths(start: Node, end: Node, visited: set[Node]) -> list[NodeSequence]:
        """Find all paths between start and target."""
        # making sure not to override visited in recursion
        visited = visited.copy()
        print(f"\nfind_paths({start}, {end}, {visited=})")

        # --- First checking if path exist in lookup table, avoiding retracing paths
        if paths.has_path(start, end):
            print("path already in PathMap")
            # ignoring paths that go back on visited nodes
            known_paths = [
                path
                for path in paths.get_paths(start, end)
                if not bool(set(path) & visited)
            ]

            # if we're sure these are all the paths, return immediately
            if paths.path_is_complete(start, end):
                return known_paths
            # otherwise, add the unique nodes that we now traversed over this path
            visited |= {node for path in known_paths for node in path}
            found_paths = known_paths
        else:
            # haven't found anything yet, but need a list for later
            print("path is unknown.")
            found_paths = []

        # --- If there were no (complete) paths, step onto unvisited neighbours
        start_neigh = set(neighbours[start]) - visited
        end_neigh = set(neighbours[end]) - visited

        print(f"Unvisited neighbours of {start}: {start_neigh}")
        print(f"Unvisited neighbours of {end}: {end_neigh}")

        # --- If start and end have neighbours in common, this forms the path
        common = set(start_neigh) & set(end_neigh)
        if bool(common):
            print(f"{start} and {end} have {common} in common")
            common_paths = [[start, c, end] for c in common]
            # Update the paths, but don't assume we've completed the list
            paths.update_paths(start, end, common_paths)
            found_paths += common_paths

            print(f"{found_paths = }")

            # if both sets of neighbours match, return the common paths
            if start_neigh == common == end_neigh:
                return found_paths

            # either start_neigh or end_neigh might still be equal to common now.
            # In that case I should use start or end respectively instead
            start_neigh = {start} if not bool(s := start_neigh - common) else s
            end_neigh = {end} if not bool(e := end_neigh - common) else e
            visited |= common

        # --- Search recursively over the remaining neighbours
        print(f"Searching between pairs of {start_neigh = } and {end_neigh = }")
        for s, t in product(sorted(start_neigh), sorted(end_neigh)):
            new_paths = [
                [start, *path, end]
                for path in find_paths(s, t, visited | {start, end})
                if len(path) > 0
            ]
            if len(new_paths) > 0:
                # update the paths, but don't assume we've completed the list
                paths.update_paths(start, target, new_paths)
                found_paths += new_paths

        print(f"Found the following paths between {start} and {end}:\n{found_paths}")
        return found_paths

    simple_paths = find_paths(source, target, set())

    if len(simple_paths) > 0:
        # Now that we've traversed the entire (sub)graph, paths are complete
        paths.update_paths(source, target, simple_paths, complete=True)

    return simple_paths


# --- Helper functions for finding d-separators.


def find_existing_paths(
    edges: Edges, paths: list[NodeSequence]
) -> list[NodeSequence]:
    """Get a list of paths that exist according to edges."""
    return [path for path in paths if undirected_path_exists(path, edges)]


def find_open_paths(edges: Edges, paths: list[NodeSequence]) -> list[NodeSequence]:
    """Get a list of open paths based on given edges."""
    return [path for path in paths if not path_has_collider(path, edges)]


def find_available_nodes(
    edges: Edges, paths: list[NodeSequence], unobserved: set[Node]
) -> set[Node]:
    """Nodes are available when they are observed and not colliders."""
    # finding all nodes that appear in paths
    nodes = {node for path in paths for node in path}
    # finding nodes that appear as a collider at least once
    colliders = {
        node
        for node in nodes
        if any(is_collider(node, path, edges) for path in paths)
    }
    return nodes - colliders - unobserved


def find_separators(
    available: set[Node], paths: list[NodeSequence]
) -> list[NodeSequence]:
    """Find d-separating sets: subset of available that appear in all paths."""
    if not bool(paths):
        return [[]]
    d_sep = []
    for i in range(len(available)):
        for c in combinations(available, i + 1):
            z = set(c)
            # if z already appears, skip it
            if z in d_sep:
                continue
            # z is a d-separator if it appears in all paths.
            closed = [path for path in paths if set(path) & z]
            if len(closed) == len(paths):
                d_sep.append(z)
                # if z is a d-separator, then any combination of z and the
                # other available nodes is also a d-separator
                d_sep.extend(z | c_ for c_ in all_combinations(available - z))

    # converting sets to sorted lists for more consistent output
    return [sorted(z) for z in d_sep]


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
    paths: PathMap,
    neighbours: NodeMap,
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

    # Finding all paths between nodes in X and nodes in Y,
    # if path wasn't pruned away by mutilation
    all_paths = [
        path
        for (x_, y_) in pairs
        for path in all_simple_paths(x_, y_, paths, neighbours)
        if undirected_path_exists(path, edges)
    ]

    # if there are no paths between X and Y, then Z (trivially) is a d-separator.
    if not bool(all_paths):
        return True

    # available nodes are all nodes that appear in paths that are not in X or Y
    # in other words, these are the potential d-separators.
    available = {node for path in all_paths for node in path if node not in x | y}

    # If no Z nodes appear in the available nodes, the Z cannot be a d-separator.
    if not (available & z):
        return False

    # If any variable Z is a collider on any of the paths, then Z cannot a d-separate
    if any(is_collider(z_, path, edges) for path in all_paths for z_ in z):
        return False

    # Z is a d-separator if it appears in the separators
    # for all pairs of path between X and Y.
    return all(
        sorted(z) in find_separators(available, paths.get_paths(*pair))
        for pair in pairs
    )


# --- Backdoor criterion
@dataclass(slots=True, frozen=True)
class BackdoorCriterion:
    """Container for partial outputs of backdoor criterion."""

    edges: set[Edge]
    backdoor_paths: list[NodeSequence] | None = None
    open_paths: list[NodeSequence] | None = None
    adjustment_sets: list[NodeSequence] | None = None

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

    def __repr__(self) -> str:
        """Legible string with backdoor criterion output."""
        if self.backdoor_paths is None:
            return "No backdoor paths found, no adjustment is necessary."
        if self.open_paths is None:
            return "No open backdoor paths found, no adjustment is necessary."

        # Formatting the backdoor paths
        rendered_open = [self.render_path(path) for path in self.open_paths]
        msg = f"Found {len(self.open_paths)} open paths:\n  {'\n  '.join(rendered_open)}\n"  # noqa: E501

        if self.adjustment_sets is None:
            msg += "No adjustment sets found."
        else:
            adj = [f"{{{', '.join(set_)}}}" for set_ in self.adjustment_sets]
            msg += f"Available adjustment sets:\n  {'\n  '.join(adj)}"

        return msg


def backdoor_criterion(
    exposure: Node,
    outcome: Node,
    edges: set[Edge],
    simple_paths: list[NodeSequence],
    unobserved: set[Node],
) -> BackdoorCriterion:
    """Perform backdoor criterion for a given list of paths."""
    unobserved |= {exposure, outcome}
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

    # --- Finding adjustment sets. Candidates are observed non-colliders.
    # the exposure and outcome themselves are should also be ignored.
    available = find_available_nodes(
        edges, backdoor_paths, unobserved | {exposure, outcome}
    )

    # adjustment sets consist of combinations of nodes that close all open paths
    adjustment_sets = find_separators(available, open_paths)

    # if no adjustment sets were found, return backdoor and open paths
    if not bool(adjustment_sets):
        return BackdoorCriterion(edges, backdoor_paths, open_paths)

    # otherwise, return everything
    return BackdoorCriterion(edges, backdoor_paths, open_paths, adjustment_sets)


# --- Conditional independencies
@dataclass(slots=True, frozen=True)
class ConditionalIndependecies:
    """Container for partial outputs of conditional independencies."""

    testable: list[str]
    untestable: list[str]

    def render_list(self, list_: list[str], title: str, do_msg: str) -> str:
        """Render a list into a string."""
        if len(list_) == 0:
            return f"No {title} conditional independencies {do_msg}."
        return (
            f"{title.capitalize()} independencies {do_msg}:\n {'\n  '.join(list_)}"
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
                f"The graph does not imply any conditional independencies {do_msg}."
            )
        return f"{self.render_testable(do_msg)}\n{self.render_untestable(do_msg)}"


def conditional_independencies(
    nodes: set[Node],
    edges: Edges,
    ignore: set[Node],
    unobserved: set[Node],
    paths: PathMap,
    neighbours: NodeMap,
    *,
    testable_only: bool,
) -> ConditionalIndependecies:
    """Find all conditional independencies implied by this graph."""
    # Depending on the show option, not all nodes are relevant
    relevant = nodes - ignore - unobserved if testable_only else nodes - ignore

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

    # isolated nodes are always independent of any other node.
    # nodes may have become isolated after mutilation
    isolated = nodes - edges_to_nodes(edges)

    for left, right in combinations(sorted(relevant), 2):
        # skipping if the nodes are direct neighbours
        if left in neighbours[right]:
            continue
        # if any of the two nodes is isolated, immediately add the pair
        if bool({left, right} & isolated):
            update(left, right, [])
            continue

        # figuring out d-separators for this pair
        simple_paths = all_simple_paths(left, right, paths, neighbours)
        existing_paths = find_existing_paths(edges, simple_paths)
        open_paths = find_open_paths(edges, existing_paths)
        available = find_available_nodes(edges, open_paths, {left, right})
        d_separators = find_minimal_separators(
            find_separators(available, open_paths)
        )
        # skipping if d_separators is the empty list -> no conditional independencies
        if bool(d_separators):
            for d_sep in sorted(d_separators):
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
    _paths: PathMap

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
        # Setting up paths, starting with the known edges
        self._paths = PathMap.from_edges(self.edges)

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
        return "" if do is None else "under " + ", ".join(f"do({n})" for n in do)

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
        paths = all_simple_paths(exposure, outcome, self._paths, self._neighbours)

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

        backdoor = backdoor_criterion(exposure, outcome, edges, paths, unobserved)
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
        ci = conditional_independencies(
            self.nodes,
            edges,
            ignore,
            unobserved,
            self._paths,
            self._neighbours,
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
        return is_d_separator(x, y, z, edges, self._paths, self._neighbours)
