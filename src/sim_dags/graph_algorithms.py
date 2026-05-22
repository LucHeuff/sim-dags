from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import polars as pl
from more_itertools import sliding_window

# --- Graph adjustments


def _over(graph: nx.DiGraph, variables: list[str] | None = None) -> nx.DiGraph:
    """Remove edges pointing into variables from graph."""
    if variables is None:
        return graph
    edges = [edge for edge in graph.edges if edge[1] in variables]
    graph.remove_edges_from(edges)
    return graph


def _under(graph: nx.DiGraph, variables: list[str] | None = None) -> nx.DiGraph:
    """Remove edges coming out of variables from graph."""
    if variables is None:
        return graph
    edges = [edge for edge in graph.edges if edge[0] in variables]
    graph.remove_edges_from(edges)
    return graph


def mutilate(
    graph: nx.DiGraph, over: list[str] | None, under: list[str] | None
) -> nx.DiGraph:
    """Mutilate graph by removing edges into over and out of under."""
    g = graph.copy()
    return _under(_over(g, over), under)


# --- D-seperation algorithms


def path_has_collider(graph: nx.DiGraph, path: list[str]) -> bool:
    """Check if this path in this graph contains a collider."""
    # Can't have a collider if path has less than 3 nodes.
    if len(path) < 3:  # noqa: PLR2004
        return False
    # Collider when u -> c <- v
    for u, c, v in sliding_window(path, 3):
        if graph.has_edge(u, c) and graph.has_edge(v, c):
            return True
    return False


def find_minimal_adjustment_set(
    available: list[str], open_paths: list[list[str]]
) -> list[list[str]] | None:
    """Find minimal adjustment set for these variables."""
    # If there are no available nodes, then there is no adjustment set
    if len(available) == 0:
        return None

    # Finding how often each node appears in the open paths
    frequency = [
        (node, sum(node in path for path in open_paths)) for node in available
    ]
    # if any single node appears as often as the number of open paths,
    # combinations do not need to be searched.

    if (max_ := max(n for _, n in frequency)) == len(open_paths):
        return [[node] for node, n in frequency if n == max_]

    # Nodes that do not appear in any path are irrelevant, ignoring these
    relevant = [node for node, n in frequency if n > 0]

    if len(relevant) == 0:  # returning when none of the nodes are relevant
        return None

    adjustment = []
    min_size = len(relevant) + 1  # adding 1 to avoid early stopping

    for size in range(2, len(relevant) + 1):  # not using min_size cause that changes
        # if we already found smaller sets than this, these aren't minimal.
        if size > min_size:
            break
        # check for combinations of this size whether they close all open paths
        for c in combinations(relevant, size):
            if all(any(node in path for node in c) for path in open_paths):
                adjustment.append(list(c))
                # update min size if this is smaller,
                # adding 1 to include sets 1 size larger, can be convenient
                min_size = min(len(c) + 1, min_size)

    return adjustment if len(adjustment) > 0 else None


def find_minimal_d_separators(
    graph: nx.DiGraph, left: str, right: str
) -> list[list[str]] | None:
    """Find minimal d-separating sets between two nodes in the graph."""
    # That means empty list means z = ∅ is a valid independency.
    # Only consider ancestors of left and right
    available = [
        node
        for node in set(nx.ancestors(graph, left)) | set(nx.ancestors(graph, right))
        if node not in [left, right]
    ]

    d_sep = []

    # First checking the empty set
    if nx.is_d_separator(graph, left, right, set()):
        d_sep.append([])

    # adding one since available may have length one,
    # and we also want to iterate over that
    for s in range(1, len(available) + 1):
        for c in combinations(available, s):
            if nx.is_minimal_d_separator(graph, left, right, set(c)):
                d_sep.append(sorted(c))  # noqa: PERF401

    if len(d_sep) == 0:
        # if nothing was added to d_sep, then there are no d-separating sets
        return None

    return d_sep


def render_path(graph: nx.DiGraph, path: list[str]) -> str:
    """Render path in a graph in a legible fashion."""
    assert nx.is_path(graph.to_undirected(), path), "path doesn't appear in DAG."

    render = f"{path[0]} "
    for u, v in sliding_window(path, 2):
        if graph.has_edge(u, v):
            render += "->"
        else:
            render += "<-"
        render += f" {v} "

    return render.strip()


# --- Backdoor criterion
@dataclass(slots=True, frozen=True)
class BackdoorCriterion:
    """Container for partial outputs of backdoor criterion."""

    backdoor_paths: list[list[str]]
    open_paths: list[str]
    adjustment_sets: list[list[str]]

    def __repr__(self) -> str:
        """Legible string with backdoor criterion output."""
        if len(self.backdoor_paths) == 0:
            return "No backdoor paths found, no adjustment is necessary."
        if len(self.open_paths) == 0:
            return "No open backdoor paths found, no adjustment is necessary."

        # Formatting the backdoor paths
        msg = f"Found {len(self.open_paths)} open paths:\n  {'\n  '.join(self.open_paths)}\n"  # noqa: E501

        if len(self.adjustment_sets) == 0:
            msg += "No adjustment sets found."
        else:
            adj = [f"{{{', '.join(set_)}}}" for set_ in self.adjustment_sets]
            msg += f"Available adjustment sets:\n  {'\n  '.join(adj)}"

        return msg


def backdoor_criterion(
    graph: nx.DiGraph,
    exposure: str,
    outcome: str,
    do: list[str],
    unobserved: set[str],
) -> BackdoorCriterion:
    """Calculate adjustment set using the backdoor criterion."""
    # Making a mutilated graph, applying do operations and removing arrows
    # coming out of the exposure (Rule 2 of do-calculus)
    graph = mutilate(graph, do, [exposure])

    # Backdoor paths are remaining undirected paths from exposure to outcome.
    # Needs a try-except since networkx doesn't handle non-existing paths gracefully
    try:
        backdoor_paths = list(
            nx.all_shortest_paths(graph.to_undirected(), exposure, outcome)
        )
    except nx.exception.NetworkXNoPath:
        backdoor_paths = []

    if len(backdoor_paths) == 0:
        return BackdoorCriterion(backdoor_paths, [], [])

    # A backdoor path is open if it doesn't have a collider on it
    open_paths = [
        path for path in backdoor_paths if not path_has_collider(graph, path)
    ]
    if len(open_paths) == 0:
        return BackdoorCriterion(backdoor_paths, [], [])

    open_ = [render_path(graph, path) for path in open_paths]

    # Finding minimal adjustment set -> only need to look at available nodes.
    # Nodes are available if they are observed ancestors of exposure
    available = list(set(nx.ancestors(graph, exposure)) - unobserved)

    if len(available) == 0:
        return BackdoorCriterion(backdoor_paths, open_, [])

    adjustment = find_minimal_adjustment_set(available, open_paths)

    if adjustment is None:
        return BackdoorCriterion(backdoor_paths, open_, [])
    return BackdoorCriterion(backdoor_paths, open_, adjustment)


# --- Conditional independencies
@dataclass(slots=True, frozen=True)
class ConditionalIndependecies:
    """Container for partial outputs of conditional indenpendencies."""

    testable: list[str]
    untestable: list[str]

    @property
    def render_testable(self) -> str:
        """Render testable independencies to a string."""
        if len(self.testable) == 0:
            return "No testable conditional independencies."
        return f"Testable independencies:\n  {'\n  '.join(self.testable)}"

    @property
    def render_untestable(self) -> str:
        """Render untestable independencies to a string."""
        if len(self.untestable) == 0:
            return "No untestable conditional independencies."
        return f"Untestable independencies:\n  {'\n  '.join(self.untestable)}"

    def __repr__(self) -> str:
        """Renders conditional independencies."""
        if len(self.testable) == 0 and len(self.untestable) == 0:
            return "The graph does not imply any conditional independencies."
        return f"{self.render_testable}\n{self.render_untestable}"


def conditional_independencies(
    graph: nx.DiGraph,
    do: list[str] | None,
    ignore: list[str] | None,
    unobserved: set[str] | None,
) -> ConditionalIndependecies:
    """Calculate conditional independencies."""
    ignore = [] if ignore is None else ignore
    unobserved = set() if unobserved is None else unobserved

    # rendering out unobserveds in brackets
    def process(var: str) -> str:
        return f"({var})" if var in unobserved else var

    def process_conditional(left: str, right: str) -> str:
        return f"{process(left)} ⫫ {process(right)}"

    def render(left: str, right: str, d_sep: list[str]) -> str:
        if len(d_sep) == 0:
            return process_conditional(left, right)
        return f"{process_conditional(left, right)} | {','.join(list(map(process, sorted(d_sep))))}"  # noqa: E501

    testable = []
    untestable = []

    nodes = [node for node in graph.nodes if node not in ignore]

    graph = mutilate(graph, do, None)
    for left, right in combinations(nodes, 2):
        indep = find_minimal_d_separators(graph, left, right)
        if indep is not None:
            for d_sep in sorted(indep):
                all_ = [left, right, *d_sep]
                if any(v in unobserved for v in all_):
                    untestable.append(render(left, right, d_sep))
                else:
                    testable.append(render(left, right, d_sep))

    return ConditionalIndependecies(testable, untestable)


def calculate_node_positions(graph: nx.DiGraph) -> pl.DataFrame:
    """Calculate positions for nodes in the DAG."""
    layers = list(nx.topological_generations(graph))

    # max space leaves edge gap on both sides
    # Horizontal position: and dividing max space evenly over all layers

    x_range = np.linspace(0, 1, len(layers))
    x_pos = {}
    pos = {}

    fixed = []

    def centered_pos(n: int) -> list[float]:
        if n == 1:
            return [0.5]
        return [0.5 + (i - (n - 1) / 2) * (1 / (n - 1)) for i in range(n)]

    for x, nodes in enumerate(layers):
        # adding first and last layer nodes to fixed nodes
        if x == 0 or x == len(layers) - 1:
            fixed.extend(nodes)
        # figure out the y positions for this layer, centered around 0.5
        y_range = centered_pos(len(nodes))
        for y, node in enumerate(nodes):
            x_pos[node] = x_range[x]
            pos[node] = (x_range[x], y_range[y])

    # letting spring layout determine y-positions
    spring = nx.spring_layout(
        graph, pos=pos, fixed=fixed, center=(0.5, 0.5), method="energy"
    )

    positions = [
        {"node": node, "x": x_pos[node], "y": spring[node][1]}
        for node in graph.nodes
    ]

    return pl.DataFrame(positions)
