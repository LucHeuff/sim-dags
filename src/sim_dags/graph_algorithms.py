import numpy as np
import polars as pl

from sim_dags.graphs import Edge, Node, NodeSequence


def calculate_node_positions(
    topological_generations: list[NodeSequence],
) -> pl.DataFrame:
    """Calculate positions for nodes in the DAG."""
    layers = topological_generations

    # max space leaves edge gap on both sides
    # Horizontal position: and dividing max space evenly over all layers

    x_range = np.linspace(0, 1, len(layers))
    positions = []

    def centered_pos(n: int) -> list[float]:
        if n == 1:
            return [0.5]
        return [0.5 + (i - (n - 1) / 2) * (1 / (n - 1)) for i in range(n)]

    for x, gen in enumerate(layers):
        # figure out the y positions for this layer, centered around 0.5
        y_range = centered_pos(len(gen))
        for y, node in enumerate(gen):
            positions.append({"node": node, "x": x_range[x], "y": y_range[y]})

    return pl.DataFrame(positions)


def dagitty_code(
    edges: set[Edge],
    topological_generations: list[NodeSequence],
    unobserved: set[Node],
) -> str:
    """Convert edges and topological generations to dagitty code."""
    pos = calculate_node_positions(topological_generations)

    # determining bounding box
    min_y = min(pos.select(pl.col("y").min()).item(), 0) - 0.05
    max_y = max(pos.select(pl.col("y").max()).item(), 1) + 0.05
    bounding_box = f'bb="-0.05,{min_y:.3f},1.05,{max_y:.3f}"'

    def parse_node(row: dict[str, str | float]) -> str:
        node = row["node"]
        latent = "latent," if node in unobserved else ""
        return f'{row["node"]} [{latent}pos="{row["x"]:.3f},{row["y"]:.3f}"]'

    str_nodes = "\n".join(parse_node(row) for row in pos.to_dicts())
    str_edges = "\n".join(f"{u} -> {v}" for (u, v) in edges)

    return f"dag {{\n{bounding_box}\n{str_nodes}\n{str_edges}\n}}"
