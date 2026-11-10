"""Parse graphs in custom .xyz format, and measure traits.
Points are stored in a tree using an undirected NetworkX graph.
Each node has a unique numerical identifier (node_num), and an attribute "pos": an (x,y) coordinate pair corresponding to its position in 2D space.
Each edge has an attribute "length": the Euclidean distance between the nodes it connects.
TO-DO:
Trait measurement
Time series
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import copy
import networkx as nx
import math
import logging
from pathlib import Path

from scipy.spatial import ConvexHull  # Import ConvexHull class
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap, to_hex
from matplotlib.patches import Rectangle
from networkx.readwrite import json_graph

from ariadne_roots.pareto_functions import (
    pareto_front,
    pareto_steiner_fast_analytical_preserve_tips_semi_random,
    pareto_front_3d_path_tortuosity,
    random_tree_3d_path_tortuosity,
    get_critical_nodes,
    straighten_tree,
    graph_costs,
    point_dist,
    _tree_hover_payload,
    random_steiner_wrapper_analytical,
)
from ariadne_roots import config


# parser = argparse.ArgumentParser(description='select file')
# parser.add_argument('-i', '--input', help='Full path to input file', required=True)
# args = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)












def get_graph_from_json(json_file):
    """Load a JSON file and convert it to a NetworkX graph.

    Args:
        json_file (str or Path): Path to the JSON file.

    Returns:
        nx.Graph: A NetworkX graph object, converted to undirected if needed.
    """
    with open(json_file, mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
        graph = json_graph.adjacency_graph(data)

    if graph.is_directed():
        graph = graph.to_undirected()

    return graph


def calc_len_PR(G, root_node):
    """For a given graph and the uppermost node, calculate the PR length."""
    bfs_paths = dict(nx.bfs_successors(G, root_node))

    PRs = []  # list of PR nodes in order of increasing depth

    for node, children in bfs_paths.items():
        if G.nodes[node]["LR_index"] is None:
            PRs.append(node)
            for child_node in children:
                if G.nodes[child_node]["LR_index"] is None:
                    # catch the last node in the PR, which won't appear in the iterator since it has no children
                    final = child_node

    PRs.append(final)

    # calculate pairwise Euclidean distances and sum
    return calc_root_len(G, PRs)


def calc_root_len(G, nodes):
    """Return the pairwise Euclidean distance along a list of consecutive nodes."""
    dist = 0

    # order matters! assumes consecutive, increasing depth
    for prev, current_node in zip(nodes, nodes[1:]):
        segment = distance(G.nodes[prev]["pos"], G.nodes[current_node]["pos"])
        dist += segment
        # might as well annotate the edges while I'm here
        G.edges[prev, current_node]["weight"] = segment

    return dist


def calc_len_LRs(H):
    """Find the total length of each LR type in the graph."""
    # minimum length (px) for LR to be considered part of the network
    # based on root hair emergence times
    # threshold = 117
    threshold = 0

    # dict of node ids : LR index, for each LR node
    idxs = nx.get_node_attributes(H, "LR_index")
    idxs = {k: v for k, v in idxs.items() if v is not None}  # drop empty (PR) nodes

    num_LRs = max(idxs.values()) + 1
    # CPL : Add a minimum LR_index. Main root is None
    min_num_LRs = min(idxs.values())  # should be 1
    results = {}

    for i in range(min_num_LRs, num_LRs):
        # gather nodes corresponding to the current LR index
        selected = []

        for node in H.nodes(data="LR_index"):
            if node[1] == i:
                selected.append(node[0])
                # make note of the root degree (should be the same for all nodes in loop)
                current_degree = H.nodes[node[0]]["root_deg"]

        # to find the shallowest node in LR, we iterate through them
        # until we find the one whose parent_node has a lesser root degree
        sub = H.subgraph(selected)
        for node in sub.nodes():
            parent_node = list(H.predecessors(node))
            # print(node, parent_node)
            assert len(parent_node) == 1  # should be a tree
            if H.nodes[parent_node[0]]["root_deg"] < current_degree:
                sub_top = node

        # now we can DFS to order all nodes by increasing depth
        ordered = list(nx.dfs_tree(sub, sub_top).nodes())

        # also include the parent_node of the shallowest node in the LR (the 'branch point')
        parent_node = list(H.predecessors(ordered[0]))
        assert len(parent_node) == 1

        nodes_list = parent_node + ordered

        length = calc_root_len(H, nodes_list)

        if length < threshold:
            H.remove_nodes_from(ordered)
        else:
            # now we can calculate the gravitropic set point angle
            # branch coordinates
            p2 = np.array(H.nodes[parent_node[0]]["pos"])
            # LR coordinates
            p3 = np.array(H.nodes[ordered[0]]["pos"])

            # recall: in our coordinate system, the top node is (0,0)
            # x increases to the right; y increases downwards
            # unit vector of gravity
            g = np.array([0, 1])
            # normalized vector of LR emergence
            lr = p3 - p2
            norm_lr = np.linalg.norm(lr)
            assert norm_lr > 0
            lr = lr / norm_lr

            # angle between LR emergence and the vector of gravity:
            # this will be symmetric, whichever side of the PR the LR is on
            theta = np.rad2deg(math.acos(np.dot(lr, g)))

            # print(f'The ordered list of nodes that make up LR #{i} is:', nodes_list)
            results[i] = [length, theta]

    assert nx.is_tree(H)
    return results
    # add LR_index awareness: all, 1 deg, 2 deg, n deg




def calculate_plot_buffer(base_min, base_max, buffer_percent=0.20, min_buffer=1.0):
    """Calculate plot buffer for axis limits.

    Computes appropriate buffer margins for plot axes based on data range,
    handling edge cases like negative coordinates, zero values, and small ranges.

    Args:
        base_min: Minimum value in the data range
        base_max: Maximum value in the data range
        buffer_percent: Buffer size as fraction of range (default 0.20 = 20%)
        min_buffer: Minimum buffer to prevent degenerate plots (default 1.0)

    Returns:
        Tuple of (min_limit, max_limit) for axis

    Examples:
        >>> calculate_plot_buffer(0, 10)  # Normal positive range
        (-2.0, 12.0)

        >>> calculate_plot_buffer(-10, 10)  # Range crossing zero
        (-14.0, 14.0)

        >>> calculate_plot_buffer(0, 0)  # Degenerate range (all zeros)
        (-1.0, 1.0)

        >>> calculate_plot_buffer(-5, -3)  # Negative range
        (-5.4, -2.6)
    """
    data_range = abs(base_max - base_min)
    buffer = max(data_range * buffer_percent, min_buffer)
    return (base_min - buffer, base_max + buffer)


def plot_all(
    front, actual, randoms, mrand, srand, dest, hover_trees=None
):  # pragma: no cover
    """Plot Pareto front with actual and random trees, with scaling and centering.

    GUI function for manual visualization. Not tested in automated test suite.
    Applies user-configured scaling and centers view on Pareto front.

    ``hover_trees`` optionally carries compact actual/random tree payloads for
    the interactive HTML hover panel.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def scale_data(data):
        """Scale data by user-configured factor."""
        return data * config.length_scale_factor

    # Scale all data
    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]

    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    scaled_randoms_x = [scale_data(x[0]) for x in randoms]
    scaled_randoms_y = [scale_data(x[1]) for x in randoms]

    scaled_mrand = scale_data(mrand)
    scaled_srand = scale_data(srand)

    # Plot scaled data
    for x, y in zip(scaled_randoms_x, scaled_randoms_y):
        plt.plot(
            x,
            y,
            marker="+",
            color="green",
            markersize=2.5,
            zorder=0.5,
            markeredgewidth=0.5,
        )

    plt.plot(
        scaled_front_x,
        scaled_front_y,
        marker="s",
        linestyle="-",
        markeredgecolor="black",
    )
    plt.plot(
        scaled_actual_x,
        scaled_actual_y,
        marker="x",
        markersize=12,
        zorder=3,
        markeredgewidth=1.5,
    )
    plt.plot(
        scaled_mrand,
        scaled_srand,
        marker="+",
        color="red",
        markersize=12,
        zorder=3,
        markeredgewidth=1.5,
    )

    ax.set_xlabel(f"Total length ({config.length_scale_unit})", fontsize=15)
    ax.set_ylabel(f"Travel distance ({config.length_scale_unit})", fontsize=15)

    # Set limits to focus on the relevant area (Pareto front centered)
    front_x_min = min(scaled_front_x)
    front_x_max = max(scaled_front_x)
    front_y_min = min(scaled_front_y)
    front_y_max = max(scaled_front_y)

    # Create a bounding box that includes Pareto front and random centroid
    # Use absolute value to handle negative coordinates correctly
    base_x_min = min(front_x_min, scaled_mrand)
    base_x_max = max(front_x_max, scaled_mrand)
    base_y_min = min(front_y_min, scaled_srand)
    base_y_max = max(front_y_max, scaled_srand)

    # Calculate axis limits with 20% buffer
    x_min, x_max = calculate_plot_buffer(base_x_min, base_x_max)
    y_min, y_max = calculate_plot_buffer(base_y_min, base_y_max)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Save as PNG
    plt.savefig(dest, bbox_inches="tight", dpi=300)

    # Also save as SVG for better quality
    svg_dest = dest.with_suffix(".svg")
    plt.savefig(svg_dest, bbox_inches="tight", format="svg")

    # Also save an interactive HTML version
    save_pareto_html(
        front,
        actual,
        randoms,
        mrand,
        srand,
        dest.with_suffix(".html"),
        hover_trees=hover_trees,
    )

    plt.close(fig)


def save_pareto_html(
    front, actual, randoms, mrand, srand, dest, hover_trees=None
):  # pragma: no cover
    """Save an interactive Plotly HTML version of the 2D Pareto plot.

    This mirrors :func:`plot_all` but keeps marker and line sizes fixed while
    zooming, which is the behavior of an interactive matplotlib window.
    ``hover_trees`` optionally supplies compact actual/random tree payloads for
    the hover panel.
    """
    import plotly.graph_objects as go

    scale = config.length_scale_factor
    unit = config.length_scale_unit

    front_entries = sorted(
        [
            (alpha, x * scale, y * scale)
            for alpha, (x, y) in front.items()
        ],
        key=lambda entry: entry[1],
    )
    front_x = [entry[1] for entry in front_entries]
    front_y = [entry[2] for entry in front_entries]
    front_alpha = [float(entry[0]) for entry in front_entries]

    random_points = [(x * scale, y * scale) for x, y in randoms]
    random_x = [p[0] for p in random_points]
    random_y = [p[1] for p in random_points]

    actual_x = actual[0] * scale
    actual_y = actual[1] * scale
    front_x_max = max(front_x)
    front_y_max = max(front_y)
    mrand_x = mrand * scale if np.isfinite(mrand) else front_x_max
    mrand_y = srand * scale if np.isfinite(srand) else front_y_max

    base_x_min = min(min(front_x), mrand_x)
    base_x_max = max(front_x_max, mrand_x)
    base_y_min = min(min(front_y), mrand_y)
    base_y_max = max(front_y_max, mrand_y)
    x_min, x_max = calculate_plot_buffer(base_x_min, base_x_max)
    y_min, y_max = calculate_plot_buffer(base_y_min, base_y_max)

    fig = go.Figure()

    if random_x:
        fig.add_trace(go.Scatter(
            x=random_x,
            y=random_y,
            mode="markers",
            name="Random trees",
            marker=dict(symbol="cross", size=5, color="green", opacity=0.7),
            hovertemplate=(
                "Total length: %{x:.2f}<br>"
                "Travel distance: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=front_x,
        y=front_y,
        customdata=front_alpha,
        mode="lines+markers",
        name="Pareto front",
        line=dict(color="#1f77b4", width=2),
        marker=dict(
            symbol="square",
            size=6,
            color="#1f77b4",
            line=dict(color="black", width=1),
        ),
        hovertemplate=(
            "Total length: %{x:.2f}<br>"
            "Travel distance: %{y:.2f}<br>"
            "Alpha: %{customdata:.4f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[actual_x],
        y=[actual_y],
        mode="markers",
        name="Actual tree",
        marker=dict(
            symbol="x",
            size=14,
            color="#ff7f0e",
            line=dict(width=2),
        ),
        hovertemplate=(
            f"Actual tree<br>Total length: {actual_x:.2f}<br>"
            f"Travel distance: {actual_y:.2f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[mrand_x],
        y=[mrand_y],
        mode="markers",
        name="Random centroid",
        marker=dict(
            symbol="cross",
            size=14,
            color="red",
            line=dict(width=2),
        ),
        hovertemplate=(
            f"Random centroid<br>Total length: {mrand_x:.2f}<br>"
            f"Travel distance: {mrand_y:.2f}<extra></extra>"
        ),
    ))

    x_label = f"Total length ({unit})" if unit else "Total length"
    y_label = f"Travel distance ({unit})" if unit else "Travel distance"

    fig.update_layout(
        title="Pareto front",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
        dragmode="zoom",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        margin=dict(l=70, r=30, t=60, b=60),
        autosize=True,
    )

    random_tree_payloads = []
    front_tree_payloads = []
    actual_tree_payload = None
    if hover_trees:
        random_tree_payloads = hover_trees.get("randoms") or []
        front_tree_payloads = hover_trees.get("front") or []
        actual_tree_payload = hover_trees.get("actual")
    if len(random_tree_payloads) != len(random_points):
        random_tree_payloads = []
    if len(front_tree_payloads) != len(front_x):
        front_tree_payloads = []

    post_script = None
    if random_tree_payloads or front_tree_payloads or actual_tree_payload:
        post_script = _tree_hover_script(
            random_tree_payloads,
            actual_tree_payload,
            trees_trace_name="Random trees",
            actual_trace_name="Actual tree",
            trees_title_prefix="Random RSA",
            actual_title="Actual RSA",
            front_trees=front_tree_payloads,
            front_trace_name="Pareto front",
            front_title_prefix="Front RSA",
        )

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(destination),
        include_plotlyjs="directory",
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        post_script=post_script,
    )


def plot_all_3d(
    front_3d, actual_3d, randoms_3d, mrand, srand, prand, save_path
):  # pragma: no cover
    """Plot the 3D Pareto front as a surface with actual plant and random trees.

    Creates a 3D visualization of the Pareto front using triangulated surface
    interpolation. The surface is colored by path coverage (z-axis) to show
    the gradient across the front.

    Data is scaled internally using config.length_scale_factor, matching the
    2D plot_all() behavior. Length (x) and distance (y) are scaled, while
    path coverage (z) is dimensionless and not scaled.

    Args:
        front_3d (dict): A dictionary of total root lengths, total distances to the base and
            path_coverages for each (alpha, beta) value on the front
        actual_3d (tuple): The actual total_root_length, total_travel_distance, and
            total_path_coverage of the original plant
        randoms_3d (list): A list of random tree costs
        mrand (float): The mean total root length of the random trees
        srand (float): The mean total travel distance of the random trees
        prand (float): The mean path coverage of the random trees
        save_path (str): The file path to save the plot
    """
    from matplotlib.tri import Triangulation
    from scipy.spatial import Delaunay

    def scale_data(data):
        """Scale data by user-configured factor."""
        return data * config.length_scale_factor

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Set labels and title with units from config
    ax.set_xlabel(
        f"Total Root Length ({config.length_scale_unit})", fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        f"Travel Distance ({config.length_scale_unit})", fontsize=12, labelpad=10
    )
    ax.set_zlabel("Path Coverage (ratio)", fontsize=12, labelpad=10)
    ax.set_title("3D Pareto Front: Root Architecture Trade-offs", fontsize=15, pad=20)

    logging.debug(f"Front 3D: {len(front_3d)} points")

    # Extract and scale x, y values; z (path coverage) is dimensionless, not scaled
    x_values = np.array([scale_data(x[0]) for x in front_3d.values()])
    y_values = np.array([scale_data(x[1]) for x in front_3d.values()])
    z_values = np.array([x[2] for x in front_3d.values()])

    # Try to create a triangulated surface plot
    surface_plotted = False
    if len(x_values) >= 3:
        try:
            # Create 2D Delaunay triangulation on (x, y) coordinates
            points_2d = np.column_stack((x_values, y_values))
            tri = Delaunay(points_2d)

            # Create matplotlib Triangulation from Delaunay result
            triangulation = Triangulation(x_values, y_values, tri.simplices)

            # Plot the triangulated surface colored by path coverage
            surf = ax.plot_trisurf(
                triangulation,
                z_values,
                cmap="viridis",
                alpha=0.7,
                edgecolor="none",
                linewidth=0,
            )

            # Add colorbar for path coverage
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label("Path Coverage", fontsize=10)

            surface_plotted = True
            logging.debug("Successfully created triangulated surface plot")

        except Exception as e:
            # Triangulation failed (e.g., collinear points)
            logging.warning(f"Triangulation failed, falling back to scatter plot: {e}")

    # Fallback: scatter plot for Pareto front if surface failed or too few points
    if not surface_plotted:
        ax.scatter(
            x_values,
            y_values,
            z_values,
            c=z_values,
            cmap="viridis",
            marker="o",
            s=50,
            alpha=0.8,
            label="Pareto Front",
        )
        logging.info("Using scatter plot for Pareto front (triangulation not possible)")

    # Plot the actual plant (orange X marker) - scale x,y but not z (dimensionless)
    ax.scatter(
        [scale_data(actual_3d[0])],
        [scale_data(actual_3d[1])],
        [actual_3d[2]],
        color="orange",
        marker="X",
        s=150,
        edgecolors="black",
        linewidths=0.5,
        label="Actual Plant",
        zorder=5,
    )

    # Plot the random tree costs (green + markers) - scale x,y but not z
    if len(randoms_3d) > 0:
        randoms_3d_array = np.array(randoms_3d)
        ax.scatter(
            randoms_3d_array[:, 0] * config.length_scale_factor,
            randoms_3d_array[:, 1] * config.length_scale_factor,
            randoms_3d_array[:, 2],
            color="green",
            marker="+",
            s=50,
            alpha=0.6,
            linewidths=1,
            label="Random Trees",
        )

    # Plot the centroid of random trees (red diamond) - scale x,y but not z
    ax.scatter(
        [scale_data(mrand)],
        [scale_data(srand)],
        [prand],
        color="red",
        marker="D",
        s=100,
        edgecolors="black",
        linewidths=0.5,
        label="Random Centroid",
        zorder=5,
    )

    # Add legend
    ax.legend(loc="upper left", fontsize=9)

    # Enable grid
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Save as PNG
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Also save as SVG for better quality
    svg_dest = save_path.with_suffix(".svg")
    plt.savefig(svg_dest, bbox_inches="tight", format="svg")

    plt.close(fig)


def distance_from_front(front, actual_tree):
    """Return the interpolated alpha for the actual tree, and its distance to the front.

    Interpolates between the two closest discrete alpha values on the Pareto front
    for higher precision (based on Matt Platre's implementation).

    Args:
        actual_tree: Tuple of (total_root_length, total_travel_distance)
        front: Dict of form {alpha: [total_root_length, total_travel_distance]}

    Returns:
        Tuple of (interpolated_alpha, scaling_distance) as Python floats
    """
    # For each alpha value, find distance to the actual tree
    distances = {}

    for alpha_value, alpha_tree in front.items():
        # Guard against division by zero (unlikely with real root data)
        if alpha_tree[0] == 0 or alpha_tree[1] == 0:
            continue
        material_ratio = actual_tree[0] / alpha_tree[0]
        transport_ratio = actual_tree[1] / alpha_tree[1]
        distances[alpha_value] = max(material_ratio, transport_ratio)

    # Find the two closest alpha values
    sorted_alphas = sorted(distances.items(), key=lambda x: x[1])
    closest = sorted_alphas[0]
    second_closest = sorted_alphas[1] if len(sorted_alphas) > 1 else closest

    alpha1, dist1 = closest
    alpha2, dist2 = second_closest

    # Linear interpolation between the two closest alphas
    # Closer distance gets higher weight
    if math.isclose(dist1, dist2, rel_tol=1e-9):
        interpolated_alpha = float(alpha1)
    else:
        total_dist = dist1 + dist2
        weight1 = dist2 / total_dist  # Inverse weighting: closer = higher weight
        weight2 = dist1 / total_dist
        interpolated_alpha = float(alpha1 * weight1 + alpha2 * weight2)

    return interpolated_alpha, float(dist1)


def calculate_tradeoff(front, actual_tree):
    """Calculate a tradeoff metric comparing the actual root to optimal architectures.

    Conn et al., 2019 (https://doi.org/10.1371/journal.pcbi.1007325) describe
    a tradeoff feature in which:

        "The numerator quantifies the excess length of the plant compared to the
        optimal minimum length of the Steiner tree. Similarly, the denominator
        quantifies the excess travel distance of the plant compared to the optimal
        minimum travel distance of the Satellite tree. A high value of this feature
        (i.e., a large numerator and small denominator) indicates that the plant
        prioritizes minimizing travel distance; a low trade-off value indicates
        the plant prioritizes minimizing total length."

    That description corresponds to an excess-based metric roughly of the form:
        (actual_length - steiner_length) / (actual_distance - satellite_distance)

    In this implementation we use a related, ratio-based metric inspired by the
    same intuition, defined as a ratio of length-per-distance values:

        Tradeoff = Actual_ratio / Optimal_ratio
                 = (actual_length / actual_distance) / (steiner_length / satellite_distance)

    This represents an alternative way to quantify the prioritization between
    minimizing total root length (material cost) and minimizing travel distance
    (transport efficiency), complementary to the alpha value on the Pareto front.

    - Steiner architecture: Minimizes total root length (material cost)
    - Satellite architecture: Minimizes travel distance (transport efficiency)

    Args:
        front: Dict of {alpha: [total_length, travel_distance]} Pareto front points
        actual_tree: Tuple of (total_root_length, travel_distance)

    Returns:
        Dict with Tradeoff metric and component values:
        - Tradeoff: Ratio of actual to optimal efficiency
        - Steiner_length: Min total root length on Pareto front
        - Steiner_distance: Travel distance of the Steiner architecture
        - Satellite_length: Total root length of the Satellite architecture
        - Satellite_distance: Travel distance of the Satellite architecture
        - Actual_ratio: actual_length / actual_distance
        - Optimal_ratio: steiner_length / satellite_distance
    """
    # Extract Pareto front values
    front_points = list(front.values())  # List of [total_length, travel_distance] pairs

    if len(front_points) < 1:
        print("Warning: Pareto front is empty, cannot calculate Tradeoff")
        return {
            "Tradeoff": None,
            "Steiner_length": None,
            "Steiner_distance": None,
            "Satellite_length": None,
            "Satellite_distance": None,
            "Actual_ratio": None,
            "Optimal_ratio": None,
        }

    # Find Steiner architecture (minimizes total root length)
    steiner_point = min(front_points, key=lambda x: x[0])
    steiner_length, steiner_distance = steiner_point

    # Find Satellite architecture (minimizes travel distance)
    satellite_point = min(front_points, key=lambda x: x[1])
    satellite_length, satellite_distance = satellite_point

    # Actual tree values
    actual_length, actual_distance = actual_tree

    # Calculate ratios - handle division by zero
    if actual_distance == 0:
        actual_ratio = None
        print("Warning: Actual travel distance is 0, actual_ratio set to None")
    else:
        actual_ratio = float(actual_length / actual_distance)

    if satellite_distance == 0:
        optimal_ratio = None
        print("Warning: Satellite travel distance is 0, optimal_ratio set to None")
    else:
        optimal_ratio = float(steiner_length / satellite_distance)

    # Calculate tradeoff
    if optimal_ratio is None or optimal_ratio == 0 or actual_ratio is None:
        tradeoff = None
        print("Warning: Could not calculate tradeoff due to invalid ratios")
    else:
        tradeoff = float(actual_ratio / optimal_ratio)

    return {
        "Tradeoff": tradeoff,
        "Steiner_length": float(steiner_length),
        "Steiner_distance": float(steiner_distance),
        "Satellite_length": float(satellite_length),
        "Satellite_distance": float(satellite_distance),
        "Actual_ratio": actual_ratio,
        "Optimal_ratio": optimal_ratio,
    }


def distance_from_front_3d(front, actual_tree):
    """Compute the epsilon indicator distance from a tree to the 3D Pareto front.

    Uses the multiplicative ε-indicator, a standard metric in multi-objective
    optimization (Zitzler et al., 2003). The epsilon value represents the
    minimum scaling factor needed to make a Pareto-optimal tree dominate the
    actual tree in all objectives.

    Implements barycentric interpolation with the 3 nearest (α, β) points for
    higher precision, matching the 2D distance_from_front() approach.

    Formula: ε = min_{r ∈ Front} max_i (actual_i / front_i)

    Interpretation:
        - ε = 1.0: Tree is on the Pareto front
        - ε > 1.0: Tree is suboptimal; ε is the scaling factor
        - ε < 1.0: Tree dominates the front (should not happen)

    References:
        Zitzler et al. (2003), IEEE Trans. Evol. Comput. 7(2), 117-132
        Chandrasekhar & Navlakha (2019), Proc. Royal Society B, Supplementary Eq. 2

    Args:
        front: Dict mapping (alpha, beta) tuples to [length, distance, tortuosity] lists
        actual_tree: Tuple of (total_root_length, total_travel_distance, path_tortuosity)

    Returns:
        Dict with keys:
            - epsilon: float - multiplicative ε-indicator (scaling factor)
            - alpha: float - interpolated alpha parameter
            - beta: float - interpolated beta parameter
            - gamma: float - computed as 1 - alpha - beta
            - epsilon_components: dict with material/transport/coverage ratios
            - corner_costs: dict with steiner/satellite/coverage corner values
    """
    # For each (alpha, beta) value, find distance to the actual tree
    distances = {}
    ratios = {}  # Store component ratios for each point

    for alpha_beta_value, alpha_beta_tree in front.items():
        # Division-by-zero guard: skip points with any zero cost dimension
        if any(v == 0 for v in alpha_beta_tree):
            continue

        material_ratio = actual_tree[0] / alpha_beta_tree[0]
        transport_ratio = actual_tree[1] / alpha_beta_tree[1]
        path_coverage_ratio = actual_tree[2] / alpha_beta_tree[2]

        epsilon = max(material_ratio, transport_ratio, path_coverage_ratio)
        distances[alpha_beta_value] = epsilon
        ratios[alpha_beta_value] = (
            material_ratio,
            transport_ratio,
            path_coverage_ratio,
        )

    # Handle edge case: no valid points
    if not distances:
        # Fall back to returning default values if all points have zeros
        return {
            "epsilon": float("inf"),
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 1.0,
            "epsilon_components": {
                "material": float("inf"),
                "transport": float("inf"),
                "coverage": float("inf"),
            },
            "corner_costs": {
                "steiner": (0.0, 0.0, 0.0),
                "satellite": (0.0, 0.0, 0.0),
                "coverage": (0.0, 0.0, 0.0),
            },
        }

    # Find the 3 nearest (alpha, beta) points by epsilon distance
    sorted_points = sorted(distances.items(), key=lambda x: x[1])

    # Get up to 3 closest points
    n_points = min(3, len(sorted_points))
    closest_points = sorted_points[:n_points]

    # Extract closest point's epsilon and ratios for the result
    closest_ab, closest_epsilon = closest_points[0]
    closest_ratios = ratios[closest_ab]

    # Barycentric interpolation using inverse-distance weighting
    if n_points == 1:
        # Single point: use directly
        alpha = float(closest_ab[0])
        beta = float(closest_ab[1])
    else:
        # Inverse distance weighting for 2 or 3 points
        # Add small epsilon to avoid division by zero when distance is exactly 0
        eps_guard = 1e-10

        weights = []
        for ab, dist in closest_points:
            weights.append(1.0 / (dist + eps_guard))

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Interpolate alpha and beta
        alpha = sum(w * ab[0] for w, (ab, _) in zip(normalized_weights, closest_points))
        beta = sum(w * ab[1] for w, (ab, _) in zip(normalized_weights, closest_points))

        # Clamp to valid range (α ≥ 0, β ≥ 0, α + β ≤ 1)
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0 - alpha, beta))

    # Compute gamma
    gamma = 1.0 - alpha - beta

    # Look up corner costs from the front
    # Steiner: α=1, β=0, γ=0 (minimizes length)
    # Satellite: α=0, β=1, γ=0 (minimizes distance)
    # Coverage: α=0, β=0, γ=1 (minimizes tortuosity)
    steiner_key = (1.0, 0.0)
    satellite_key = (0.0, 1.0)
    coverage_key = (0.0, 0.0)

    steiner_costs = front.get(steiner_key, [0.0, 0.0, 0.0])
    satellite_costs = front.get(satellite_key, [0.0, 0.0, 0.0])
    coverage_costs = front.get(coverage_key, [0.0, 0.0, 0.0])

    return {
        "epsilon": float(closest_epsilon),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "epsilon_components": {
            "material": float(closest_ratios[0]),
            "transport": float(closest_ratios[1]),
            "coverage": float(closest_ratios[2]),
        },
        "corner_costs": {
            "steiner": tuple(float(v) for v in steiner_costs),
            "satellite": tuple(float(v) for v in satellite_costs),
            "coverage": tuple(float(v) for v in coverage_costs),
        },
    }


DEFAULT_N_RANDOM = 100

REGION_CLASSIFICATION_FIELDS = (
    "mean_rounds_normalized",
    "ci_low_normalized",
    "ci_high_normalized",
    "local_mean_alpha",
    "local_alpha_ci_low",
    "local_alpha_ci_high",
)


def pareto_calcs(H, n_random=DEFAULT_N_RANDOM, tree_payloads=None):
    """Perform Pareto-related calculations.

    The analytical tip-preserving Steiner front is computed with the rigid
    greedy generator. The analytical randomized Steiner baseline requires an
    undirected graph. A converted copy is used so the caller's graph is not
    modified.

    Args:
        H: NetworkX graph representing the root system (directed or undirected).
        n_random: Number of analytical random trees used for the baseline.
        tree_payloads: Optional mutable dict. When provided, it is filled with
            compact ``"front"``, ``"actual"``, and ``"randoms"`` tree payloads
            for HTML hover panels. It does not affect the CSV results.

    Returns:
        results, front, randoms
    """
    H_pareto = H.to_undirected()

    front_tree_payloads = {} if tree_payloads is not None else None

    front, actual = pareto_front(
        H_pareto,
        tree_payloads=front_tree_payloads,
    )

    if tree_payloads is not None:
        ordered_keys = sorted(front, key=lambda alpha: front[alpha][0])
        tree_payloads["front"] = [
            front_tree_payloads[alpha]
            for alpha in ordered_keys
            if alpha in front_tree_payloads
        ]
    mactual, sactual = actual

    # Calculate tradeoff metrics
    tradeoff_info = calculate_tradeoff(front, actual)

    plant_alpha, plant_scaling = distance_from_front(front, actual)

    randoms, random_trees = random_steiner_wrapper_analytical(
        H_pareto, n_random, optimization_rounds=0
    )

    if tree_payloads is not None:
        tree_payloads["actual"] = _tree_hover_payload(
            straighten_tree(H_pareto), None, None
        )
        tree_payloads["randoms"] = [
            _tree_hover_payload(tree, None, None)
            for tree in random_trees
        ]

    # centroid of randoms
    if randoms:
        mrand = float(np.mean([x[0] for x in randoms]))
        srand = float(np.mean([x[1] for x in randoms]))
        rand_alpha, rand_scaling = distance_from_front(front, (mrand, srand))
    else:
        mrand = float("nan")
        srand = float("nan")
        rand_alpha = float("nan")
        rand_scaling = float("nan")

    # assemble dict for export
    results = {
        "Total root length": float(mactual),
        "Travel distance": float(sactual),
        "alpha": float(plant_alpha),
        "scaling distance to front": float(plant_scaling),
        "Total root length (random)": mrand,
        "Travel distance (random)": srand,
        "alpha (random)": float(rand_alpha),
        "scaling (random)": float(rand_scaling),
    }

    # Merge tradeoff metrics into results
    results.update(tradeoff_info)

    return results, front, randoms


def classify_region(
    H,
    front,
    actual,
    n_samples=5000,
    region_factor=0.5,
    save_path=None,
    save_html_path=None,
    scale_factor=1.0,
    unit="",
    progress_callback=None,
    verbose=False,
):
    """Classify a root by its local optimization level and trade-off direction.

    Thin UI-oriented wrapper around
    ``exploration.estimate_rounds_for_region``. It reuses the Pareto front and
    straightened-tree reference already computed by :func:`pareto_calcs`,
    optionally saves the sampling plot, and returns only the dimensionless
    fields that belong in the CSV report.

    Args:
        H: NetworkX graph representing the root system.
        front: Pareto front dict from :func:`pareto_calcs`.
        actual: Straightened-tree (wiring, travel) reference from
            :func:`pareto_calcs`.
        n_samples: Number of semi-random architectures to sample.
        region_factor: Ellipse semi-axis scale as a fraction of the actual
            tree's scaling distance to the front.
        save_path: Optional path for the static Matplotlib sampling plot.
        save_html_path: Optional path for the interactive Plotly sampling plot.
        scale_factor: Length scale factor used only for plot axes.
        unit: Length unit label used in the interactive plot.
        progress_callback: Optional callable(done, total, n_inside) invoked
            during semi-random tree generation.
        verbose: Print progress from the sampling routine.

    Returns:
        dict with the six region-classification CSV fields.
    """
    sampling = estimate_rounds_for_region(
        H.to_undirected(),
        n_samples=n_samples,
        region_factor=region_factor,
        plot=save_path is not None,
        show=False,
        save_path=save_path,
        front=front,
        actual=actual,
        scale_factor=scale_factor,
        progress_callback=progress_callback,
        verbose=verbose,
        collect_trees=save_html_path is not None,
    )

    if save_html_path is not None:
        save_sampling_html(
            sampling,
            save_html_path,
            scale_factor=scale_factor,
            region_factor=region_factor,
            unit=unit,
        )

    def as_float(value):
        return None if value is None else float(value)

    return {
        "mean_rounds_normalized": as_float(sampling["mean_rounds_normalized"]),
        "ci_low_normalized": as_float(sampling["ci_low_normalized"]),
        "ci_high_normalized": as_float(sampling["ci_high_normalized"]),
        "local_mean_alpha": as_float(sampling["mean_alpha"]),
        "local_alpha_ci_low": as_float(sampling["alpha_ci_low"]),
        "local_alpha_ci_high": as_float(sampling["alpha_ci_high"]),
    }


def pareto_calcs_3d_path_tortuosity(H):
    """Perform Pareto-related calculations using 3D Pareto Front with path tortuosity.

    Uses the multiplicative ε-indicator to measure distance from the Pareto front.
    Returns interpolated (α, β, γ) parameters and epsilon components.

    Args:
        H (nx.Graph): NetworkX graph representing the root system.

    Returns:
        tuple: (results_dict, front, randoms) where results_dict contains:
            - Total root length, Travel distance, Path tortuosity (actual tree)
            - alpha_3d, beta_3d, gamma_3d (interpolated Pareto weights)
            - epsilon_3d (multiplicative ε-indicator)
            - epsilon_3d_material/transport/coverage (ratio components)
            - Same fields for random trees with "(random)" suffix
            - Corner costs: Steiner_*_3d, Satellite_*_3d, Coverage_*_3d
    """
    # Calculate the Pareto front using the 3D path tortuosity
    front, actual = pareto_front_3d_path_tortuosity(H)
    # Extract the actual tree values
    # mactual is the total root length, sactual is the total travel distance, and pactual is the path tortuosity
    mactual, sactual, pactual = actual

    # Calculate the epsilon indicator and interpolated (alpha, beta, gamma)
    plant_result = distance_from_front_3d(front, actual)

    # Generate random trees
    randoms = random_tree_3d_path_tortuosity(H)

    # Calculate the mean costs of the random trees
    mrand = float(np.mean([x[0] for x in randoms]))
    srand = float(np.mean([x[1] for x in randoms]))
    prand = float(np.mean([x[2] for x in randoms]))

    # Calculate epsilon indicator for the random tree centroid
    rand_result = distance_from_front_3d(front, (mrand, srand, prand))

    # Extract corner costs from plant_result (same for both calls)
    corner_costs = plant_result["corner_costs"]

    # Assemble the results dictionary with _3d suffix for 3D-specific fields
    results = {
        # Actual tree measurements (same names as 2D for shared fields)
        "Total root length": float(mactual),
        "Travel distance": float(sactual),
        "Path tortuosity": float(pactual),
        # Interpolated Pareto parameters
        "alpha_3d": plant_result["alpha"],
        "beta_3d": plant_result["beta"],
        "gamma_3d": plant_result["gamma"],
        # Epsilon indicator (distance from front)
        "epsilon_3d": plant_result["epsilon"],
        # Epsilon components (which dimension drives epsilon)
        "epsilon_3d_material": plant_result["epsilon_components"]["material"],
        "epsilon_3d_transport": plant_result["epsilon_components"]["transport"],
        "epsilon_3d_coverage": plant_result["epsilon_components"]["coverage"],
        # Random tree measurements
        "Total root length (random)": mrand,
        "Travel distance (random)": srand,
        "Path tortuosity (random)": prand,
        # Random tree Pareto parameters
        "alpha_3d (random)": rand_result["alpha"],
        "beta_3d (random)": rand_result["beta"],
        "gamma_3d (random)": rand_result["gamma"],
        "epsilon_3d (random)": rand_result["epsilon"],
        # Random tree epsilon components
        "epsilon_3d_material (random)": rand_result["epsilon_components"]["material"],
        "epsilon_3d_transport (random)": rand_result["epsilon_components"]["transport"],
        "epsilon_3d_coverage (random)": rand_result["epsilon_components"]["coverage"],
        # Corner architecture reference costs
        "Steiner_length_3d": corner_costs["steiner"][0],
        "Steiner_distance_3d": corner_costs["steiner"][1],
        "Steiner_tortuosity_3d": corner_costs["steiner"][2],
        "Satellite_length_3d": corner_costs["satellite"][0],
        "Satellite_distance_3d": corner_costs["satellite"][1],
        "Satellite_tortuosity_3d": corner_costs["satellite"][2],
        "Coverage_length_3d": corner_costs["coverage"][0],
        "Coverage_distance_3d": corner_costs["coverage"][1],
        "Coverage_tortuosity_3d": corner_costs["coverage"][2],
    }

    return results, front, randoms


### CONVEX HULL calculations






def distance(pos1, pos2):
    """Calculate the Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calc_zones(G, root_node):
    """Calculate the Branched Zone, Basal Zone, and Apical Zone lengths along the primary root."""
    # Perform BFS to find nodes along the primary root
    bfs_paths = dict(nx.bfs_successors(G, root_node))

    # Collect primary root nodes
    PR_nodes = []
    for node, children in bfs_paths.items():
        if G.nodes[node].get("LR_index") is None:  # Primary root node
            PR_nodes.append(node)
            for child in children:
                if G.nodes[child].get("LR_index") is None:
                    PR_nodes.append(child)

    # Ensure PR_nodes are in order along the root path
    PR_nodes = list(dict.fromkeys(PR_nodes))

    # Identify the first and last lateral root insertion points
    first_lr_insertion_point = None
    last_lr_insertion_point = None

    for node in PR_nodes:
        neighbors = list(G.neighbors(node))
        if any(G.nodes[neighbor].get("LR_index") is not None for neighbor in neighbors):
            if first_lr_insertion_point is None:
                first_lr_insertion_point = node
            last_lr_insertion_point = node

    # Calculate zone lengths
    branched_zone_length = 0
    basal_zone_length = 0
    apical_zone_length = 0

    found_first = False
    found_last = False

    for prev, current in zip(PR_nodes, PR_nodes[1:]):
        segment_length = distance(G.nodes[prev]["pos"], G.nodes[current]["pos"])

        if current == first_lr_insertion_point:
            found_first = True
        if current == last_lr_insertion_point:
            found_last = True

        if found_first and not found_last:
            branched_zone_length += segment_length
        elif not found_first:
            basal_zone_length += segment_length
        elif found_last:
            apical_zone_length += segment_length

    return {
        "branched_zone_length": branched_zone_length,
        "basal_zone_length": basal_zone_length,
        "apical_zone_length": apical_zone_length,
    }


def calc_len_LRs_with_distances(H):
    """Calculate the 2D Euclidean distance for each lateral root from the first node to the last node, excluding intermediate nodes, and return the total length of each LR type in the graph."""
    # minimum length (px) for LR to be considered part of the network

    threshold = 0

    # dict of node ids : LR index, for each LR node
    idxs = nx.get_node_attributes(H, "LR_index")
    idxs = {k: v for k, v in idxs.items() if v is not None}  # drop empty (PR) nodes

    num_LRs = max(idxs.values()) + 1
    min_num_LRs = min(idxs.values())  # should be 1

    results = {}

    for i in range(min_num_LRs, num_LRs):
        # gather nodes corresponding to the current LR index
        selected = []

        for node in H.nodes(data="LR_index"):
            if node[1] == i:
                selected.append(node[0])
                # make note of the root degree (should be the same for all nodes in loop)
                current_degree = H.nodes[node[0]]["root_deg"]

        # to find the shallowest node in LR, we iterate through them
        # until we find the one whose parent_node has a lesser root degree
        sub = H.subgraph(selected)
        for node in sub.nodes():
            parent_node = list(H.predecessors(node))
            # print(node, parent_node)
            assert len(parent_node) == 1  # should be a tree
            if H.nodes[parent_node[0]]["root_deg"] < current_degree:
                sub_top = node

        # now we can DFS to order all nodes by increasing depth
        ordered = list(nx.dfs_tree(sub, sub_top).nodes())

        # also include the parent_node of the shallowest node in the LR (the 'branch point')
        parent_node = list(H.predecessors(ordered[0]))
        assert len(parent_node) == 1

        nodes_list = parent_node + ordered

        # Calculate the distance for the LR
        length = calc_root_len(H, nodes_list)

        # Exclude intermediate nodes
        if length < threshold:
            H.remove_nodes_from(ordered)
        else:
            # Now we can calculate the Euclidean distance from the first node to the last node
            # excluding intermediate nodes
            first_node_pos = H.nodes[nodes_list[0]]["pos"]
            last_node_pos = H.nodes[nodes_list[-1]]["pos"]
            distance_lr = distance(first_node_pos, last_node_pos)

            results[i] = [length, distance_lr]

    assert nx.is_tree(H)
    return results


def find_lowermost_node_of_primary_root(G, root_node):
    """Find the lowermost node of the primary root."""
    descendants = nx.descendants(G, root_node)
    lowermost_node = max(
        descendants, key=lambda node: G.nodes[node]["pos"][1]
    )  # Find the node with the maximum y-coordinate
    return G.nodes[lowermost_node]["pos"]


def calculate_distance(p1, p2):
    """Compute 2D Euclidean distance between two (x,y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def analyze(G, enable_3d=True, n_random=DEFAULT_N_RANDOM, tree_payloads=None):
    """Report basic root metrics for a given graph.

    Args:
        G: NetworkX graph representing the root system.
        enable_3d: If True, compute 3D Pareto analysis (slower).
                   If False, return empty 3D results.
        n_random: Number of analytical random trees used for the 2D
                  random baseline.
        tree_payloads: Optional mutable dict filled with compact actual/random
                  tree payloads for the interactive Pareto HTML hover panel.

    Returns:
        Tuple of (results, front, randoms, results_3d, front_3d, randoms_3d).
        When enable_3d=False, results_3d, front_3d, randoms_3d are empty.
    """
    # check that graph is indeed a tree (acyclic, undirected, connected)
    assert nx.is_tree(G)

    # independent deep copy of G, with LRs below threshold excluded
    H = copy.deepcopy(G)

    # find top ("root") node
    for node in H.nodes(data="pos"):
        if node[1] == [0, 0]:
            root_node = node[0]
    # the pareto functions are hardcoded to assume node 0 is the top.

    assert root_node == 0

    # PR len
    len_PR = calc_len_PR(H, root_node)
    # print('PR length is:', len_PR)

    # LR len/number
    LR_info = calc_len_LRs(H)
    num_LRs = len(LR_info)
    lens_LRs = [x[0] for x in LR_info.values()]
    angles_LRs = [x[1] for x in LR_info.values()]
    # print('LR lengths are:', lens_LRs)
    # print('Set point angles are:', angles_LRs)

    # primary LR density
    density_LRs = num_LRs / len_PR
    # print('LR density is:', num_LRs/len_PR)

    # Calculate the Euclidean distance between the uppermost node and the lowermost node of the primary root
    uppermost_node_pos = H.nodes[root_node]["pos"]
    lowermost_node_pos = find_lowermost_node_of_primary_root(H, root_node)
    distance_root = calculate_distance(uppermost_node_pos, lowermost_node_pos)

    results, front, randoms = pareto_calcs(
        H, n_random=n_random, tree_payloads=tree_payloads
    )

    # 3D Pareto analysis is optional (slower: 10,201 iterations vs 101 for 2D)
    if enable_3d:
        results_3d, front_3d, randoms_3d = pareto_calcs_3d_path_tortuosity(H)
    else:
        results_3d, front_3d, randoms_3d = {}, {}, []

    # Calculate lateral root distances with lengths and first-to-last distances
    lateral_root_info = calc_len_LRs_with_distances(H)
    num_LRs = len(lateral_root_info)

    # Extract lengths and distances
    lens_LRs = [info[0] for info in lateral_root_info.values()]
    distances_LRs = [info[1] for info in lateral_root_info.values()]

    # Convex Hull calculations
    points = np.array([H.nodes[node]["pos"] for node in H.nodes()])
    hull = ConvexHull(points)

    # Barycenter (centroid) of the Convex Hull
    # Centroid formula: (mean x, mean y) of the vertices of the convex hull
    hull_points = points[hull.vertices]
    barycenter_x = np.mean(hull_points[:, 0])
    barycenter_y = np.mean(hull_points[:, 1])
    barycenter = (barycenter_x, barycenter_y)

    # Find the uppermost node (node with the minimum y-coordinate)
    uppermost_node = min(H.nodes(data="pos"), key=lambda node: node[1][1])
    uppermost_node_pos = uppermost_node[1]

    # Build quadrilateral (barycenter and uppermost node form the quadrilateral)
    barycenter_y_displacement = abs(
        barycenter_y - uppermost_node_pos[1]
    )  # Displacement in y-direction
    barycenter_x_displacement = abs(
        barycenter_x - uppermost_node_pos[0]
    )  # Displacement in x-direction

    # Calculate Branched, Basal, and Apical Zones
    zone_lengths = calc_zones(H, root_node)
    branched_zone_length = zone_lengths["branched_zone_length"]
    basal_zone_length = zone_lengths["basal_zone_length"]
    apical_zone_length = zone_lengths["apical_zone_length"]

    # Ensure basal zone length is 0 if it equals the primary root length
    if basal_zone_length == len_PR:
        basal_zone_length = 0

    # Branched Zone density

    branched_zone_density = (
        num_LRs / branched_zone_length if branched_zone_length != 0 else 0
    )

    # Calculate mean and median (convert to Python floats for clean CSV serialization)
    mean_LR_lengths = float(np.mean(lens_LRs))
    median_LR_lengths = float(np.median(lens_LRs))
    median_LR_angles = float(np.median(angles_LRs))
    mean_LR_angles = float(np.mean(angles_LRs))
    mean_LR_distances = float(np.mean(distances_LRs))
    median_LR_distances = float(np.median(distances_LRs))
    sum_LR_distances = float(np.sum(distances_LRs))

    # Calculate the total distance (sum of LR distances and PR minimal distance)
    total_distance = sum_LR_distances + distance_root

    # Add lateral root lengths and distances to the results dictionary
    results["PR length"] = len_PR
    results["PR_minimal_length"] = distance_root
    results["Basal Zone length"] = basal_zone_length
    results["Branched Zone length"] = branched_zone_length
    results["Apical Zone length"] = apical_zone_length
    results["Mean LR lengths"] = mean_LR_lengths
    results["Mean LR minimal lengths"] = mean_LR_distances
    results["Median LR lengths"] = median_LR_lengths
    results["Median LR minimal lengths"] = median_LR_distances
    results["sum LR minimal lengths"] = sum_LR_distances
    results["Mean LR angles"] = mean_LR_angles
    results["Median LR angles"] = median_LR_angles
    results["LR count"] = num_LRs
    results["LR density"] = density_LRs
    results["Branched Zone density"] = branched_zone_density
    # Convert list elements to Python floats for clean CSV serialization
    results["LR lengths"] = [float(x) for x in lens_LRs]
    results["LR angles"] = [float(x) for x in angles_LRs]
    results["LR minimal lengths"] = [float(x) for x in distances_LRs]
    results["Barycenter x displacement"] = float(barycenter_x_displacement)
    results["Barycenter y displacement"] = float(barycenter_y_displacement)
    results["Total minimal Distance"] = (
        total_distance  # Add the total distance to the results
    )

    # Calculate the material cost (total root length)
    Total_root_length = len_PR + sum(lens_LRs)

    # Calculate the ratio of the material cost with the Total minimal Distance
    material_distance_ratio = Total_root_length / total_distance

    results["Tortuosity"] = material_distance_ratio

    # Calculating convex hull area
    points = np.array([H.nodes[node]["pos"] for node in H.nodes()])
    hull = ConvexHull(points)
    convex_hull_area = hull.volume  # Convex hull area in 2D is the same as its volume

    results["Convex Hull Area"] = float(convex_hull_area)

    return results, front, randoms, results_3d, front_3d, randoms_3d

def _optimization_colorscale(n_colors=256):
    """Return the HSV color scale used for the optimization-level colorbar."""
    hue_values = np.linspace(0, 0.67, n_colors)
    sat_values = np.ones(n_colors) * 0.8
    val_values = np.ones(n_colors) * 0.9
    hsv = np.stack([hue_values, sat_values, val_values], axis=1)
    rgb = hsv_to_rgb(hsv)
    return [[i / (n_colors - 1), to_hex(rgb[i])] for i in range(n_colors)]


def _colorbar_marker_script(mean_norm):
    """Return JavaScript that draws the mean-optimization marker on the colorbar."""
    mean_json = "null" if mean_norm is None else repr(float(mean_norm))

    script = r"""
(function () {
  const gd = document.getElementById("{plot_id}");
  if (!gd) return;

  const meanOptimization = __MEAN__;

  // ------------------------------------------------------------------
  // Optimization-level marker, drawn as an SVG overlay on the actual
  // colorbar rectangle. It is appended to the main SVG so it stays in
  // front of the colorbar, and its width is measured from the bar.
  // ------------------------------------------------------------------
  function drawColorbarMarker() {
    const colorbar = gd.querySelector(".colorbar");
    if (!colorbar) return;
    const main = colorbar.ownerSVGElement || gd.querySelector(".main-svg");
    if (!main) return;

    const old = gd.querySelector(".rsa-opt-marker");
    if (old) old.remove();
    if (meanOptimization === null || meanOptimization === undefined) return;

    const background =
      colorbar.querySelector("rect.cboutline") ||
      colorbar.querySelector("rect.cbbg") ||
      colorbar.querySelector("rect");
    if (!background) return;

    const rect = background.getBoundingClientRect();
    const matrix = main.getScreenCTM();
    if (!rect.width || !rect.height || !matrix) return;

    const inverse = matrix.inverse();
    const yPixel = rect.top + (1 - meanOptimization) * rect.height;
    const start = main.createSVGPoint();
    const end = main.createSVGPoint();
    start.x = rect.left;
    start.y = yPixel;
    end.x = rect.left + rect.width;
    end.y = yPixel;
    const a = start.matrixTransform(inverse);
    const b = end.matrixTransform(inverse);

    const line = document.createElementNS(
      "http://www.w3.org/2000/svg", "line"
    );
    line.setAttribute("class", "rsa-opt-marker");
    line.setAttribute("x1", a.x);
    line.setAttribute("x2", b.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("y2", b.y);
    line.setAttribute("stroke", "black");
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-linecap", "butt");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    line.setAttribute("pointer-events", "none");
    main.appendChild(line);
  }

  function scheduleColorbarMarker() {
    window.requestAnimationFrame(drawColorbarMarker);
  }

  gd.on("plotly_afterplot", scheduleColorbarMarker);
  window.addEventListener("resize", scheduleColorbarMarker);
  scheduleColorbarMarker();
})();
"""
    return script.replace("__MEAN__", mean_json)


def _tree_hover_script(
    trees,
    actual_tree,
    trees_trace_name,
    actual_trace_name,
    trees_title_prefix,
    actual_title,
    front_trees=None,
    front_trace_name="Pareto front",
    front_title_prefix="Front RSA",
):
    """Return JavaScript that draws tree payloads in a hover panel."""
    tree_json = json.dumps(trees, separators=(",", ":"))
    actual_tree_json = "null" if actual_tree is None else json.dumps(
        actual_tree, separators=(",", ":")
    )
    front_trees_json = json.dumps(front_trees or [], separators=(",", ":"))
    trees_trace_json = json.dumps(trees_trace_name)
    actual_trace_json = json.dumps(actual_trace_name)
    front_trace_json = json.dumps(front_trace_name)
    trees_title_json = json.dumps(trees_title_prefix)
    actual_title_json = json.dumps(actual_title)
    front_title_json = json.dumps(front_title_prefix)

    script = r"""
(function () {
  // {plot_id} is replaced by Plotly with the id of the graph div.
  const gd = document.getElementById("{plot_id}");
  if (!gd) return;

  const trees = __TREES__;
  const actualTree = __ACTUAL_TREE__;
  const frontTrees = __FRONT_TREES__;
  const treesTraceName = __TREES_TRACE__;
  const actualTraceName = __ACTUAL_TRACE__;
  const frontTraceName = __FRONT_TRACE__;
  const treesTitlePrefix = __TREES_TITLE__;
  const actualTitle = __ACTUAL_TITLE__;
  const frontTitlePrefix = __FRONT_TITLE__;

  // ------------------------------------------------------------------
  // Left-side tree panel, shown alongside Plotly's normal hover box.
  // The drawing mirrors visualisation.draw_tree_simple: brown branches,
  // green base, red tips, inverted y axis and equal aspect.
  // ------------------------------------------------------------------
  if (!trees.length && !frontTrees.length && !actualTree) return;

  const style = document.createElement("style");
  style.textContent = `
    #rsa-hover-tree {
      position: fixed;
      display: none;
      z-index: 10000;
      width: 180px;
      box-sizing: border-box;
      padding: 2px 2px 2px 2px;
      background: rgba(255, 255, 255, 0.98);
      border: 1px solid #777;
      border-radius: 6px;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
      pointer-events: none;
      font-family: "Open Sans", verdana, arial, sans-serif;
    }
    #rsa-hover-tree-title {
      font-size: 13px;
      font-weight: 600;
      color: #222;
      line-height: 1.25;
      white-space: nowrap;
    }
    #rsa-hover-tree-meta {
      margin-top: 2px;
      font-size: 11px;
      color: #555;
      line-height: 1.25;
      white-space: nowrap;
    }
    #rsa-hover-tree-svg {
      display: block;
      width: 100%;
      height: 245px;
      margin-top: 1px;
    }
  `;
  document.head.appendChild(style);

  const panel = document.createElement("div");
  panel.id = "rsa-hover-tree";
  panel.innerHTML = `
    <div id="rsa-hover-tree-title"></div>
    <div id="rsa-hover-tree-meta"></div>
    <svg id="rsa-hover-tree-svg"
         preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg"></svg>
  `;
  document.body.appendChild(panel);

  const title = panel.querySelector("#rsa-hover-tree-title");
  const meta = panel.querySelector("#rsa-hover-tree-meta");
  const svg = panel.querySelector("#rsa-hover-tree-svg");
  const svgNS = "http://www.w3.org/2000/svg";

  function hideTreePanel() {
    panel.style.display = "none";
  }

  function sizePanelToTitle() {
    const range = document.createRange();
    range.selectNodeContents(title);
    const textWidth = Math.ceil(range.getBoundingClientRect().width);
    // The frame is title-width plus a small margin. Padding is intentionally
    // tiny, so most of the width is available to the tree drawing.
    panel.style.width = `${Math.max(120, textWidth + 22)}px`;
  }

  function drawTree(payload, titleText) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!payload || !payload.p || !payload.p.length) return;

    title.textContent = titleText;
    sizePanelToTitle();

    const points = payload.p;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const point of points) {
      minX = Math.min(minX, point[0]);
      maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]);
      maxY = Math.max(maxY, point[1]);
    }

    const xSpan = Math.max(maxX - minX, 1e-6);
    const ySpan = Math.max(maxY - minY, 1e-6);
    // Small padding on each axis: the tree should fill the frame, while
    // keeping the same equal-aspect projection as draw_tree_simple().
    const padX = xSpan * 0.02;
    const padY = ySpan * 0.02;
    const viewWidth = xSpan + 2 * padX;
    const viewHeight = ySpan + 2 * padY;
    svg.setAttribute(
      "viewBox",
      `${minX - padX} ${minY - padY} ${viewWidth} ${viewHeight}`
    );

    const edges = payload.e || [];
    const degree = new Array(points.length).fill(0);

    for (const edge of edges) {
      const a = edge[0], b = edge[1];
      if (!points[a] || !points[b]) continue;
      degree[a] += 1;
      degree[b] += 1;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", points[a][0]);
      line.setAttribute("y1", points[a][1]);
      line.setAttribute("x2", points[b][0]);
      line.setAttribute("y2", points[b][1]);
      line.setAttribute("stroke", "#3D2B1F");
      line.setAttribute("stroke-width", "1.2");
      line.setAttribute("stroke-opacity", "0.7");
      line.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(line);
    }

    const bounds = svg.getBoundingClientRect();
    const scaleX = bounds.width ? bounds.width / viewWidth : 1;
    const scaleY = bounds.height ? bounds.height / viewHeight : 1;
    const uniformScale = Math.min(scaleX, scaleY) || 1;
    const pixelScale = 1 / uniformScale;
    const baseIndex = payload.b === undefined ? 0 : payload.b;

    // Tips: degree-one nodes other than the base.
    for (let i = 0; i < points.length; i += 1) {
      if (i === baseIndex || degree[i] !== 1) continue;
      const tip = document.createElementNS(svgNS, "circle");
      tip.setAttribute("cx", points[i][0]);
      tip.setAttribute("cy", points[i][1]);
      tip.setAttribute("r", 2.3 * pixelScale);
      tip.setAttribute("fill", "#FF0000");
      tip.setAttribute("stroke", "black");
      tip.setAttribute("stroke-width", "0.4");
      tip.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(tip);
    }

    const base = document.createElementNS(svgNS, "circle");
    base.setAttribute("cx", points[baseIndex][0]);
    base.setAttribute("cy", points[baseIndex][1]);
    base.setAttribute("r", 4.5 * pixelScale);
    base.setAttribute("fill", "#2E8B57");
    base.setAttribute("stroke", "black");
    base.setAttribute("stroke-width", "0.5");
    base.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(base);

    const details = [];
    if (typeof payload.a === "number" && Number.isFinite(payload.a)) {
      details.push(`α = ${payload.a.toFixed(3)}`);
    }
    if (typeof payload.r === "number" && Number.isFinite(payload.r)) {
      details.push(`rounds = ${payload.r}`);
    }
    meta.textContent = details.join(" · ");
  }

  function positionTreePanel(event) {
    const source = event && event.event ? event.event : null;
    const width = panel.offsetWidth;
    const height = panel.offsetHeight;
    const gap = 20;
    const margin = 10;

    if (!source || typeof source.clientX !== "number") {
      panel.style.left = `${margin}px`;
      panel.style.top = "70px";
      return;
    }

    let left = source.clientX - width - gap;
    if (left < margin) {
      left = Math.min(
        source.clientX + gap,
        window.innerWidth - width - margin
      );
    }
    left = Math.max(margin, left);

    let top = source.clientY - height / 2;
    top = Math.max(
      margin,
      Math.min(top, window.innerHeight - height - margin)
    );

    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  }

  gd.on("plotly_hover", function (event) {
    const point = event.points && event.points[0];
    if (!point || !point.data) {
      hideTreePanel();
      return;
    }

    let payload = null;
    let titleText = "";
    if (point.data.name === treesTraceName) {
      const index = point.pointNumber;
      payload = trees[index];
      titleText = `${treesTitlePrefix} #${index + 1}`;
    } else if (point.data.name === frontTraceName) {
      const index = point.pointNumber;
      payload = frontTrees[index];
      titleText = `${frontTitlePrefix} #${index + 1}`;
    } else if (point.data.name === actualTraceName) {
      payload = actualTree;
      titleText = actualTitle;
    }

    if (!payload) {
      hideTreePanel();
      return;
    }
    panel.style.display = "block";
    drawTree(payload, titleText);
    positionTreePanel(event);
  });

  gd.on("plotly_unhover", hideTreePanel);
})();
"""
    return (
        script.replace("__TREES__", tree_json)
        .replace("__ACTUAL_TREE__", actual_tree_json)
        .replace("__FRONT_TREES__", front_trees_json)
        .replace("__TREES_TRACE__", trees_trace_json)
        .replace("__ACTUAL_TRACE__", actual_trace_json)
        .replace("__FRONT_TRACE__", front_trace_json)
        .replace("__TREES_TITLE__", trees_title_json)
        .replace("__ACTUAL_TITLE__", actual_title_json)
        .replace("__FRONT_TITLE__", front_title_json)
    )


def _sampling_post_script(trees, actual_tree, mean_norm):
    """Return the JavaScript that adds the colorbar marker and tree hover panel."""
    return "\n".join(
        [
            _colorbar_marker_script(mean_norm),
            _tree_hover_script(
                trees,
                actual_tree,
                trees_trace_name="Samples",
                actual_trace_name="Actual (straightened)",
                trees_title_prefix="Sampled RSA",
                actual_title="Actual (straightened) RSA",
            ),
        ]
    )


def save_sampling_html(result, dest, scale_factor=1.0, region_factor=0.5,
                       unit=""):
    """Save an interactive Plotly HTML version of the region-sampling plot.

    Plotly keeps marker sizes fixed while zooming, unlike a static SVG where
    the whole drawing scales. The data are the same as in the Matplotlib
    version, optionally scaled to the user's unit.
    """
    import plotly.graph_objects as go

    scale = float(scale_factor)
    front = result["front"]
    actual = result["actual"]
    radius_w, radius_t = result["radius"]
    all_points = result["all_points"]
    max_rounds = result["max_rounds"]
    trees = result.get("trees") or []
    if len(trees) != len(all_points):
        trees = []
    actual_tree = result.get("actual_tree")

    front_points = sorted(front.values(), key=lambda p: p[0])
    front_x = [p[0] * scale for p in front_points]
    front_y = [p[1] * scale for p in front_points]

    theta = np.linspace(0, 2 * np.pi, 200)
    ellipse_x = (actual[0] + radius_w * np.cos(theta)) * scale
    ellipse_y = (actual[1] + radius_t * np.sin(theta)) * scale

    x_label = f"Total length ({unit})" if unit else "Total length"
    y_label = f"Travel distance ({unit})" if unit else "Travel distance"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=front_x, y=front_y, mode="lines", name="Pareto front",
        line=dict(color="black", width=2), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y, mode="lines", name="Target region",
        line=dict(color="red", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[actual[0] * scale], y=[actual[1] * scale], mode="markers",
        name="Actual (straightened)",
        marker=dict(
            symbol="star", size=18, color="black",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "Actual (straightened)<br>"
            f"Total length: {actual[0] * scale:.2f}<br>"
            f"Travel distance: {actual[1] * scale:.2f}<extra></extra>"
        ),
    ))

    if all_points:
        xs = [p[0] * scale for p in all_points]
        ys = [p[1] * scale for p in all_points]
        rounds = [p[2] for p in all_points]
        alphas = [p[3] for p in all_points]
        inside = [p[4] for p in all_points]
        norm_rounds = [r / max(max_rounds, 1) for r in rounds]
        hover = [
            f"Total length: {x:.2f}<br>Travel distance: {y:.2f}<br>"
            f"Optimization rounds: {r}<br>Alpha: {a:.3f}<br>Inside: {i}"
            for x, y, r, a, i in zip(xs, ys, rounds, alphas, inside)
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name="Samples",
            marker=dict(
                size=[6 + 14 * a for a in alphas],
                color=norm_rounds,
                colorscale=_optimization_colorscale(),
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Optimization level<br>(0 = fully random)",
                    x=1.02,
                    y=0.5,
                    len=0.75,
                    thickness=20,
                    xpad=0,
                ),
                opacity=0.7,
                line=dict(width=0),
            ),
            text=hover,
            hoverinfo="text",
        ))

    # The black marker and the left-side tree hover panel are added by the
    # post-script below, after Plotly has laid out the colorbar in the DOM.
    mean_norm = result.get("mean_rounds_normalized")

    if result["n_inside"] > 0:
        stats_text = (
            f"Mean opt. level: {result['mean_rounds_normalized']:.3f} "
            f"(CI: [{result['ci_low_normalized']:.3f}, "
            f"{result['ci_high_normalized']:.3f}])<br>"
            f"Mean α: {result['mean_alpha']:.3f} "
            f"(CI: [{result['alpha_ci_low']:.3f}, "
            f"{result['alpha_ci_high']:.3f}])<br>"
            f"n inside: {result['n_inside']} "
            f"(n_eff: {result['n_eff']:.1f})"
        )
    else:
        stats_text = "No points inside region"

    fig.add_annotation(
        x=0.99, y=0.01, xref="paper", yref="paper",
        text=stats_text, showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1,
        font=dict(size=11),
    )
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="Point size: α value<br>(small = low α, large = high α)",
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)", borderpad=4,
        font=dict(size=10),
    )

    fig.update_layout(
        title=f"Sampling for rounds estimation (region_factor={region_factor})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
        dragmode="zoom",
        legend=dict(
            x=0.01, y=0.88, bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=70, r=30, t=60, b=60),
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(destination),
        include_plotlyjs="directory",
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        post_script=_sampling_post_script(trees, actual_tree, mean_norm),
        div_id="ariadne-sampling-plot",
    )


def estimate_rounds_for_region(G, n_samples=5000, region_factor=0.5,
                                confidence=0.95, plot=True, verbose=True,
                                show=True, save_path=None, front=None,
                                actual=None, scale_factor=1.0,
                                progress_callback=None, collect_trees=False):
    """
    Sample semi-random RSAs and estimate the number of optimization rounds
    needed to reach an elliptical region around the actual (straightened) tree.

    The region is an ellipse centered at the actual tree's (wiring, travel)
    coordinates, with semi-axes proportional to the actual tree's scaling
    distance to the front.

    Inside points are weighted by proximity to the actual tree (weight 1 at
    center, 0 at boundary). Weighted statistics are computed for both the
    optimization rounds and the alpha values.

    The Pareto front returned and plotted is refined with every sampled
    architecture that is not dominated by the current front. The target
    ellipse itself remains fixed, based on the original front and actual tree.

    Args:
        G: Input graph.
        n_samples: Number of semi-random trees to generate.
        region_factor: Semi-axis scale as a fraction of the actual's scaling
                       distance. Default 0.5.
        confidence: Confidence level for the interval.
        plot: Create the sampling plot if True. A plot is also created when
              save_path is provided, even if plot is False.
        verbose: Print progress.
        show: Display the plot with plt.show() if True.
        save_path: Optional path for saving the plot (format inferred from
                   the extension, e.g. .svg).
        front: Optional Pareto front already computed for G. Reused to avoid
               recomputing it.
        actual: Optional straightened-tree reference already computed for G.
        scale_factor: Multiplicative factor applied to plot axes only.
        progress_callback: Optional callable(done, total, n_inside) invoked
            periodically during semi-random tree generation.
        collect_trees: Store a compact geometry payload for every sampled tree
            in ``result['trees']`` and the straightened tree in
            ``result['actual_tree']``. Used by the interactive HTML hover panel.

    Returns:
        dict with weighted stats for rounds and alpha, plus raw data.
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    if front is None or actual is None:
        if verbose:
            print("Computing Pareto front...")
        front, actual = pareto_front(H)

    from ariadne_roots.quantify import distance_from_front
    _, scaling_distance = distance_from_front(front, actual)

    # The greedy front is only an approximation. Keep a local copy that will
    # be refined with any non-dominated sampled architecture.
    updated_front = {
        float(alpha): [float(costs[0]), float(costs[1])]
        for alpha, costs in front.items()
    }

    actual_tree = (
        _tree_hover_payload(straighten_tree(H), None, None)
        if collect_trees
        else None
    )

    scaling_excess = max(0.0, (scaling_distance - 1) / scaling_distance)
    if scaling_excess > 0:
        radius_w = region_factor * scaling_excess * actual[0]
        radius_t = region_factor * scaling_excess * actual[1]
    else:
        # Degenerate ellipse if the actual tree lies on the front; use a tiny
        # non-zero radius so the normalized distance remains defined.
        radius_w = max(abs(actual[0]) * 1e-6, 1e-9)
        radius_t = max(abs(actual[1]) * 1e-6, 1e-9)
        if verbose:
            print("Warning: actual tree lies on the front; using a tiny region.")

    if verbose:
        print(f"Actual tree: wiring={actual[0]:.1f}, travel={actual[1]:.1f}")
        print(f"Scaling distance to front: {scaling_distance:.4f}")
        print(f"Ellipse radii: ({radius_w:.4f}, {radius_t:.4f})")

    max_rounds = len(critical_nodes) - 1

    if verbose:
        print(f"Max rounds: {max_rounds}")
        print(f"Generating {n_samples} semi-random trees...")

    all_points = []
    inside_data = []  # (rounds, alpha, weight)
    trees = []

    report_every = max(1, n_samples // 100)
    for i in range(n_samples):
        alpha = np.random.uniform(0, 1)
        rounds = np.random.randint(0, max_rounds + 1)

        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
            H, alpha=alpha, optimization_rounds=rounds
        )

        if tree is not None:
            w, t = graph_costs(tree, critical_nodes=critical_nodes)

            # Incorporate non-dominated samples into the plotted front.
            is_dominated = any(
                fw <= w and ft <= t and (fw < w or ft < t)
                for fw, ft in updated_front.values()
            )
            if not is_dominated:
                dominated_keys = [
                    key
                    for key, (fw, ft) in updated_front.items()
                    if w <= fw and t <= ft and (w < fw or t < ft)
                ]
                for key in dominated_keys:
                    del updated_front[key]

                is_duplicate = any(
                    fw == w and ft == t
                    for fw, ft in updated_front.values()
                )
                if not is_duplicate:
                    key = float(alpha)
                    while key in updated_front:
                        key += 1e-12
                    updated_front[key] = [float(w), float(t)]

            d_norm = np.sqrt(((w - actual[0]) / radius_w) ** 2 +
                             ((t - actual[1]) / radius_t) ** 2)
            inside = d_norm <= 1
            weight = max(0.0, 1.0 - d_norm)

            all_points.append((w, t, rounds, alpha, inside))

            if collect_trees:
                trees.append(_tree_hover_payload(tree, rounds, alpha))

            if inside:
                inside_data.append((rounds, alpha, weight))

        if progress_callback is not None and (
            (i + 1) % report_every == 0 or i + 1 == n_samples
        ):
            progress_callback(i + 1, n_samples, len(inside_data))

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples}, inside: {len(inside_data)}")

    if verbose:
        print(f"\nTotal inside region: {len(inside_data)}")

    if len(inside_data) == 0:
        result = {
            'n_inside': 0,
            'mean_rounds': None, 'ci_low': None, 'ci_high': None,
            'mean_rounds_normalized': None,
            'ci_low_normalized': None, 'ci_high_normalized': None,
            'mean_alpha': None, 'alpha_ci_low': None, 'alpha_ci_high': None,
            'all_points': all_points, 'actual': actual, 'front': updated_front,
            'scaling_distance': scaling_distance,
            'radius': (radius_w, radius_t),
            'max_rounds': max_rounds,
            'trees': trees,
            'actual_tree': actual_tree,
        }
    else:
        from scipy import stats

        rounds_arr = np.array([d[0] for d in inside_data])
        alphas_arr = np.array([d[1] for d in inside_data])
        weights_arr = np.array([d[2] for d in inside_data])

        total_weight = np.sum(weights_arr)
        # Effective sample size (Kish's formula)
        n_eff = total_weight ** 2 / np.sum(weights_arr ** 2)

        # Weighted mean helper
        def weighted_mean(values):
            return np.sum(weights_arr * values) / total_weight

        # Weighted std (reliability weights)
        def weighted_std(values, mean_val):
            var = np.sum(weights_arr * (values - mean_val) ** 2) / total_weight
            return np.sqrt(var)

        # Weighted CI helper
        def weighted_ci(values):
            mean_val = weighted_mean(values)
            std_val = weighted_std(values, mean_val)
            if n_eff > 1:
                sem = std_val / np.sqrt(n_eff)
                t_crit = stats.t.ppf((1 + confidence) / 2, df=n_eff - 1)
                return mean_val, mean_val - t_crit * sem, mean_val + t_crit * sem
            else:
                return mean_val, mean_val, mean_val

        mean_rounds, ci_low, ci_high = weighted_ci(rounds_arr)
        mean_alpha, alpha_ci_low, alpha_ci_high = weighted_ci(alphas_arr)

        result = {
            'n_inside': len(inside_data),
            'n_eff': n_eff,
            'mean_rounds': mean_rounds,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mean_rounds_normalized': mean_rounds / max_rounds,
            'ci_low_normalized': ci_low / max_rounds,
            'ci_high_normalized': ci_high / max_rounds,
            'mean_alpha': mean_alpha,
            'alpha_ci_low': alpha_ci_low,
            'alpha_ci_high': alpha_ci_high,
            'all_points': all_points,
            'actual': actual,
            'front': updated_front,
            'scaling_distance': scaling_distance,
            'radius': (radius_w, radius_t),
            'max_rounds': max_rounds,
            'trees': trees,
            'actual_tree': actual_tree,
        }

        if verbose:
            print(f"\nWeighted results (n_inside={len(inside_data)}, "
                  f"n_eff={n_eff:.1f}):")
            print(f"  Mean rounds: {mean_rounds:.2f} "
                  f"(normalized: {mean_rounds / max_rounds:.3f})")
            print(f"  {confidence*100:.0f}% CI: [{ci_low:.2f}, {ci_high:.2f}] "
                  f"(normalized: [{ci_low / max_rounds:.3f}, "
                  f"{ci_high / max_rounds:.3f}])")
            print(f"  Mean alpha: {mean_alpha:.3f}")
            print(f"  {confidence*100:.0f}% CI: [{alpha_ci_low:.3f}, "
                  f"{alpha_ci_high:.3f}]")

    if plot or save_path is not None:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Plot-only scaling; raw values stay in the returned statistics.
        def plot_scale(value):
            return value * scale_factor

        # Pareto front (refined with non-dominated samples)
        front_points = sorted(updated_front.values(), key=lambda p: p[0])
        fx = [plot_scale(p[0]) for p in front_points]
        fy = [plot_scale(p[1]) for p in front_points]
        ax.plot(fx, fy, '-', color='black', linewidth=2,
                label='Pareto Front', zorder=3)

        # Ellipse around actual tree
        theta = np.linspace(0, 2 * np.pi, 200)
        cx = plot_scale(actual[0]) + plot_scale(radius_w) * np.cos(theta)
        cy = plot_scale(actual[1]) + plot_scale(radius_t) * np.sin(theta)
        ax.plot(cx, cy, '--', color='red', linewidth=2,
                label='Target region', zorder=2)

        # Actual tree
        ax.scatter(plot_scale(actual[0]), plot_scale(actual[1]),
                   marker='*', color='black', s=250,
                   zorder=6, label='Actual (straightened)', edgecolors='white')

        # All points, colored by optimization level, sized by alpha
        if all_points:
            pts_rounds = np.array([p[2] for p in all_points])
            pts_alphas = np.array([p[3] for p in all_points])
            pts_x = [plot_scale(p[0]) for p in all_points]
            pts_y = [plot_scale(p[1]) for p in all_points]

            norm_rounds = pts_rounds / max(max_rounds, 1)
            hue = 0.67 * norm_rounds
            saturation = np.ones_like(hue) * 0.8
            value = np.ones_like(hue) * 0.9
            colors_hsv = np.stack([hue, saturation, value], axis=1)
            colors_rgb = hsv_to_rgb(colors_hsv)
            point_sizes = 10 + 70 * pts_alphas

            ax.scatter(pts_x, pts_y, c=colors_rgb, s=point_sizes, zorder=1,
                       edgecolors='none', alpha=0.7, label='Samples')

        ax.set_xlabel("Total length", fontsize=13)
        ax.set_ylabel("Travel distance", fontsize=13)
        ax.set_title(f"Sampling for rounds estimation "
                     f"(region_factor={region_factor})", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Vector colorbar for optimization level. Drawing the gradient as
        # rectangles keeps the saved SVG fully vector (no embedded PNG).
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        n_colors = 256
        hue_values = np.linspace(0, 0.67, n_colors)
        sat_values = np.ones(n_colors) * 0.8
        val_values = np.ones(n_colors) * 0.9
        hsv_cmap = np.stack([hue_values, sat_values, val_values], axis=1)
        rgb_cmap = hsv_to_rgb(hsv_cmap)
        cmap = LinearSegmentedColormap.from_list('rand_cmap', rgb_cmap)

        n_steps = 256
        for i in range(n_steps):
            cbar_ax.add_patch(
                Rectangle(
                    (0, i / n_steps),
                    1,
                    1 / n_steps,
                    facecolor=cmap(i / (n_steps - 1)),
                    edgecolor='none',
                    linewidth=0,
                )
            )
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, 1)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)
        cbar_ax.yaxis.set_label_position('right')
        cbar_ax.set_ylabel(
            'Optimization level\n(0 = fully random)',
            fontsize=9,
            rotation=90,
            labelpad=8,
        )

        # Horizontal bar at mean optimization level
        if len(inside_data) > 0:
            mean_pct = result['mean_rounds'] / max_rounds
            cbar_ax.axhline(
                y=mean_pct, color='black', linewidth=2.5, xmin=0, xmax=1
            )

        # Size legend note
        ax.text(0.02, 0.98, 'Point size: α value\n(small = low α, large = high α)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Statistics text box
        if len(inside_data) > 0:
            stats_text = (
                f"Mean opt. level: {result['mean_rounds_normalized']:.3f} "
                f"(CI: [{result['ci_low_normalized']:.3f}, "
                f"{result['ci_high_normalized']:.3f}])\n"
                f"Mean α: {result['mean_alpha']:.3f} "
                f"(CI: [{result['alpha_ci_low']:.3f}, "
                f"{result['alpha_ci_high']:.3f}])\n"
                f"n inside: {result['n_inside']} "
                f"(n_eff: {result['n_eff']:.1f})"
            )
        else:
            stats_text = "No points inside region"

        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                          edgecolor='gray'))

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return result
