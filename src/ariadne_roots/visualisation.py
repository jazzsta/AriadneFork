import copy
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from ariadne_roots.quantify import (
    calc_len_LRs, calculate_tradeoff, distance_from_front, calc_len_PR,
    find_lowermost_node_of_primary_root, calculate_distance,
    pareto_calcs_3d_path_tortuosity, calc_len_LRs_with_distances, calc_zones
)
from ariadne_roots.pareto_functions import pareto_front


def pareto_calcs(G, random_func, n_random=100):
    """
    Perform Pareto-related calculations for a given random tree generator.

    Args:
        G: Input graph
        random_func: Function that takes (G, n) and returns (costs, trees)
        n_random: Number of random trees to generate

    Returns:
        results, front, randoms
    """
    front, actual = pareto_front(G)
    mactual, sactual = actual

    tradeoff_info = calculate_tradeoff(front, actual)
    plant_alpha, plant_scaling = distance_from_front(front, actual)

    print(f"Generating random trees using {random_func.__name__}...")
    randoms = random_func(G, n_random)[0]
    print(f"  Got {len(randoms)} random trees")

    if len(randoms) > 0:
        mrand = np.mean([x[0] for x in randoms])
        srand = np.mean([x[1] for x in randoms])
        rand_alpha, rand_scaling = distance_from_front(front, (mrand, srand))
    else:
        mrand, srand = 0.0, 0.0
        rand_alpha, rand_scaling = 0.0, 0.0

    results = {
        "Total root length": mactual,
        "Travel distance": sactual,
        "alpha": plant_alpha,
        "scaling distance to front": plant_scaling,
        "Total root length (random)": mrand,
        "Travel distance (random)": srand,
        "alpha (random)": rand_alpha,
        "scaling (random)": rand_scaling,
    }

    results.update(tradeoff_info)
    return results, front, randoms


def analyze(G, random_func=None, enable_3d=True, n_random=100):
    """
    Report basic root metrics for a given graph.
    Optionally includes random tree analysis.

    Args:
        G: NetworkX graph representing the root system
        random_func: Optional function to generate random trees for comparison
        enable_3d: If True, compute 3D Pareto analysis
        n_random: Number of random trees to generate

    Returns:
        results, front, randoms, results_3d, front_3d, randoms_3d
    """
    assert nx.is_tree(G)
    H = copy.deepcopy(G)

    for node in H.nodes(data="pos"):
        if node[1] == [0, 0]:
            root_node = node[0]
    assert root_node == 0

    len_PR = calc_len_PR(H, root_node)
    LR_info = calc_len_LRs(H)
    num_LRs = len(LR_info)
    lens_LRs = [x[0] for x in LR_info.values()]
    angles_LRs = [x[1] for x in LR_info.values()]
    density_LRs = num_LRs / len_PR

    uppermost_node_pos = H.nodes[root_node]["pos"]
    lowermost_node_pos = find_lowermost_node_of_primary_root(H, root_node)
    distance_root = calculate_distance(uppermost_node_pos, lowermost_node_pos)

    if random_func is not None:
        results, front, randoms = pareto_calcs(H, random_func, n_random)
    else:
        front, actual = pareto_front(H)
        results = {
            "Total root length": actual[0],
            "Travel distance": actual[1],
        }
        randoms = []

    if enable_3d:
        results_3d, front_3d, randoms_3d = pareto_calcs_3d_path_tortuosity(H)
    else:
        results_3d, front_3d, randoms_3d = {}, {}, []

    lateral_root_info = calc_len_LRs_with_distances(H)
    num_LRs = len(lateral_root_info)
    lens_LRs = [info[0] for info in lateral_root_info.values()]
    distances_LRs = [info[1] for info in lateral_root_info.values()]

    points = np.array([H.nodes[node]["pos"] for node in H.nodes()])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    barycenter_x = np.mean(hull_points[:, 0])
    barycenter_y = np.mean(hull_points[:, 1])

    uppermost_node = min(H.nodes(data="pos"), key=lambda node: node[1][1])
    uppermost_node_pos = uppermost_node[1]
    barycenter_y_displacement = abs(barycenter_y - uppermost_node_pos[1])
    barycenter_x_displacement = abs(barycenter_x - uppermost_node_pos[0])

    zone_lengths = calc_zones(H, root_node)
    branched_zone_length = zone_lengths["branched_zone_length"]
    basal_zone_length = zone_lengths["basal_zone_length"]
    apical_zone_length = zone_lengths["apical_zone_length"]

    if basal_zone_length == len_PR:
        basal_zone_length = 0

    branched_zone_density = num_LRs / branched_zone_length if branched_zone_length != 0 else 0

    mean_LR_lengths = float(np.mean(lens_LRs))
    median_LR_lengths = float(np.median(lens_LRs))
    median_LR_angles = float(np.median(angles_LRs))
    mean_LR_angles = float(np.mean(angles_LRs))
    mean_LR_distances = float(np.mean(distances_LRs))
    median_LR_distances = float(np.median(distances_LRs))
    sum_LR_distances = float(np.sum(distances_LRs))
    total_distance = sum_LR_distances + distance_root

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
    results["LR lengths"] = [float(x) for x in lens_LRs]
    results["LR angles"] = [float(x) for x in angles_LRs]
    results["LR minimal lengths"] = [float(x) for x in distances_LRs]
    results["Barycenter x displacement"] = float(barycenter_x_displacement)
    results["Barycenter y displacement"] = float(barycenter_y_displacement)
    results["Total minimal Distance"] = total_distance

    Total_root_length = len_PR + sum(lens_LRs)
    material_distance_ratio = Total_root_length / total_distance
    results["Tortuosity"] = material_distance_ratio

    points = np.array([H.nodes[node]["pos"] for node in H.nodes()])
    hull = ConvexHull(points)
    convex_hull_area = hull.volume
    results["Convex Hull Area"] = float(convex_hull_area)

    return results, front, randoms, results_3d, front_3d, randoms_3d


def calculate_plot_buffer(min_val, max_val, buffer_percent=0.2):
    """Calculate axis limits with buffer."""
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    buffer = range_val * buffer_percent
    return min_val - buffer, max_val + buffer

def comparaplot(G, random1, random2, n_random=500, dest=None):
    """
    Plot Pareto front comparing two random tree generators.

    Args:
        G: Input graph
        random1: First random tree function (e.g., random_tree)
        random2: Second random tree function (e.g., generate_random_trees_from_input)
        n_random: Number of random trees per generator
        dest: Optional path to save the plot
    """
    # Make a copy and ensure edge weights exist
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

    # Generate Pareto front and actual costs
    front, actual = pareto_front(H)

    # Generate trees from both functions
    print(f"Generating trees with {random1.__name__}...")
    randoms1 = random1(H, n_random)[0]
    print(f"  Got {len(randoms1)} trees")

    print(f"Generating trees with {random2.__name__}...")
    randoms2 = random2(H, n_random)[0]
    print(f"  Got {len(randoms2)} trees")

    # Compute centroids
    if len(randoms1) > 0:
        m1 = np.mean([x[0] for x in randoms1])
        s1 = np.mean([x[1] for x in randoms1])
    else:
        m1, s1 = 0.0, 0.0

    if len(randoms2) > 0:
        m2 = np.mean([x[0] for x in randoms2])
        s2 = np.mean([x[1] for x in randoms2])
    else:
        m2, s2 = 0.0, 0.0

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    def scale_data(data):
        return data * scale_factor

    # Scale all data
    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]
    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    scaled_r1_x = [scale_data(x[0]) for x in randoms1]
    scaled_r1_y = [scale_data(x[1]) for x in randoms1]
    scaled_r2_x = [scale_data(x[0]) for x in randoms2]
    scaled_r2_y = [scale_data(x[1]) for x in randoms2]

    scaled_m1 = scale_data(m1)
    scaled_s1 = scale_data(s1)
    scaled_m2 = scale_data(m2)
    scaled_s2 = scale_data(s2)

    # Plot first random generator
    for i, (x, y) in enumerate(zip(scaled_r1_x, scaled_r1_y)):
        plt.plot(x, y, marker="+", color="green", markersize=2.5, zorder=0.5,
                 markeredgewidth=0.5, alpha=0.5,
                 label=random1.__name__ if i == 0 else '')

    # Plot second random generator
    for i, (x, y) in enumerate(zip(scaled_r2_x, scaled_r2_y)):
        plt.plot(x, y, marker="o", color="blue", markersize=3.0, zorder=0.6,
                 markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                 label=random2.__name__ if i == 0 else '')

    # Plot Pareto front
    plt.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
             markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)

    # Plot actual tree
    plt.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
             zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')

    # Plot centroids
    plt.plot(scaled_m1, scaled_s1, marker="+", color="red",
             markersize=14, zorder=3, markeredgewidth=2.0,
             label=f'{random1.__name__} Mean')
    plt.plot(scaled_m2, scaled_s2, marker="o", color="cyan",
             markersize=12, zorder=3, markeredgewidth=2.0,
             label=f'{random2.__name__} Mean')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=15)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=15)
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f"Pareto Front: {random1.__name__} vs {random2.__name__}", fontsize=16)
    ax.grid(True, alpha=0.3)

    # Calculate bounds
    all_x = scaled_front_x + scaled_r1_x + scaled_r2_x + [scaled_actual_x, scaled_m1, scaled_m2]
    all_y = scaled_front_y + scaled_r1_y + scaled_r2_y + [scaled_actual_y, scaled_s1, scaled_s2]

    x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
    y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=300)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def comparaplot_with_trees(G, random1, random2, n_random=500, dest=None):
    """
    Master plot: Pareto front comparison surrounded by example tree graphs.

    Layout:
    ┌──────────┬──────────────────────────────────────────────┐
    │ Tree 1   │                                              │
    ├──────────┤         Pareto Front Plot                    │
    │ Tree 2   │           (comparaplot)                      │
    ├──────────┤                                              │
    │ Tree 3   │                                              │
    ├──────────┼────────┬────────┬──────────┬─────────────────┤
    │ Tree 4   │ Tree 5 │ Tree 6 │ Tree 7   │    Tree 8       │
    └──────────┴────────┴────────┴──────────┴─────────────────┘

    Trees:
      1. Actual tree
      2. Random1 tree (closest to centroid)
      3. Random2 tree (closest to centroid)
      4. Satellite tree (alpha=0)
      5. Steiner tree (alpha=0.25)
      6. Steiner tree (alpha=0.5)
      7. Steiner tree (alpha=0.75)
      8. Steiner tree (alpha=1.0)
    """
    from ariadne_roots.pareto_functions import (
        pareto_front, satellite_tree, pareto_steiner_fast, graph_costs, point_dist
    )

    # Make a copy and ensure edge weights exist
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Generate Pareto front and actual costs
    front, actual = pareto_front(H)

    # Generate random trees from both methods
    print(f"Generating trees with {random1.__name__}...")
    costs1, trees1 = random1(H, n_random)
    print(f"  Got {len(trees1)} trees")

    print(f"Generating trees with {random2.__name__}...")
    costs2, trees2 = random2(H, n_random)
    print(f"  Got {len(trees2)} trees")

    # Generate steiner/satellite trees for key alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    steiner_trees = {}
    for alpha in alphas:
        if alpha == 0.0:
            steiner_trees[alpha] = satellite_tree(H)
        else:
            steiner_trees[alpha] = pareto_steiner_fast(H, alpha)

    # Compute centroids
    if len(costs1) > 0:
        m1 = np.mean([x[0] for x in costs1])
        s1 = np.mean([x[1] for x in costs1])
    else:
        m1, s1 = 0.0, 0.0

    if len(costs2) > 0:
        m2 = np.mean([x[0] for x in costs2])
        s2 = np.mean([x[1] for x in costs2])
    else:
        m2, s2 = 0.0, 0.0

    # Find trees closest to the mean for each method
    def find_closest_to_mean(costs, trees, m, s):
        if len(costs) == 0:
            return None
        best_idx = min(range(len(costs)),
                       key=lambda i: (costs[i][0] - m) ** 2 + (costs[i][1] - s) ** 2)
        return trees[best_idx]

    tree1_mean = find_closest_to_mean(costs1, trees1, m1, s1)
    tree2_mean = find_closest_to_mean(costs2, trees2, m2, s2)

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    # ---- CREATE FIGURE WITH GRIDSPEC (4 rows × 5 columns) ----
    fig = plt.figure(figsize=(26, 18))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.9],
                          width_ratios=[0.35, 1, 1, 1, 1])

    # Main Pareto plot: rows 0-2, cols 1-4 (3 rows × 4 columns)
    ax_main = fig.add_subplot(gs[0:3, 1:5])

    def scale_data(data):
        return data * scale_factor

    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]
    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    scaled_r1_x = [scale_data(x[0]) for x in costs1]
    scaled_r1_y = [scale_data(x[1]) for x in costs1]
    scaled_r2_x = [scale_data(x[0]) for x in costs2]
    scaled_r2_y = [scale_data(x[1]) for x in costs2]

    scaled_m1 = scale_data(m1)
    scaled_s1 = scale_data(s1)
    scaled_m2 = scale_data(m2)
    scaled_s2 = scale_data(s2)

    # Plot random clouds
    for i, (x, y) in enumerate(zip(scaled_r1_x, scaled_r1_y)):
        ax_main.plot(x, y, marker="+", color="green", markersize=2.5, zorder=0.5,
                     markeredgewidth=0.5, alpha=0.5,
                     label=random1.__name__ if i == 0 else '')

    for i, (x, y) in enumerate(zip(scaled_r2_x, scaled_r2_y)):
        ax_main.plot(x, y, marker="o", color="blue", markersize=3.0, zorder=0.6,
                     markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                     label=random2.__name__ if i == 0 else '')

    # Plot Pareto front
    ax_main.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
                 markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)

    # Plot actual tree
    ax_main.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
                 zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')

    # Plot centroids
    ax_main.plot(scaled_m1, scaled_s1, marker="+", color="red",
                 markersize=14, zorder=3, markeredgewidth=2.0,
                 label=f'{random1.__name__} Mean')
    ax_main.plot(scaled_m2, scaled_s2, marker="o", color="cyan",
                 markersize=12, zorder=3, markeredgewidth=2.0,
                 label=f'{random2.__name__} Mean')

    ax_main.set_xlabel(f"Total length ({scale_unit})", fontsize=11)
    ax_main.set_ylabel(f"Travel distance ({scale_unit})", fontsize=11)
    ax_main.legend(loc='best', fontsize=7)
    ax_main.set_title(f"Pareto Front: {random1.__name__} vs {random2.__name__}", fontsize=13)
    ax_main.grid(True, alpha=0.3)

    all_x = scaled_front_x + scaled_r1_x + scaled_r2_x + [scaled_actual_x, scaled_m1, scaled_m2]
    all_y = scaled_front_y + scaled_r1_y + scaled_r2_y + [scaled_actual_y, scaled_s1, scaled_s2]
    x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
    y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    # ---- HELPER TO DRAW A SIMPLIFIED TREE (only tips, base, and edges) ----
    def draw_simplified_tree(ax, tree, title_str):
        if tree is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(title_str, fontsize=7)
            return

        base_node = 0
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]

        # Get positions of all nodes
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        # Draw ALL edges (they connect through internal nodes too)
        for u, v in tree.edges():
            x_vals = [all_pos[u][0], all_pos[v][0]]
            y_vals = [all_pos[u][1], all_pos[v][1]]
            ax.plot(x_vals, y_vals, color="#3D2B1F", linewidth=0.4, alpha=0.7)

        # Draw base node
        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=15, zorder=3, edgecolors='black', linewidth=0.3)
        # Draw tip nodes
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=8, zorder=3, edgecolors='black', linewidth=0.2)

        ax.invert_yaxis()
        ax.set_title(title_str, fontsize=7)
        ax.axis('equal')
        ax.axis('off')

    # Left column (rows 0-2, col 0): 3 trees
    ax1 = fig.add_subplot(gs[0, 0])
    draw_simplified_tree(ax1, H, "1. Actual Tree")

    ax2 = fig.add_subplot(gs[1, 0])
    draw_simplified_tree(ax2, tree1_mean, f"2. {random1.__name__}\n(closest to mean)")

    ax3 = fig.add_subplot(gs[2, 0])
    draw_simplified_tree(ax3, tree2_mean, f"3. {random2.__name__}\n(closest to mean)")

    # Bottom row (row 3, cols 0-4): 5 trees
    bottom_trees = [
        (steiner_trees[0.0], "4. Satellite (α=0)"),
        (steiner_trees[0.25], "5. Steiner (α=0.25)"),
        (steiner_trees[0.5], "6. Steiner (α=0.5)"),
        (steiner_trees[0.75], "7. Steiner (α=0.75)"),
        (steiner_trees[1.0], "8. Steiner (α=1.0)"),
    ]
    for i, (tree, label) in enumerate(bottom_trees):
        ax = fig.add_subplot(gs[3, i])
        draw_simplified_tree(ax, tree, label)

    # Adjust spacing
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig


