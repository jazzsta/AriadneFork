import copy
import random

import numpy as np
import matplotlib.pyplot as plt

from ariadne_roots.exploration import pareto_steiner_partial_random, generate_trees_from_tips_topdown, \
    local_optimize_tree_pareto_set, update_all_distances, straighten_tree
from ariadne_roots.pareto_functions import point_dist, pareto_front, \
    pareto_steiner_fast_preserve_tips, satellite_tree, graph_costs,\
    pareto_front_generic, pareto_cost, get_critical_nodes, pareto_steiner_fast_analytical_preserve_tips


def calculate_plot_buffer(min_val, max_val, buffer_percent=0.2):
    """Calculate axis limits with buffer."""
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    buffer = range_val * buffer_percent
    return min_val - buffer, max_val + buffer


def comparaplot(G, random1, random2, pareto_front, n_random=500, dest=None):
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


def comparaplot_axisfix(G,random1, random2, pareto_front, n_random=500, dest=None):
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
        ax.plot(x, y, marker="+", color="green", markersize=2.5, zorder=0.5,
                 markeredgewidth=0.5, alpha=0.5,
                 label=random1.__name__ if i == 0 else '')

    # Plot second random generator
    for i, (x, y) in enumerate(zip(scaled_r2_x, scaled_r2_y)):
        ax.plot(x, y, marker="o", color="blue", markersize=3.0, zorder=0.6,
                 markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                 label=random2.__name__ if i == 0 else '')

    # Plot Pareto front
    ax.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
             markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)

    # Plot actual tree
    ax.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
             zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')

    # Plot centroids
    ax.plot(scaled_m1, scaled_s1, marker="+", color="red",
             markersize=14, zorder=3, markeredgewidth=2.0,
             label=f'{random1.__name__} Mean')
    ax.plot(scaled_m2, scaled_s2, marker="o", color="cyan",
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

    # Center Y-axis around the Steiner trees
    # Satellite (α=0) has smallest travel distance, Steiner (α=1) has largest
    steiner_y = max(scaled_front_y)
    satellite_y = min(scaled_front_y)
    mid_y = satellite_y + (steiner_y - satellite_y) * (2/3)
    half_range = (steiner_y - satellite_y)

    y_min = mid_y - half_range
    y_max = mid_y + half_range

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=300)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def comparaplot_with_trees(G, random1, random2, pareto_front, n_random=500, dest=None):
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
            steiner_trees[alpha] = pareto_steiner_fast_preserve_tips(H, alpha)

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


def compare_pareto_fronts(G, dest=None):
    """
    Compare the original Pareto front with the tip-preserving version.
    Plots both fronts and the actual tree on the same axes.

    Args:
        G: Input graph
        dest: Optional path to save the plot
    """

    # Make a copy and ensure edge weights exist
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Generate both Pareto fronts
    print("Computing original Pareto front...")
    front_orig, actual_orig = pareto_front(H)

    print("Computing tip-preserving Pareto front...")
    front_preserve, actual_preserve = pareto_front_preserve_tips(H)

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(10, 8))

    def scale_data(data):
        return data * scale_factor

    # Original Pareto front
    orig_x = [scale_data(x[0]) for x in front_orig.values()]
    orig_y = [scale_data(x[1]) for x in front_orig.values()]
    ax.plot(orig_x, orig_y, 'o-', color='orange', markersize=4,
            linewidth=2, label='Original Pareto Front', zorder=2)

    # Tip-preserving Pareto front
    preserve_x = [scale_data(x[0]) for x in front_preserve.values()]
    preserve_y = [scale_data(x[1]) for x in front_preserve.values()]
    ax.plot(preserve_x, preserve_y, 's--', color='blue', markersize=4,
            linewidth=2, label='Tip-Preserving Pareto Front', zorder=2)

    # Actual trees
    ax.scatter(scale_data(actual_orig[0]), scale_data(actual_orig[1]),
               marker='x', color='orange', s=100, linewidth=2,
               label='Actual (Original)', zorder=4)
    ax.scatter(scale_data(actual_preserve[0]), scale_data(actual_preserve[1]),
               marker='+', color='blue', s=100, linewidth=2,
               label='Actual (Preserving)', zorder=4)

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax.set_title("Pareto Front: Original vs Tip-Preserving", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Print numerical comparison
    print(f"\nOriginal front: {len(front_orig)} points")
    print(f"  Total length range: [{min(orig_x):.0f}, {max(orig_x):.0f}]")
    print(f"  Travel distance range: [{min(orig_y):.0f}, {max(orig_y):.0f}]")
    print(f"Preserving front: {len(front_preserve)} points")
    print(f"  Total length range: [{min(preserve_x):.0f}, {max(preserve_x):.0f}]")
    print(f"  Travel distance range: [{min(preserve_y):.0f}, {max(preserve_y):.0f}]")

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=300)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax

def compare_pareto_fronts_multi(graphs, steiner_functions, labels=None, colors=None, dest=None):
    """
    Plot Pareto fronts from multiple graphs and/or Steiner algorithms on the same axes.

    Args:
        graphs: List of nx.Graph objects, or a single graph
        steiner_functions: List of Steiner functions, each taking (G, alpha) and returning a tree
        labels: Optional list of labels for the legend (default: auto-generated)
        colors: Optional list of colors for each front (default: auto-cycled)
        dest: Optional path to save the plot
    """
    # Accept single graph as well
    if not isinstance(graphs, list):
        graphs = [graphs]

    if labels is None:
        labels = []
        for g_idx, G in enumerate(graphs):
            for fn in steiner_functions:
                if len(graphs) > 1:
                    labels.append(f"G{g_idx+1}: {fn.__name__}")
                else:
                    labels.append(fn.__name__)

    n_fronts = len(graphs) * len(steiner_functions)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_fronts, 1)))

    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    all_x = []
    all_y = []

    front_idx = 0
    for g_idx, G in enumerate(graphs):
        print(f"Processing graph {g_idx+1}/{len(graphs)}...")

        H = copy.deepcopy(G)
        for u, v in H.edges():
            if 'weight' not in H[u][v]:
                pos_u = H.nodes[u]['pos']
                pos_v = H.nodes[v]['pos']
                H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

        critical_nodes = get_critical_nodes(H)
        orig_w, orig_t = graph_costs(H, critical_nodes=critical_nodes)

        all_x.append(scale_data(orig_w))
        all_y.append(scale_data(orig_t))

        # Plot actual tree for this graph
        ax.scatter(scale_data(orig_w), scale_data(orig_t),
                   marker='*', color='black', s=200, linewidth=2, zorder=5,
                   label='Actual Tree' if g_idx == 0 else None, edgecolors='white')
        ax.annotate(str(g_idx+1), (scale_data(orig_w), scale_data(orig_t)),
                    textcoords="offset points", xytext=(8, 8), fontsize=10,
                    fontweight='bold', color='black')

        for fn_idx, steiner_fn in enumerate(steiner_functions):
            label = labels[front_idx]
            print(f"  {label}...")
            front, _ = pareto_front_generic(H, steiner_fn)

            fx = [scale_data(v[0]) for v in front.values()]
            fy = [scale_data(v[1]) for v in front.values()]
            all_x.extend(fx)
            all_y.extend(fy)

            ax.plot(fx, fy, 'o-', color=colors[front_idx], markersize=4,
                    linewidth=2.5, alpha=0.8, label=label, zorder=2)

            front_idx += 1

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Pareto Front Comparison ({len(graphs)} graph(s), {len(steiner_functions)} algorithm(s))",
                 fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def print_tree(G, node=0, parent=None, indent=0):
    pos = G.nodes[node].get('pos', [0,0])
    ntype = 'B' if node==0 else ('T' if G.nodes[node].get('is_tip',False) else 'I')
    print(f"{'  '*indent}{node} ({ntype}) [{pos[0]:.0f},{pos[1]:.0f}]")
    for child in G.neighbors(node):
        if child != parent:
            print_tree(G, child, node, indent+1)

def comparaplot_with_trees_steinerDEG(G, random1, random2, n_random=500, dest=None):
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
      4. Partial random (85% optimal)
      5. Partial random (90% optimal)
      6. Partial random (93% optimal)
      7. Partial random (96% optimal)
      8. Partial random (98% optimal)
    """

    # Make a copy and ensure edge weights exist
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Generate Pareto front and actual costs
    front, actual = pareto_front_preserve_tips_fixed_spacing_binary(H)

    # Generate random trees from both methods
    print(f"Generating trees with {random1.__name__}...")
    costs1, trees1 = random1(H, n_random)
    print(f"  Got {len(trees1)} trees")

    print(f"Generating trees with {random2.__name__}...")
    costs2, trees2 = random2(H, n_random)
    print(f"  Got {len(trees2)} trees")

    # Generate partial random trees at different optimal fractions
    optimal_fractions = [0.85, 0.90, 0.93, 0.96, 0.98]
    partial_labels = [
        "4. Partial (85% opt)",
        "5. Partial (90% opt)",
        "6. Partial (93% opt)",
        "7. Partial (96% opt)",
        "8. Partial (98% opt)",
    ]
    partial_trees = {}
    partial_costs = {}
    n_partial = n_random // 5

    for frac in optimal_fractions:
        print(f"Generating partial random trees ({frac:.0%} optimal)...")
        costs_p, trees_p = [], []
        for _ in range(n_partial):
            alpha = random.uniform(0.089, 0.19)
            tree = pareto_steiner_partial_random(H, alpha=alpha, optimal_fraction=frac)
            if tree is not None:
                m, s = graph_costs(tree,get_critical_nodes(tree))
                costs_p.append((m, s))
                trees_p.append(tree)
        partial_costs[frac] = costs_p
        partial_trees[frac] = trees_p
        print(f"  Got {len(trees_p)} trees")

    # Compute centroids
    m1 = np.mean([x[0] for x in costs1]) if costs1 else 0.0
    s1 = np.mean([x[1] for x in costs1]) if costs1 else 0.0
    m2 = np.mean([x[0] for x in costs2]) if costs2 else 0.0
    s2 = np.mean([x[1] for x in costs2]) if costs2 else 0.0

    # Find trees closest to the mean for each method
    def find_closest_to_mean(costs, trees, m, s):
        if len(costs) == 0:
            return None
        best_idx = min(range(len(costs)),
                       key=lambda i: (costs[i][0] - m) ** 2 + (costs[i][1] - s) ** 2)
        return trees[best_idx]

    tree1_mean = find_closest_to_mean(costs1, trees1, m1, s1)
    tree2_mean = find_closest_to_mean(costs2, trees2, m2, s2)

    # Find representative trees for each partial random fraction (closest to median)
    partial_repr = {}
    for frac in optimal_fractions:
        costs_p = partial_costs[frac]
        trees_p = partial_trees[frac]
        if len(costs_p) > 0:
            med_m = np.median([c[0] for c in costs_p])
            med_s = np.median([c[1] for c in costs_p])
            partial_repr[frac] = find_closest_to_mean(costs_p, trees_p, med_m, med_s)
        else:
            partial_repr[frac] = None

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

    # Main Pareto plot: rows 0-2, cols 1-4
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

    # Plot partial random clouds with different colors
    partial_colors = ['purple', 'magenta', 'orange', 'brown', 'pink']
    for idx, frac in enumerate(optimal_fractions):
        costs_p = partial_costs[frac]
        color = partial_colors[idx]
        for i, (x, y) in enumerate(costs_p):
            ax_main.plot(scale_data(x), scale_data(y), marker='d', color=color,
                         markersize=2.0, zorder=0.4, alpha=0.4,
                         label=f'Partial {frac:.0%}' if i == 0 else '')

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
    ax_main.legend(loc='best', fontsize=6, ncol=2)
    ax_main.set_title(f"Pareto Front with Partial Random Trees", fontsize=13)
    ax_main.grid(True, alpha=0.3)

    all_x = scaled_front_x + scaled_r1_x + scaled_r2_x + [scaled_actual_x, scaled_m1, scaled_m2]
    all_y = scaled_front_y + scaled_r1_y + scaled_r2_y + [scaled_actual_y, scaled_s1, scaled_s2]
    for frac in optimal_fractions:
        if frac in partial_costs:
            all_x.extend([scale_data(c[0]) for c in partial_costs[frac]])
            all_y.extend([scale_data(c[1]) for c in partial_costs[frac]])
    x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
    y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    # ---- HELPER TO DRAW A SIMPLIFIED TREE ----
    def draw_simplified_tree(ax, tree, title_str):
        if tree is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(title_str, fontsize=7)
            return

        base_node = 0
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        for u, v in tree.edges():
            x_vals = [all_pos[u][0], all_pos[v][0]]
            y_vals = [all_pos[u][1], all_pos[v][1]]
            ax.plot(x_vals, y_vals, color="#3D2B1F", linewidth=0.4, alpha=0.7)

        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=15, zorder=3, edgecolors='black', linewidth=0.3)
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

    # Bottom row (row 3, cols 0-4): 5 partial random trees
    for i, frac in enumerate(optimal_fractions):
        ax = fig.add_subplot(gs[3, i])
        tree = partial_repr.get(frac)
        draw_simplified_tree(ax, tree, partial_labels[i])

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


def comparaplot_with_trees_proximity(G, n_random=500, dest=None):
    """
    Master plot comparing random spanning trees, top-down, and bottom-up
    with varying proximity bias.

    Trees:
      1. Actual tree
      2. Random spanning tree (closest to centroid)
      3. Top-down tree (closest to centroid)
      4. Bottom-up (bias=0.0)
      5. Bottom-up (bias=0.25)
      6. Bottom-up (bias=0.5)
      7. Bottom-up (bias=0.75)
      8. Bottom-up (bias=1.0)
    """

    # Make a copy and ensure edge weights exist
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Generate Pareto front and actual costs
    front, actual = pareto_front_preserve_tips_fixed_spacing_binary(H)

    # Generate random spanning trees
    print("Generating random spanning trees...")
    costs_rand, trees_rand = random_tree(H, n_random)
    print(f"  Got {len(trees_rand)} trees")

    # Generate top-down trees
    print("Generating top-down trees...")
    costs_td, trees_td = generate_trees_from_tips_topdown(H, n_trees=n_random, verbose=False)
    print(f"  Got {len(trees_td)} trees")

    # Generate bottom-up trees at different proximity biases
    proximity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    proximity_labels = [
        "4. Bottom-up (bias=0.0)",
        "5. Bottom-up (bias=0.25)",
        "6. Bottom-up (bias=0.5)",
        "7. Bottom-up (bias=0.75)",
        "8. Bottom-up (bias=1.0)",
    ]
    bottomup_trees = {}
    bottomup_costs = {}
    n_per_level = n_random // 5

    for bias in proximity_levels:
        print(f"Generating bottom-up trees (proximity_bias={bias})...")
        costs_b, trees_b = generate_trees_from_tips_bottomup(
            H, n_trees=n_per_level, proximity_bias=bias, verbose=False
        )
        bottomup_costs[bias] = costs_b
        bottomup_trees[bias] = trees_b
        print(f"  Got {len(trees_b)} trees")

    # Compute centroids
    m_rand = np.mean([x[0] for x in costs_rand]) if costs_rand else 0.0
    s_rand = np.mean([x[1] for x in costs_rand]) if costs_rand else 0.0
    m_td = np.mean([x[0] for x in costs_td]) if costs_td else 0.0
    s_td = np.mean([x[1] for x in costs_td]) if costs_td else 0.0

    def find_closest_to_mean(costs, trees, m, s):
        if len(costs) == 0:
            return None
        best_idx = min(range(len(costs)),
                       key=lambda i: (costs[i][0] - m) ** 2 + (costs[i][1] - s) ** 2)
        return trees[best_idx]

    tree_rand_mean = find_closest_to_mean(costs_rand, trees_rand, m_rand, s_rand)
    tree_td_mean = find_closest_to_mean(costs_td, trees_td, m_td, s_td)

    # Find representative trees for each proximity level (closest to median)
    bottomup_repr = {}
    for bias in proximity_levels:
        costs_b = bottomup_costs[bias]
        trees_b = bottomup_trees[bias]
        if len(costs_b) > 0:
            med_m = np.median([c[0] for c in costs_b])
            med_s = np.median([c[1] for c in costs_b])
            bottomup_repr[bias] = find_closest_to_mean(costs_b, trees_b, med_m, med_s)
        else:
            bottomup_repr[bias] = None

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    # ---- CREATE FIGURE WITH GRIDSPEC ----
    fig = plt.figure(figsize=(26, 18))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.9],
                          width_ratios=[0.35, 1, 1, 1, 1])

    ax_main = fig.add_subplot(gs[0:3, 1:5])

    def scale_data(data):
        return data * scale_factor

    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]
    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    # Plot random spanning trees (green)
    for i, (x, y) in enumerate(zip(
        [scale_data(c[0]) for c in costs_rand],
        [scale_data(c[1]) for c in costs_rand]
    )):
        ax_main.plot(x, y, marker="+", color="green", markersize=2.5, zorder=0.5,
                     markeredgewidth=0.5, alpha=0.5,
                     label='Random Spanning' if i == 0 else '')

    # Plot top-down trees (blue)
    for i, (x, y) in enumerate(zip(
        [scale_data(c[0]) for c in costs_td],
        [scale_data(c[1]) for c in costs_td]
    )):
        ax_main.plot(x, y, marker="o", color="blue", markersize=2.5, zorder=0.6,
                     markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                     label='Top-Down' if i == 0 else '')

    # Plot bottom-up clouds with different colors
    bu_colors = ['purple', 'magenta', 'orange', 'brown', 'red']
    for idx, bias in enumerate(proximity_levels):
        costs_b = bottomup_costs[bias]
        color = bu_colors[idx]
        for i, (x, y) in enumerate(costs_b):
            ax_main.plot(scale_data(x), scale_data(y), marker='d', color=color,
                         markersize=2.0, zorder=0.4, alpha=0.4,
                         label=f'BU bias={bias}' if i == 0 else '')

    # Plot Pareto front
    ax_main.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
                 markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)

    # Plot actual tree
    ax_main.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
                 zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')

    # Plot centroids
    ax_main.plot(scale_data(m_rand), scale_data(s_rand), marker="+", color="darkgreen",
                 markersize=14, zorder=3, markeredgewidth=2.0, label='Random Mean')
    ax_main.plot(scale_data(m_td), scale_data(s_td), marker="o", color="darkblue",
                 markersize=12, zorder=3, markeredgewidth=2.0, label='Top-Down Mean')

    ax_main.set_xlabel(f"Total length ({scale_unit})", fontsize=11)
    ax_main.set_ylabel(f"Travel distance ({scale_unit})", fontsize=11)
    ax_main.legend(loc='best', fontsize=6, ncol=2)
    ax_main.set_title("Pareto Front: Random vs Top-Down vs Bottom-Up", fontsize=13)
    ax_main.grid(True, alpha=0.3)

    all_x = scaled_front_x + [scaled_actual_x]
    all_y = scaled_front_y + [scaled_actual_y]
    for c in costs_rand:
        all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    for c in costs_td:
        all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    for bias in proximity_levels:
        for c in bottomup_costs[bias]:
            all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
    y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    # ---- HELPER TO DRAW A SIMPLIFIED TREE ----
    def draw_simplified_tree(ax, tree, title_str):
        if tree is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(title_str, fontsize=7)
            return

        base_node = 0
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        for u, v in tree.edges():
            x_vals = [all_pos[u][0], all_pos[v][0]]
            y_vals = [all_pos[u][1], all_pos[v][1]]
            ax.plot(x_vals, y_vals, color="#3D2B1F", linewidth=0.4, alpha=0.7)

        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=15, zorder=3, edgecolors='black', linewidth=0.3)
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=8, zorder=3, edgecolors='black', linewidth=0.2)

        ax.invert_yaxis()
        ax.set_title(title_str, fontsize=7)
        ax.axis('equal')
        ax.axis('off')

    # Left column
    ax1 = fig.add_subplot(gs[0, 0])
    draw_simplified_tree(ax1, H, "1. Actual Tree")

    ax2 = fig.add_subplot(gs[1, 0])
    draw_simplified_tree(ax2, tree_rand_mean, "2. Random Spanning\n(closest to mean)")

    ax3 = fig.add_subplot(gs[2, 0])
    draw_simplified_tree(ax3, tree_td_mean, "3. Top-Down\n(closest to mean)")

    # Bottom row
    for i, bias in enumerate(proximity_levels):
        ax = fig.add_subplot(gs[3, i])
        tree = bottomup_repr.get(bias)
        draw_simplified_tree(ax, tree, proximity_labels[i])

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


def comparaplot_with_trees_proximity_only(G, n_random=500, dest=None):
    """
    Compare top-down vs bottom-up with varying proximity bias.

    Trees:
      1. Actual tree
      2. Top-down tree (closest to centroid)
      3. Bottom-up (bias=0.0, closest to centroid)
      4. Bottom-up (bias=0.0)
      5. Bottom-up (bias=0.25)
      6. Bottom-up (bias=0.5)
      7. Bottom-up (bias=0.75)
      8. Bottom-up (bias=1.0)
    """

    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    front, actual = pareto_front_preserve_tips_fixed_spacing_binary(H)

    # Generate top-down trees
    print("Generating top-down trees...")
    costs_td, trees_td = generate_trees_from_tips_topdown(H, n_trees=n_random, verbose=True)
    print(f"  Got {len(trees_td)} trees")

    # Generate bottom-up trees at different proximity biases
    proximity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    proximity_labels = [
        "4. Bottom-up (bias=0.0)",
        "5. Bottom-up (bias=0.25)",
        "6. Bottom-up (bias=0.5)",
        "7. Bottom-up (bias=0.75)",
        "8. Bottom-up (bias=1.0)",
    ]
    bottomup_costs = {}
    bottomup_trees = {}
    n_per_level = n_random // 5

    for bias in proximity_levels:
        print(f"Generating bottom-up trees (bias={bias})...")
        costs_b, trees_b = generate_trees_from_tips_bottomup_pb(
            H, n_trees=n_per_level, proximity_bias=bias, verbose=False
        )
        bottomup_costs[bias] = costs_b
        bottomup_trees[bias] = trees_b
        print(f"  Got {len(trees_b)} trees")

    m_td = np.mean([x[0] for x in costs_td]) if costs_td else 0.0
    s_td = np.mean([x[1] for x in costs_td]) if costs_td else 0.0

    def find_closest_to_median(costs, trees):
        if len(costs) == 0:
            return None
        med_m = np.median([c[0] for c in costs])
        med_s = np.median([c[1] for c in costs])
        best_idx = min(range(len(costs)),
                       key=lambda i: (costs[i][0] - med_m) ** 2 + (costs[i][1] - med_s) ** 2)
        return trees[best_idx]

    tree_td_repr = find_closest_to_median(costs_td, trees_td)

    bottomup_repr = {}
    bottomup_medians = {}
    for bias in proximity_levels:
        costs_b = bottomup_costs[bias]
        trees_b = bottomup_trees[bias]
        if len(costs_b) > 0:
            bottomup_repr[bias] = find_closest_to_median(costs_b, trees_b)
            bottomup_medians[bias] = (np.median([c[0] for c in costs_b]),
                                       np.median([c[1] for c in costs_b]))
        else:
            bottomup_repr[bias] = None
            bottomup_medians[bias] = (0, 0)

    # Representative for bias=0.0 for tree panel
    tree_bu_repr = bottomup_repr.get(0.0)

    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig = plt.figure(figsize=(26, 18))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.9],
                          width_ratios=[0.35, 1, 1, 1, 1])

    ax_main = fig.add_subplot(gs[0:3, 1:5])

    def scale_data(data):
        return data * scale_factor

    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]
    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    # Top-down cloud (blue)
    for i, (x, y) in enumerate(costs_td):
        ax_main.plot(scale_data(x), scale_data(y), marker="o", color="blue",
                     markersize=2.5, zorder=0.6, markeredgewidth=0.5, alpha=0.5,
                     fillstyle='none', label='Top-Down' if i == 0 else '')

    # Bottom-up clouds with color gradient (purple → red)
    bu_colors = ['purple', 'magenta', 'orange', 'brown', 'red']
    for idx, bias in enumerate(proximity_levels):
        costs_b = bottomup_costs[bias]
        color = bu_colors[idx]
        for i, (x, y) in enumerate(costs_b):
            ax_main.plot(scale_data(x), scale_data(y), marker='d', color=color,
                         markersize=2.0, zorder=0.4, alpha=0.4,
                         label=f'BU bias={bias}' if i == 0 else '')

    # Pareto front
    ax_main.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
                 markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)

    # Actual tree
    ax_main.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
                 zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')

    # Centroids
    ax_main.plot(scale_data(m_td), scale_data(s_td), marker="o", color="darkblue",
                 markersize=12, zorder=3, markeredgewidth=2.0, label='Top-Down Median')
    for idx, bias in enumerate(proximity_levels):
        m_b, s_b = bottomup_medians[bias]
        ax_main.plot(scale_data(m_b), scale_data(s_b), marker='d', color=bu_colors[idx],
                     markersize=10, zorder=3, markeredgewidth=1.5,
                     label=f'BU bias={bias} median' if idx == 0 else None)

    ax_main.set_xlabel(f"Total length ({scale_unit})", fontsize=11)
    ax_main.set_ylabel(f"Travel distance ({scale_unit})", fontsize=11)
    ax_main.legend(loc='best', fontsize=10, ncol=2)
    ax_main.set_title("Pareto Front: Top-Down vs Bottom-Up (Proximity Bias)", fontsize=13)
    ax_main.grid(True, alpha=0.3)

    all_x = scaled_front_x + [scaled_actual_x]
    all_y = scaled_front_y + [scaled_actual_y]
    for c in costs_td:
        all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    for bias in proximity_levels:
        for c in bottomup_costs[bias]:
            all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
    y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    def draw_simplified_tree(ax, tree, title_str):
        if tree is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(title_str, fontsize=7)
            return
        base_node = 0
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}
        for u, v in tree.edges():
            ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                    color="#3D2B1F", linewidth=0.4, alpha=0.7)
        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=15, zorder=3, edgecolors='black', linewidth=0.3)
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=8, zorder=3, edgecolors='black', linewidth=0.2)
        ax.invert_yaxis()
        ax.set_title(title_str, fontsize=7)
        ax.axis('equal')
        ax.axis('off')

    # Left column
    ax1 = fig.add_subplot(gs[0, 0])
    draw_simplified_tree(ax1, H, "1. Actual Tree")

    ax2 = fig.add_subplot(gs[1, 0])
    draw_simplified_tree(ax2, tree_td_repr, "2. Top-Down\n(closest to median)")

    ax3 = fig.add_subplot(gs[2, 0])
    draw_simplified_tree(ax3, tree_bu_repr, "3. Bottom-Up (bias=0.0)\n(closest to median)")

    # Bottom row
    for i, bias in enumerate(proximity_levels):
        ax = fig.add_subplot(gs[3, i])
        draw_simplified_tree(ax, bottomup_repr.get(bias), proximity_labels[i])

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


def visualize_local_optimization(G, steiner_function, n_alphas=10,
                                 step_size=50.0, n_restarts=3,
                                 n_random_nodes=5, dest=None):
    """
    Compare original Pareto front with locally optimized trees.
    Shows the front in black, and each alpha's optimized set as colored clouds
    connected to their origin point on the front.
    """

    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Compute original Pareto front
    print("Computing original Pareto front...")
    front, actual = pareto_front_generic(H, steiner_function)

    # Sample alphas for local optimization
    alphas = np.linspace(0, 0.9, n_alphas)
    colors = plt.cm.tab10(np.linspace(0, 1, n_alphas))

    all_optimized_points = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        print(f"Local optimization for alpha={alpha:.2f}...")
        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function(H, alpha)

        pareto_set = local_optimize_tree_pareto_set(
            tree, alpha, step_size=step_size, n_restarts=n_restarts,
            n_random_nodes=n_random_nodes, verbose=False
        )

        for _, wiring, travel in pareto_set:
            all_optimized_points[alpha].append((wiring, travel))

        print(f"  Found {len(pareto_set)} Pareto-optimal configurations")

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(12, 9))

    def scale_data(data):
        return data * scale_factor

    # Plot original Pareto front (black solid line)
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=2, markersize=5,
            label='Original Pareto Front', zorder=3)

    # Plot actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='x', color='black', s=100, linewidth=2, zorder=5,
               label='Actual Tree')

    # Plot locally optimized points for each alpha
    for i, alpha in enumerate(alphas):
        points = all_optimized_points[alpha]
        if not points:
            continue

        opt_x = [scale_data(p[0]) for p in points]
        opt_y = [scale_data(p[1]) for p in points]

        # Plot the optimized cloud
        ax.scatter(opt_x, opt_y, color=colors[i], s=20, alpha=0.5, zorder=4,
                   label=f'Local opt (α={alpha:.1f})')

        # Draw dashed line from origin point on front to each optimized point
        if alpha in front:
            origin_x = scale_data(front[alpha][0])
            origin_y = scale_data(front[alpha][1])

            for px, py in zip(opt_x, opt_y):
                ax.plot([origin_x, px], [origin_y, py],
                        color=colors[i], linewidth=0.5, alpha=0.3, linestyle='--')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax.set_title(f"Pareto Front with Local Optimization ({steiner_function.__name__})",
                 fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_x = front_x + [scale_data(actual[0])]
    all_y = front_y + [scale_data(actual[1])]
    for alpha in alphas:
        for p in all_optimized_points[alpha]:
            all_x.append(scale_data(p[0]))
            all_y.append(scale_data(p[1]))

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def visualize_local_optimization_single_tree(tree, step_size=50.0, n_restarts=5,
                                             max_displacement=300.0,
                                             n_random_nodes=20, n_alphas=11,
                                             dest=None):
    """
    Take a single tree, run local optimization across multiple alpha values,
    and visualize all optimized configurations in the Pareto plane.
    Each alpha starts from the original tree independently.
    """
    # Ensure edge weights on the original tree
    for u, v in tree.edges():
        if 'weight' not in tree[u][v]:
            tree[u][v]['weight'] = point_dist(tree.nodes[u]['pos'], tree.nodes[v]['pos'])
    update_all_distances(tree)

    # Get initial costs
    critical_nodes = get_critical_nodes(tree)
    init_wiring, init_travel = graph_costs(tree, critical_nodes=critical_nodes)

    # Limit internal nodes
    base_node = 0
    internal_nodes = [n for n in tree.nodes()
                      if n != base_node
                      and not tree.nodes[n].get('is_tip', False)
                      and not tree.nodes[n].get('is_base', False)]
    actual_n_random = min(n_random_nodes, len(internal_nodes))

    # Search across multiple alphas
    alphas = np.linspace(0, 1, n_alphas)
    colors = plt.cm.viridis(np.linspace(0, 1, n_alphas))

    all_points = []  # (wiring, travel, alpha)

    for alpha in alphas:
        print(f"Alpha={alpha:.1f}...")

        # Start fresh from the original tree each time
        tree_copy = tree.copy()
        for u, v in tree_copy.edges():
            if 'weight' not in tree_copy[u][v]:
                tree_copy[u][v]['weight'] = point_dist(
                    tree_copy.nodes[u]['pos'], tree_copy.nodes[v]['pos']
                )
        update_all_distances(tree_copy)

        pareto_set = local_optimize_tree_pareto_set(
            tree_copy, alpha, step_size=step_size, n_restarts=n_restarts,
            max_displacement=max_displacement, n_random_nodes=actual_n_random,
            verbose=False
        )
        for _, w, t in pareto_set:
            all_points.append((w, t, alpha))
        print(f"  Found {len(pareto_set)} configurations")

    print(f"Total: {len(all_points)} optimized configurations")

    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(10, 8))

    def scale_data(data):
        return data * scale_factor

    # Plot optimized points colored by alpha
    for i, alpha in enumerate(alphas):
        pts = [(w, t) for w, t, a in all_points if a == alpha]
        if not pts:
            continue
        ax.scatter(
            [scale_data(w) for w, _ in pts],
            [scale_data(t) for _, t in pts],
            color=colors[i], s=20, alpha=0.5, zorder=3,
            label=f'α={alpha:.1f}' if i % 2 == 0 else ''
        )

    # Plot initial tree
    ax.scatter(
        scale_data(init_wiring), scale_data(init_travel),
        marker='*', color='red', s=300, zorder=5, linewidth=2,
        edgecolors='black', label='Initial tree'
    )

    # Draw lines from initial to all optimized points
    for w, t, _ in all_points:
        ax.plot(
            [scale_data(init_wiring), scale_data(w)],
            [scale_data(init_travel), scale_data(t)],
            color='gray', linewidth=0.2, alpha=0.2, zorder=1
        )

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax.set_title(f"Local Optimization Across α Values\n"
                 f"({n_alphas} α's, {n_restarts} restarts, {actual_n_random} nodes)",
                 fontsize=13)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    all_x = [scale_data(init_wiring)] + [scale_data(w) for w, _, _ in all_points]
    all_y = [scale_data(init_travel)] + [scale_data(t) for _, t, _ in all_points]

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, all_points


def visualize_local_chaotization(G, steiner_function, n_alphas=10,
                                 step_size=50.0, n_restarts=3,
                                 max_displacement=200.0, n_random_nodes=5,
                                 dest=None):
    """
    Visualize how trees drift away from the Pareto front under random perturbations.
    Points are colored by how many chaotization steps they've undergone.
    """

    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Compute original Pareto front
    print("Computing original Pareto front...")
    front, actual = pareto_front_generic(H, steiner_function)

    # Sample alphas for chaotization
    alphas = np.linspace(0, 0.9, n_alphas)
    colors = plt.cm.tab10(np.linspace(0, 1, n_alphas))

    all_chaotic_points = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        print(f"Chaotization for alpha={alpha:.2f}...")
        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function(H, alpha)

        configs = local_chaotize_tree_pareto_set(
            tree, alpha, step_size=step_size, n_restarts=n_restarts,
            max_displacement=max_displacement, n_random_nodes=n_random_nodes,
            verbose=False
        )

        # Find closest front alpha for origin
        front_alphas = list(front.keys())
        closest_alpha = min(front_alphas, key=lambda a: abs(a - alpha))
        origin_w, origin_t = front[closest_alpha][0], front[closest_alpha][1]

        for _, wiring, travel, steps in configs:
            all_chaotic_points[alpha].append((wiring, travel, steps, origin_w, origin_t))

        print(f"  Collected {len(configs)} configurations")

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    # Plot original Pareto front
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=2, markersize=5,
            label='Original Pareto Front', zorder=3)

    # Plot actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='x', color='black', s=100, linewidth=2, zorder=5,
               label='Actual Tree')

    # Collect all steps for color normalization
    all_steps = []
    for alpha in alphas:
        for _, _, steps, _, _ in all_chaotic_points[alpha]:
            all_steps.append(steps)

    if all_steps:
        vmin, vmax = min(all_steps), max(all_steps)
    else:
        vmin, vmax = 0, 1

    # Plot chaotic points for each alpha
    for i, alpha in enumerate(alphas):
        points = all_chaotic_points[alpha]
        if not points:
            continue

        # Extract steps for coloring
        steps_list = [s for _, _, s, _, _ in points]

        # Plot points colored by steps (darker = more steps)
        scatter = ax.scatter(
            [scale_data(w) for w, _, _, _, _ in points],
            [scale_data(t) for _, t, _, _, _ in points],
            c=steps_list, cmap='YlOrRd', s=15, alpha=0.5, zorder=4,
            vmin=vmin, vmax=vmax,
            label=f'α={alpha:.1f}' if i == 0 else ''
        )

        # Draw lines from origin to points (thin, transparent)
        for w, t, steps, origin_w, origin_t in points:
            alpha_line = min(1.0, steps / max(vmax, 1))
            ax.plot(
                [scale_data(origin_w), scale_data(w)],
                [scale_data(origin_t), scale_data(t)],
                color='gray', linewidth=0.3, alpha=alpha_line * 0.5, zorder=1
            )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Chaotization steps', fontsize=10)

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax.set_title(f"Pareto Front with Chaotic Perturbations ({steiner_function.__name__})\n"
                 f"Points colored by number of perturbation steps",
                 fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_x = front_x + [scale_data(actual[0])]
    all_y = front_y + [scale_data(actual[1])]
    for alpha in alphas:
        for w, t, _, _, _ in all_chaotic_points[alpha]:
            all_x.append(scale_data(w))
            all_y.append(scale_data(t))

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_steiner_sequence(G, steiner_function, alphas=[0.2, 0.4, 0.6, 0.8, 1.0],
                          dest=None, figsize=(30, 5), min_length_factor=0.5,spacing_factor=1.0):
    """
    Plot the original digitized tree followed by Steiner trees for different alpha values.
    Original tree shows only tips and base (no internal nodes).
    """
    # Ensure edge weights on a copy
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']), np.array(H.nodes[v]['pos']))

    n_plots = 1 + len(alphas)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # ---- Original digitized tree (tips + base only) ----
    ax = axes[0]
    base_node = 0
    tips = [n for n in H.nodes() if H.degree(n) == 1 and n != base_node]
    all_pos = {n: H.nodes[n]['pos'] for n in H.nodes()}

    # Draw edges between all connected nodes (but only base and tips are visible)
    for u, v in H.edges():
        ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                color="#3D2B1F", linewidth=0.8, alpha=0.7)

    ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
               c="#2E8B57", s=50, zorder=3, edgecolors='black', linewidth=0.5)
    tip_x = [all_pos[t][0] for t in tips]
    tip_y = [all_pos[t][1] for t in tips]
    ax.scatter(tip_x, tip_y, c="#FF0000", s=15, zorder=3, edgecolors='black', linewidth=0.3)

    ax.invert_yaxis()
    ax.set_title("Original RSA", fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')

    # ---- Steiner trees for each alpha ----
    for i, alpha in enumerate(alphas):
        ax = axes[i + 1]

        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function(H, alpha, min_length_factor,spacing_factor)

        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        for u, v in tree.edges():
            ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                    color="#3D2B1F", linewidth=0.8, alpha=0.7)

        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=50, zorder=3, edgecolors='black', linewidth=0.5)
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=15, zorder=3, edgecolors='black', linewidth=0.3)

        ax.invert_yaxis()
        ax.set_title(f"α = {alpha}", fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_steiner_sequence_double(G, steiner_function_1, steiner_function_2,
                                 alphas=[0.2, 0.4, 0.6, 0.8, 1.0],
                                 dest=None, figsize=(30, 10),
                                 min_length_factor=0.5, spacing_factor=1.0,
                                 label_1="Method 1", label_2="Method 2"):
    """
    Plot the original digitized tree followed by Steiner trees for two different
    Steiner functions, arranged in two rows.

    Top row: steiner_function_1
    Bottom row: steiner_function_2
    First column in both rows: original RSA
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']),
                                           np.array(H.nodes[v]['pos']))

    n_cols = 1 + len(alphas)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    # ---- Original RSA (both rows, first column) ----
    for row in [0, 1]:
        ax = axes[row, 0]
        draw_tree_simple(ax, H, "Original RSA")

    # ---- Row 0: steiner_function_1 ----
    for i, alpha in enumerate(alphas):
        ax = axes[0, i + 1]
        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function_1(H, alpha, min_length_factor, spacing_factor)
        draw_tree_simple(ax, tree, f"{label_1}\nα = {alpha}")

    # ---- Row 1: steiner_function_2 ----
    for i, alpha in enumerate(alphas):
        ax = axes[1, i + 1]
        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function_2(H, alpha, min_length_factor, spacing_factor)
        draw_tree_simple(ax, tree, f"{label_2}\nα = {alpha}")

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


def draw_tree_simple(ax, tree, title_str, base_node = 0):
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        for u, v in tree.edges():
            ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                    color="#3D2B1F", linewidth=0.8, alpha=0.7)

        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=50, zorder=3, edgecolors='black', linewidth=0.5)
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=15, zorder=3, edgecolors='black', linewidth=0.3)

        ax.invert_yaxis()
        ax.set_title(title_str, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')



def print_tree_debug(G, node=0, parent=None, indent=0):
    """Print tree with degree and children count for each node."""
    deg = G.degree(node)
    ntype = 'B' if node == 0 else ('T' if G.nodes[node].get('is_tip', False) else 'I')
    pos = G.nodes[node].get('pos', [0, 0])

    # Count children (neighbors excluding parent)
    children = [n for n in G.neighbors(node) if n != parent]
    n_children = len(children)

    print(f"{'  ' * indent}{node} ({ntype}) deg={deg} kids={n_children} [{pos[0]:.0f},{pos[1]:.0f}]")

    for child in children:
        print_tree_debug(G, child, node, indent + 1)



def visualize_randomness_ensemble(G, steiner_function, n_random=200,
                                  step_size=50.0, n_restarts=3,
                                  max_displacement=200.0, n_random_nodes=5,
                                  dest=None):
    """
    Comprehensive visualization showing:
    - Pareto front with chaotic perturbations (colored by steps)
    - Random spanning trees, top-down, bottom-up with centroids
    - Satellite tree excluded from chaotization
    """

    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    # Compute original Pareto front
    print("Computing original Pareto front...")
    front, actual = pareto_front_generic(H, steiner_function)

    # Generate random spanning trees (original algorithm)
    print(f"Generating {n_random} random spanning trees...")
    costs_spanning, _ = random_tree(H, n_random)

    # Generate top-down trees
    print(f"Generating {n_random} top-down trees...")
    costs_topdown, _ = generate_trees_from_tips_topdown(H, n_trees=n_random, verbose=False)

    # Generate bottom-up trees
    print(f"Generating {n_random} bottom-up trees...")
    costs_bottomup, _ = generate_trees_from_tips_bottomup(H, n_trees=n_random, verbose=False)

    # Compute centroids
    centroid_spanning = (np.mean([c[0] for c in costs_spanning]),
                         np.mean([c[1] for c in costs_spanning])) if costs_spanning else (0, 0)
    centroid_topdown = (np.mean([c[0] for c in costs_topdown]),
                        np.mean([c[1] for c in costs_topdown])) if costs_topdown else (0, 0)
    centroid_bottomup = (np.mean([c[0] for c in costs_bottomup]),
                         np.mean([c[1] for c in costs_bottomup])) if costs_bottomup else (0, 0)

    # Chaotization at fewer alphas, skipping alpha=0 (satellite tree)
    alphas = np.linspace(0.1, 0.9, 5)  # Skip alpha=0

    all_chaotic_points = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        print(f"Chaotization for alpha={alpha:.2f}...")
        tree = steiner_function(H, alpha)

        configs = local_chaotize_tree_pareto_set(
            tree, step_size=step_size, n_restarts=n_restarts,
            max_displacement=max_displacement, n_random_nodes=n_random_nodes,
            verbose=False
        )

        front_alphas = list(front.keys())
        closest_alpha = min(front_alphas, key=lambda a: abs(a - alpha))
        origin_w, origin_t = front[closest_alpha][0], front[closest_alpha][1]

        for _, wiring, travel, steps in configs:
            all_chaotic_points[alpha].append((wiring, travel, steps, origin_w, origin_t))

        print(f"  Collected {len(configs)} configurations")

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(16, 11))

    def scale_data(data):
        return data * scale_factor

    # Plot original Pareto front (thick black line)
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=3, markersize=6,
            label='Pareto Front', zorder=10)

    # Plot actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=200, linewidth=2, zorder=11,
               label='Actual Tree', edgecolors='white')

    # Plot random spanning trees (small green dots) with centroid
    if costs_spanning:
        ax.scatter([scale_data(c[0]) for c in costs_spanning],
                   [scale_data(c[1]) for c in costs_spanning],
                   marker='.', color='green', s=8, alpha=0.4, zorder=2,
                   label=f'Spanning ({len(costs_spanning)})')
        ax.scatter(scale_data(centroid_spanning[0]), scale_data(centroid_spanning[1]),
                   marker='+', color='darkgreen', s=150, linewidth=2.5, zorder=12)

    # Plot top-down trees (small blue dots) with centroid
    if costs_topdown:
        ax.scatter([scale_data(c[0]) for c in costs_topdown],
                   [scale_data(c[1]) for c in costs_topdown],
                   marker='.', color='blue', s=8, alpha=0.4, zorder=2,
                   label=f'Top-Down ({len(costs_topdown)})')
        ax.scatter(scale_data(centroid_topdown[0]), scale_data(centroid_topdown[1]),
                   marker='+', color='darkblue', s=150, linewidth=2.5, zorder=12)

    # Plot bottom-up trees (small red dots) with centroid
    if costs_bottomup:
        ax.scatter([scale_data(c[0]) for c in costs_bottomup],
                   [scale_data(c[1]) for c in costs_bottomup],
                   marker='.', color='red', s=8, alpha=0.4, zorder=2,
                   label=f'Bottom-Up ({len(costs_bottomup)})')
        ax.scatter(scale_data(centroid_bottomup[0]), scale_data(centroid_bottomup[1]),
                   marker='+', color='darkred', s=150, linewidth=2.5, zorder=12)

    # Plot chaotic points (colored by steps)
    all_steps = []
    for alpha in alphas:
        for _, _, steps, _, _ in all_chaotic_points[alpha]:
            all_steps.append(steps)

    vmin, vmax = (min(all_steps), max(all_steps)) if all_steps else (0, 1)

    for i, alpha in enumerate(alphas):
        points = all_chaotic_points[alpha]
        if not points:
            continue

        steps_list = [s for _, _, s, _, _ in points]

        scatter = ax.scatter(
            [scale_data(w) for w, _, _, _, _ in points],
            [scale_data(t) for _, t, _, _, _ in points],
            c=steps_list, cmap='YlOrRd', s=12, alpha=0.6, zorder=5,
            vmin=vmin, vmax=vmax,
            label='Chaotic' if i == 0 else ''
        )

        # Draw faint lines from origin
        for w, t, steps, origin_w, origin_t in points:
            alpha_line = min(1.0, steps / max(vmax, 1))
            ax.plot(
                [scale_data(origin_w), scale_data(w)],
                [scale_data(origin_t), scale_data(t)],
                color='gray', linewidth=0.2, alpha=alpha_line * 0.3, zorder=1
            )

    # Colorbar for chaotization steps
    if all_steps:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Chaotization steps', fontsize=10)

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Randomness Landscape: Front, Chaotization & Random Trees\n"
                 f"({steiner_function.__name__})",
                 fontsize=15)
    ax.legend(loc='best', fontsize=11, ncol=2, markerscale=2)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_x = front_x + [scale_data(actual[0])]
    all_y = front_y + [scale_data(actual[1])]
    for costs in [costs_spanning, costs_topdown, costs_bottomup]:
        for c in costs:
            all_x.append(scale_data(c[0]))
            all_y.append(scale_data(c[1]))
    all_x.extend([scale_data(centroid_spanning[0]), scale_data(centroid_topdown[0]), scale_data(centroid_bottomup[0])])
    all_y.extend([scale_data(centroid_spanning[1]), scale_data(centroid_topdown[1]), scale_data(centroid_bottomup[1])])
    for alpha in alphas:
        for w, t, _, _, _ in all_chaotic_points[alpha]:
            all_x.append(scale_data(w))
            all_y.append(scale_data(t))

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_random_sequence(G, random_function, n_examples=5, dest=None, figsize=(30, 5)):
    """
    Plot the original digitized tree followed by random trees generated
    by the given random function. Original tree shows only tips and base.

    Args:
        G: Input graph
        random_function: Function that takes (G, n) and returns (costs, trees)
        n_examples: Number of random trees to show
        dest: Optional path to save the plot
        figsize: Figure size
    """
    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']), np.array(H.nodes[v]['pos']))

    # Generate random trees
    print(f"Generating trees with {random_function.__name__}...")
    costs, trees = random_function(H, n_examples)
    print(f"  Got {len(trees)} trees")

    n_plots = 1 + min(n_examples, len(trees))
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # ---- Original digitized tree (tips + base only) ----
    ax = axes[0]
    base_node = 0
    tips = [n for n in H.nodes() if H.degree(n) == 1 and n != base_node]
    all_pos = {n: H.nodes[n]['pos'] for n in H.nodes()}

    for u, v in H.edges():
        ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                color="#3D2B1F", linewidth=0.8, alpha=0.7)

    ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
               c="#2E8B57", s=50, zorder=3, edgecolors='black', linewidth=0.5)
    tip_x = [all_pos[t][0] for t in tips]
    tip_y = [all_pos[t][1] for t in tips]
    ax.scatter(tip_x, tip_y, c="#FF0000", s=15, zorder=3, edgecolors='black', linewidth=0.3)

    ax.invert_yaxis()
    ax.set_title("Original RSA", fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')

    # ---- Random trees ----
    for i, (tree, (wiring, travel)) in enumerate(zip(trees[:n_examples], costs[:n_examples])):
        ax = axes[i + 1]

        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        for u, v in tree.edges():
            ax.plot([all_pos[u][0], all_pos[v][0]], [all_pos[u][1], all_pos[v][1]],
                    color="#3D2B1F", linewidth=0.8, alpha=0.7)

        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=50, zorder=3, edgecolors='black', linewidth=0.5)
        tip_x = [all_pos[t][0] for t in tips]
        tip_y = [all_pos[t][1] for t in tips]
        ax.scatter(tip_x, tip_y, c="#FF0000", s=15, zorder=3, edgecolors='black', linewidth=0.3)

        ax.invert_yaxis()
        # Show costs in title
        ax.set_title(f"{random_function.__name__} #{i + 1}\nL={wiring:.0f}, D={travel:.0f}",
                     fontsize=11, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig


import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase



def plot_randomization_landscape(G, n_trees=500, dest=None):
    """
    Plot trees generated with randomized alpha and optimization rounds.
    Color = optimization level (blue = Pareto-optimal, red = fully random)
    Point size = alpha value (1 = low alpha, 4 = high alpha)
    Blue diamonds (no black perimeter) mark fully Pareto-optimal trees.
    """
    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']), np.array(H.nodes[v]['pos']))

    # Compute Pareto front for reference
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto
    )


    # Compute straightened tree costs
    straight = straighten_tree(H)
    straight_m, straight_s = graph_costs(straight, critical_nodes=get_critical_nodes(H))

    n_tips = len(get_critical_nodes(H)) - 1
    max_rounds = n_tips

    costs = []
    alphas_used = []
    rounds_used = []
    is_optimal = []

    n_random = n_trees - 200

    print(f"Generating {n_trees} randomized trees...")
    print(f"Max rounds (fully Pareto): {max_rounds}")

    # 100 trees at max rounds (fully Pareto-optimal)
    print("  Generating 100 fully Pareto-optimal trees...")
    for i in range(100):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=max_rounds
        )
        if tree is not None:
            critical_nodes = get_critical_nodes(H)
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(max_rounds)
            is_optimal.append(True)

    # 100 trees at 0 rounds (fully random)
    print("  Generating 100 fully random trees...")
    for i in range(100):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=0
        )
        if tree is not None:
            critical_nodes = get_critical_nodes(H)
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(0)
            is_optimal.append(False)

    # Rest with random rounds
    print(f"  Generating {n_random} trees with random rounds...")
    for i in range(n_random):
        alpha = np.random.uniform(0, 1)
        rounds = np.random.randint(0, max_rounds + 1)

        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=rounds
        )
        if tree is not None:
            critical_nodes = get_critical_nodes(H)
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(rounds)
            is_optimal.append(False)

        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{n_random}")

    print(f"  Total: {len(costs)} trees, {sum(is_optimal)} optimal")

    # Try to import config for scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    # Plot Pareto front
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=2, markersize=5,
            label='Pareto Front', zorder=3)

    # Plot actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=200, linewidth=2, zorder=5,
               label='Actual Tree', edgecolors='white')

    # Plot straightened tree
    ax.scatter(scale_data(straight_m), scale_data(straight_s),
               marker='+', color='darkred', s=200, linewidth=2.5, zorder=6,
               label='Straightened Tree')

    # Separate optimal from regular
    opt_idx = [i for i, flag in enumerate(is_optimal) if flag]
    reg_idx = [i for i, flag in enumerate(is_optimal) if not flag]

    # Plot regular points: color = optimization level, size = alpha
    if reg_idx:
        reg_rounds = np.array([rounds_used[i] for i in reg_idx])
        reg_alphas = np.array([alphas_used[i] for i in reg_idx])
        reg_costs_x = [scale_data(costs[i][0]) for i in reg_idx]
        reg_costs_y = [scale_data(costs[i][1]) for i in reg_idx]

        norm_rounds = reg_rounds / max(max_rounds, 1)
        hue = 0.67 * norm_rounds
        saturation = np.ones_like(hue) * 0.8
        value = np.ones_like(hue) * 0.9
        colors_hsv = np.stack([hue, saturation, value], axis=1)
        colors_rgb = hsv_to_rgb(colors_hsv)

        # Point size from alpha: map [0,1] to [10, 80]
        point_sizes = 10 + 70 * reg_alphas

        ax.scatter(reg_costs_x, reg_costs_y, c=colors_rgb, s=point_sizes, zorder=4,
                   edgecolors='none', alpha=0.7, label='Randomized trees')

    # Plot optimal points as blue diamonds, no black perimeter
    if opt_idx:
        opt_costs_x = [scale_data(costs[i][0]) for i in opt_idx]
        opt_costs_y = [scale_data(costs[i][1]) for i in opt_idx]
        opt_alphas = np.array([alphas_used[i] for i in opt_idx])
        opt_point_sizes = 30 + 70 * opt_alphas  # same size scaling, slightly larger base

        ax.scatter(opt_costs_x, opt_costs_y, marker='D', color='blue', s=opt_point_sizes,
                   zorder=7, edgecolors='none', label='Fully Pareto-optimal')

    # Colorbar for optimization rounds
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])

    # Create a colormap that matches the hue progression: red (hue=0) -> blue (hue=0.67)
    n_colors = 256
    hue_values = np.linspace(0.67, 0, n_colors)  # blue -> red
    sat_values = np.ones(n_colors) * 0.8
    val_values = np.ones(n_colors) * 0.9
    hsv_cmap = np.stack([hue_values, sat_values, val_values], axis=1)
    rgb_cmap = hsv_to_rgb(hsv_cmap)
    cmap = LinearSegmentedColormap.from_list('rand_cmap', rgb_cmap)

    norm = plt.Normalize(vmin=0, vmax=max_rounds)
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Optimization rounds\n(0 = fully random)', fontsize=9)

    # Size legend
    ax.text(0.02, 0.98, 'Point size: α value\n(small = low α, large = high α)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Optimization Landscape: Blue = Optimal, Red = Random\n"
                 f"({len(costs)} trees, diamonds = fully Pareto-optimal)",
                 fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_x = front_x + [scale_data(actual[0]), scale_data(straight_m)] + [scale_data(c[0]) for c in costs]
    all_y = front_y + [scale_data(actual[1]), scale_data(straight_s)] + [scale_data(c[1]) for c in costs]

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_optimization_landscape_with_random(G, random_function, n_trees=500, dest=None,
                                            random_color='green', random_marker='o',
                                            random_label=None):
    """
    Plot the full optimization landscape (randomized Steiner trees with color/size
    encoding) PLUS an extra cloud from an external random tree generator.

    Args:
        G: Input graph
        random_function: External random generator, e.g. random_tree_normalized.
                         Must accept (G) and return a list of (wiring, travel) tuples.
        n_trees: Number of randomized Steiner trees
        dest: Optional path to save the plot
        random_color: Edge color for the external random cloud
        random_marker: Marker for the external random cloud
        random_label: Legend label for the external random cloud
    """
    if random_label is None:
        random_label = random_function.__name__

    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']),
                                           np.array(H.nodes[v]['pos']))

    critical_nodes_orig = get_critical_nodes(H)

    # Pareto front
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto
    )

    # Straightened tree
    straight = straighten_tree(H)
    straight_m, straight_s = graph_costs(straight, critical_nodes=critical_nodes_orig)

    # ----- External random generator -----
    print(f"Generating external random trees with {random_label}...")
    costs_ext = random_function(H)
    if isinstance(costs_ext, tuple):
        costs_ext = costs_ext[0]
    print(f"  Got {len(costs_ext)} trees")

    m_ext = np.mean([c[0] for c in costs_ext]) if costs_ext else 0.0
    s_ext = np.mean([c[1] for c in costs_ext]) if costs_ext else 0.0

    # ----- Randomized Steiner trees -----
    n_tips = len(critical_nodes_orig) - 1
    max_rounds = n_tips

    costs = []
    alphas_used = []
    rounds_used = []
    is_optimal = []

    n_random = n_trees - 200

    print(f"Generating {n_trees} randomized Steiner trees...")
    print(f"Max rounds (fully Pareto): {max_rounds}")

    # 100 trees at max rounds
    print("  Generating 100 fully Pareto-optimal trees...")
    for i in range(100):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=max_rounds
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes_orig)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(max_rounds)
            is_optimal.append(True)

    # 100 trees at 0 rounds
    print("  Generating 100 fully random trees...")
    for i in range(100):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=0
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes_orig)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(0)
            is_optimal.append(False)

    # Rest with random rounds
    print(f"  Generating {n_random} trees with random rounds...")
    for i in range(n_random):
        alpha = np.random.uniform(0, 1)
        rounds = np.random.randint(0, max_rounds + 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=rounds
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes_orig)
            costs.append((m, s))
            alphas_used.append(alpha)
            rounds_used.append(rounds)
            is_optimal.append(False)

        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{n_random}")

    print(f"  Total: {len(costs)} Steiner trees, {sum(is_optimal)} optimal")

    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    # Pareto front
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=2, markersize=5,
            label='Pareto Front', zorder=3)

    # Actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=200, linewidth=2, zorder=5,
               label='Actual Tree', edgecolors='white')

    # Straightened tree
    ax.scatter(scale_data(straight_m), scale_data(straight_s),
               marker='+', color='darkred', s=200, linewidth=2.5, zorder=6,
               label='Straightened Tree')

    # ----- External random cloud (empty circles) -----
    if costs_ext:
        ext_x = [scale_data(c[0]) for c in costs_ext]
        ext_y = [scale_data(c[1]) for c in costs_ext]
        ax.scatter(ext_x, ext_y, marker=random_marker, facecolors='none',
                   edgecolors=random_color, s=15, alpha=0.5, linewidth=0.5,
                   zorder=1, label=random_label)

    # External centroid
    ax.scatter(scale_data(m_ext), scale_data(s_ext),
               marker=random_marker, color=random_color, s=150, zorder=8,
               edgecolors='black', linewidth=1.5,
               label=f'{random_label} Mean')

    # Separate optimal from regular Steiner trees
    opt_idx = [i for i, flag in enumerate(is_optimal) if flag]
    reg_idx = [i for i, flag in enumerate(is_optimal) if not flag]

    # Regular Steiner points
    if reg_idx:
        reg_rounds = np.array([rounds_used[i] for i in reg_idx])
        reg_alphas = np.array([alphas_used[i] for i in reg_idx])
        reg_costs_x = [scale_data(costs[i][0]) for i in reg_idx]
        reg_costs_y = [scale_data(costs[i][1]) for i in reg_idx]

        norm_rounds = reg_rounds / max(max_rounds, 1)
        hue = 0.67 * norm_rounds
        saturation = np.ones_like(hue) * 0.8
        value = np.ones_like(hue) * 0.9
        colors_hsv = np.stack([hue, saturation, value], axis=1)
        colors_rgb = hsv_to_rgb(colors_hsv)

        point_sizes = 10 + 70 * reg_alphas

        ax.scatter(reg_costs_x, reg_costs_y, c=colors_rgb, s=point_sizes, zorder=4,
                   edgecolors='none', alpha=0.7, label='Randomized Steiner')

    # Optimal Steiner points (blue diamonds)
    if opt_idx:
        opt_costs_x = [scale_data(costs[i][0]) for i in opt_idx]
        opt_costs_y = [scale_data(costs[i][1]) for i in opt_idx]
        opt_alphas = np.array([alphas_used[i] for i in opt_idx])
        opt_point_sizes = 30 + 70 * opt_alphas

        ax.scatter(opt_costs_x, opt_costs_y, marker='D', color='blue', s=opt_point_sizes,
                   zorder=7, edgecolors='none', label='Fully Pareto-optimal')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    n_colors = 256
    hue_values = np.linspace(0.67, 0, n_colors)
    sat_values = np.ones(n_colors) * 0.8
    val_values = np.ones(n_colors) * 0.9
    hsv_cmap = np.stack([hue_values, sat_values, val_values], axis=1)
    rgb_cmap = hsv_to_rgb(hsv_cmap)
    cmap = LinearSegmentedColormap.from_list('rand_cmap', rgb_cmap)

    norm = plt.Normalize(vmin=0, vmax=max_rounds)
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Optimization rounds\n(0 = fully random)', fontsize=9)

    # Size legend
    ax.text(0.02, 0.98, 'Point size: α value\n(small = low α, large = high α)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Optimization Landscape + {random_label}\n"
                 f"({len(costs)} Steiner + {len(costs_ext)} {random_label} trees)",
                 fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Axis limits
    all_x = front_x + [scale_data(actual[0]), scale_data(straight_m)]
    all_y = front_y + [scale_data(actual[1]), scale_data(straight_s)]
    for c in costs:
        all_x.append(scale_data(c[0]));
        all_y.append(scale_data(c[1]))
    for c in costs_ext:
        all_x.append(scale_data(c[0]));
        all_y.append(scale_data(c[1]))
    all_x.append(scale_data(m_ext));
    all_y.append(scale_data(s_ext))

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax




def draw_tree(ax, tree, title_str, base_node = 0):
        tips = [n for n in tree.nodes() if tree.degree(n) == 1 and n != base_node]
        internals = [n for n in tree.nodes() if n != base_node and tree.degree(n) > 1]
        all_pos = {n: tree.nodes[n]['pos'] for n in tree.nodes()}

        # Draw edges
        for u, v in tree.edges():
            ax.plot([all_pos[u][0], all_pos[v][0]],
                    [all_pos[u][1], all_pos[v][1]],
                    color="#3D2B1F", linewidth=0.8, alpha=0.7)

        # Draw internal nodes
        if internals:
            ix = [all_pos[n][0] for n in internals]
            iy = [all_pos[n][1] for n in internals]
            ax.scatter(ix, iy, c="#A0522D", s=20, zorder=3,
                       edgecolors='black', linewidth=0.3)

        # Draw tips
        tx = [all_pos[n][0] for n in tips]
        ty = [all_pos[n][1] for n in tips]
        ax.scatter(tx, ty, c="#FF0000", s=15, zorder=3,
                   edgecolors='black', linewidth=0.3)

        # Draw base
        ax.scatter(all_pos[base_node][0], all_pos[base_node][1],
                   c="#2E8B57", s=50, zorder=4, edgecolors='black', linewidth=0.5)

        ax.invert_yaxis()
        ax.set_title(title_str, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')


def plot_rsa_and_abstraction(G, dest=None, figsize=(16, 7)):
    """
    Side-by-side plot of the original digitized RSA and its straightened abstraction.

    Left: Original RSA with all nodes and edges
    Right: Abstracted version with only branching nodes, tips, and base,
           connected by straight edges

    Args:
        G: NetworkX graph representing the digitized RSA
        dest: Optional path to save the plot
        figsize: Figure size tuple (width, height)

    Returns:
        fig, (ax1, ax2): Figure and axes objects
    """
    H = straighten_tree(G)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    draw_tree(ax1, G, f"Original RSA\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    draw_tree(ax2, H, f"Abstracted RSA\n({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)")

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, (ax1, ax2)


def compare_fronts_boxplot(graphs, steiner_fn_1, steiner_fn_2,
                           label_1="Method 1", label_2="Method 2",
                           dest=None):
    """
    For each graph, compute the ratio of hypervolumes of the Pareto fronts
    produced by two Steiner functions, and plot as a boxplot.

    Ratio = hypervolume(front_1) / hypervolume(front_2).
    Values > 1 mean front_1 is better (larger dominated area).
    """
    ratios = []

    for i, G in enumerate(graphs):
        print(f"Processing graph {i + 1}/{len(graphs)}...")

        H = copy.deepcopy(G)
        for u, v in H.edges():
            if 'weight' not in H[u][v]:
                H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']),
                                               np.array(H.nodes[v]['pos']))

        front_1, _ = pareto_front_generic(H, steiner_fn_1)
        front_2, _ = pareto_front_generic(H, steiner_fn_2)

        ratio, hv_1, hv_2 = compare_front_hypervolumes(front_1, front_2)
        ratios.append(ratio)
        print(f"  {label_1} HV: {hv_1:.0f}, {label_2} HV: {hv_2:.0f}, ratio: {ratio:.4f}")

    if not ratios:
        print("No valid ratios computed.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(ratios, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')

    x_jitter = np.random.normal(1, 0.04, len(ratios))
    ax.scatter(x_jitter, ratios, alpha=0.6, color='black', s=30, zorder=3)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_ylabel(f"Hypervolume ratio: {label_1} / {label_2}", fontsize=13)
    ax.set_title(f"Pareto Front Comparison\n({len(ratios)} graphs)", fontsize=14)
    ax.set_xticklabels([f"{label_1} vs {label_2}"])

    median = np.median(ratios)
    ax.annotate(f"Median: {median:.4f}", xy=(1, median),
                xytext=(1.15, median), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.grid(True, alpha=0.3, axis='y')

    print(f"\nSummary: {len(ratios)} graphs")
    print(f"  Median ratio: {np.median(ratios):.4f}")
    print(f"  Mean ratio: {np.mean(ratios):.4f}")
    print(f"  Fraction > 1: {np.mean(np.array(ratios) > 1):.2%}")

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, ratios


def comparaplot_four_random(G, n_random=[500,200,300,500], dest=None, verbose=True):
    """
    Plot Pareto front comparing four random tree generators:
    - random_tree_normalized (random spanning tree, normalized)
    - generate_trees_from_tips_topdown (top-down)
    - generate_trees_from_tips_bottomup (bottom-up)
    - pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random
      with optimization_rounds=0 (random Steiner)

    Args:
        G: Input graph
        n_random: Number of random trees per generator
        dest: Optional path to save the plot
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Pareto front
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto
    )

    # 1. Random spanning tree (normalized)
    print("Generating random spanning trees (normalized)...")
    costs_rand = random_tree_normalized(H)[:n_random[0]]
    print(f"  Got {len(costs_rand)} trees")

    # 2. Top-down
    print("Generating top-down trees...")
    costs_td, trees_td = generate_trees_from_tips_topdown(H, n_trees=n_random[1], verbose=verbose)
    print(f"  Got {len(costs_td)} trees")

    # 3. Bottom-up
    print("Generating bottom-up trees...")
    costs_bu, trees_bu = generate_trees_from_tips_bottomup(H, n_trees=n_random[2], verbose=verbose)
    print(f"  Got {len(costs_bu)} trees")

    # 4. Random Steiner (optimization_rounds=0)
    print("Generating random Steiner trees...")
    costs_rs = []
    for i in range(n_random[3]):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=alpha, optimization_rounds=0
        )
        if i%100 == 0:
            print(f"  Got {len(costs_rs)} random Steiner trees")
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs_rs.append((m, s))
    print(f"  Got {len(costs_rs)} trees")

    # Centroids
    def centroid(costs_list):
        if len(costs_list) == 0:
            return 0.0, 0.0
        return np.mean([x[0] for x in costs_list]), np.mean([x[1] for x in costs_list])

    m_rand, s_rand = centroid(costs_rand)
    m_td, s_td = centroid(costs_td)
    m_bu, s_bu = centroid(costs_bu)
    m_rs, s_rs = centroid(costs_rs)

    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    # Pareto front
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, 'o-', color='black', linewidth=2, markersize=5,
            label='Pareto Front', zorder=3)

    # Actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=250, linewidth=2, zorder=6,
               label='Actual Tree', edgecolors='white')

    # Plot each generator with distinct marker/color
    def plot_cloud(costs_list, marker, color, label, zorder):
        if len(costs_list) == 0:
            return
        xs = [scale_data(c[0]) for c in costs_list]
        ys = [scale_data(c[1]) for c in costs_list]
        ax.scatter(xs, ys, marker=marker, color=color, s=12, alpha=0.5,
                   zorder=zorder, edgecolors='none', label=label)

    def plot_centroid(m, s, marker, color, size, label, zorder):
        ax.scatter(scale_data(m), scale_data(s), marker=marker, color=color,
                   s=size, zorder=zorder, edgecolors='black', linewidth=1.5,
                   label=label)

    plot_cloud(costs_rand, 'o', 'green', 'Random Spanning (normalized)', 1)
    plot_cloud(costs_td, 's', 'blue', 'Top-Down', 1)
    plot_cloud(costs_bu, '^', 'orange', 'Bottom-Up', 1)
    plot_cloud(costs_rs, 'v', 'purple', 'Random Steiner', 1)

    plot_centroid(m_rand, s_rand, 'o', 'darkgreen', 150, 'Spanning Mean', 5)
    plot_centroid(m_td, s_td, 's', 'darkblue', 120, 'Top-Down Mean', 5)
    plot_centroid(m_bu, s_bu, '^', 'darkorange', 120, 'Bottom-Up Mean', 5)
    plot_centroid(m_rs, s_rs, 'v', 'darkviolet', 120, 'Rand Steiner Mean', 5)

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Pareto Front: Four Random Tree Generators ({n_random} each)", fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Axis limits
    all_x = front_x + [scale_data(actual[0])]
    all_y = front_y + [scale_data(actual[1])]
    for costs_list in [costs_rand, costs_td, costs_bu, costs_rs]:
        for c in costs_list:
            all_x.append(scale_data(c[0]))
            all_y.append(scale_data(c[1]))
    all_x.extend([scale_data(m_rand), scale_data(m_td), scale_data(m_bu), scale_data(m_rs)])
    all_y.extend([scale_data(s_rand), scale_data(s_td), scale_data(s_bu), scale_data(s_rs)])

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def compare_fixed_spacing_granularity(G, spacing_list=None, min_length_factor=0.5, dest=None):
    """
    Compare Pareto fronts from the fixed-spacing algorithm at different granularities
    against the analytical continuous algorithm.

    Fixed-spacing fronts cycle through line styles: solid, dashed, dotted, dash-dot.
    """
    if spacing_list is None:
        spacing_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Continuous front
    print("Computing continuous Pareto front...")
    front_continuous, actual_continuous = pareto_front_generic(
        H, pareto_steiner_fast_analytical_preserve_tips
    )

    # Fixed-spacing fronts at different granularities
    fronts_fixed = {}
    for spf in spacing_list:
        print(f"Computing fixed-spacing front (spacing_factor={spf})...")

        def steiner_fn(G, alpha, spf=spf):
            return pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
                G, alpha, min_length_factor=min_length_factor, spacing_factor=spf
            )

        front, actual = pareto_front_generic(H, steiner_fn)
        fronts_fixed[spf] = (front, actual)

    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    # Line styles cycle: solid, dashed, dotted, dash-dot
    line_styles = ['-', '--', ':', '-.']

    # Blue-to-green color gradient
    n_fixed = len(spacing_list)
    blue_to_green = []
    for i in range(n_fixed):
        t = i / max(n_fixed - 1, 1)
        r = 0.2
        g = 0.4 + t * 0.3
        b = 0.8 - t * 0.5
        blue_to_green.append((r, g, b))

    # Plot fixed-spacing fronts with alternating line styles
    for idx, spf in enumerate(spacing_list):
        front, _ = fronts_fixed[spf]
        fx = [scale_data(x[0]) for x in front.values()]
        fy = [scale_data(x[1]) for x in front.values()]
        linestyle = line_styles[idx % len(line_styles)]
        ax.plot(fx, fy, marker='', linestyle=linestyle, color=blue_to_green[idx],
                linewidth=2.0, label=f'Fixed spf={spf}', zorder=2, alpha=0.85)

    # Plot continuous front
    fx_c = [scale_data(x[0]) for x in front_continuous.values()]
    fy_c = [scale_data(x[1]) for x in front_continuous.values()]
    ax.plot(fx_c, fy_c, 'o-', color='red', markersize=5, linewidth=2.5,
            label='Continuous (analytical)', zorder=3)

    # Actual tree
    ax.scatter(scale_data(actual_continuous[0]), scale_data(actual_continuous[1]),
               marker='*', color='black', s=250, linewidth=2, zorder=5,
               label='Actual Tree', edgecolors='white')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Pareto Front Granularity: Fixed-Spacing vs Continuous", fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Axis limits
    all_x = fx_c + [scale_data(actual_continuous[0])]
    all_y = fy_c + [scale_data(actual_continuous[1])]
    for spf in spacing_list:
        front, _ = fronts_fixed[spf]
        all_x.extend([scale_data(x[0]) for x in front.values()])
        all_y.extend([scale_data(x[1]) for x in front.values()])

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax

def compare_fixed_spacing_granularity_with_original(G, spacing_list=None, min_length_factor=0.5, dest=None):
    """
    Compare Pareto fronts from fixed-spacing at different granularities,
    the analytical continuous algorithm, and the original tip-preserving algorithm.
    """
    if spacing_list is None:
        spacing_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Continuous front
    print("Computing analytical continuous Pareto front...")
    front_continuous, actual_continuous = pareto_front_generic(
        H, pareto_steiner_fast_analytical_preserve_tips
    )

    # Original tip-preserving front
    print("Computing original tip-preserving Pareto front...")
    front_original, _ = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips
    )

    # Fixed-spacing fronts at different granularities
    fronts_fixed = {}
    for spf in spacing_list:
        print(f"Computing fixed-spacing front (spacing_factor={spf})...")

        def steiner_fn(G, alpha, spf=spf):
            return pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
                G, alpha, min_length_factor=min_length_factor, spacing_factor=spf
            )

        front, actual = pareto_front_generic(H, steiner_fn)
        fronts_fixed[spf] = (front, actual)

    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    fig, ax = plt.subplots(figsize=(14, 10))

    def scale_data(data):
        return data * scale_factor

    line_styles = ['-', '--', ':', '-.']
    n_fixed = len(spacing_list)
    blue_to_green = []
    for i in range(n_fixed):
        t = i / max(n_fixed - 1, 1)
        r = 0.2
        g = 0.4 + t * 0.3
        b = 0.8 - t * 0.5
        blue_to_green.append((r, g, b))

    # Plot fixed-spacing fronts
    for idx, spf in enumerate(spacing_list):
        front, _ = fronts_fixed[spf]
        fx = [scale_data(x[0]) for x in front.values()]
        fy = [scale_data(x[1]) for x in front.values()]
        linestyle = line_styles[idx % len(line_styles)]
        ax.plot(fx, fy, marker='', linestyle=linestyle, color=blue_to_green[idx],
                linewidth=2.0, label=f'Fixed spf={spf}', zorder=2, alpha=0.85)

    # Plot continuous front
    fx_c = [scale_data(x[0]) for x in front_continuous.values()]
    fy_c = [scale_data(x[1]) for x in front_continuous.values()]
    ax.plot(fx_c, fy_c, 'o-', color='red', markersize=5, linewidth=2.5,
            label='Continuous (analytical)', zorder=3)

    # Plot original tip-preserving front
    fx_o = [scale_data(x[0]) for x in front_original.values()]
    fy_o = [scale_data(x[1]) for x in front_original.values()]
    ax.plot(fx_o, fy_o, 's--', color='purple', markersize=5, linewidth=2.5,
            label='Original (preserve tips)', zorder=3)

    # Actual tree
    ax.scatter(scale_data(actual_continuous[0]), scale_data(actual_continuous[1]),
               marker='*', color='black', s=250, linewidth=2, zorder=5,
               label='Actual Tree', edgecolors='white')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(f"Pareto Front Comparison", fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Axis limits
    all_x = fx_c + fx_o + [scale_data(actual_continuous[0])]
    all_y = fy_c + fy_o + [scale_data(actual_continuous[1])]
    for spf in spacing_list:
        front, _ = fronts_fixed[spf]
        all_x.extend([scale_data(x[0]) for x in front.values()])
        all_y.extend([scale_data(x[1]) for x in front.values()])

    if all_x:
        x_min, x_max = calculate_plot_buffer(min(all_x), max(all_x))
        y_min, y_max = calculate_plot_buffer(min(all_y), max(all_y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax