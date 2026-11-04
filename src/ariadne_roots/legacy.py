import numpy as np
from scipy.spatial import ConvexHull


DEFAULT_ALPHAS = np.concatenate([
    np.arange(0, 0.1, 0.001),   # 0 to 0.095: steps of 0.001
    np.arange(0.1, 0.3, 0.01),  # 0.1 to 0.29: steps of 0.01
    np.arange(0.3, 0.6, 0.02),  # 0.3 to 0.58: steps of 0.02
    np.arange(0.6, 1.01, 0.05), # 0.6 to 1.0: steps of 0.05
])
DEFAULT_ALPHAS = np.clip(DEFAULT_ALPHAS, 0.0, 1.0)


def steiner_points_fixed_spacing(p1, p2, spacing):
    """
    Given two points p1 and p2, return evenly spaced intermediate points
    along the line segment with a fixed spacing distance.

    If the segment is shorter than spacing, returns an empty list.
    The number of points is determined entirely by the segment length and spacing.

    Args:
        p1: Starting point
        p2: Ending point
        spacing: Desired distance between consecutive points

    Returns:
        List of intermediate points as tuples
    """
    total_dist = point_dist(p1, p2)

    if total_dist <= spacing:
        return []

    # Number of segments of length 'spacing' that fit
    n_segments = int(total_dist / spacing)

    if n_segments <= 1:
        return []

    midpoints = []
    for i in range(1, n_segments):
        t = i / n_segments
        point = tuple(p1[j] + t * (p2[j] - p1[j]) for j in range(len(p1)))
        midpoints.append(point)

    return midpoints


# Legacy: alternate optimization-landscape visualization.
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


# Legacy: broad randomness-ensemble visualization with local chaotization.
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


# Legacy (exploration.py): Perform Pareto-related calculations for a given random tree generator.
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


# Legacy (exploration.py): Report basic root metrics for a given graph.
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


# Legacy (exploration.py): Build a tree where a fraction of tips are connected optimally (using the
def pareto_steiner_partial_random(G, alpha, optimal_fraction=0.5, min_length_factor=1.0):
    """
    Build a tree where a fraction of tips are connected optimally (using the
    fixed-spacing binary tree algorithm), and the remaining tips are connected randomly
    to existing Steiner points.

    Args:
        G: Original graph
        alpha: Pareto weight
        optimal_fraction: Fraction of tips to connect optimally (0 to 1)
        min_length_factor: Multiplier for minimum edge length

    Returns:
        H: Partially optimal, partially random tree
    """
    assert 0 <= alpha <= 1
    assert 0 <= optimal_fraction <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G)
    min_length = min_length_factor * min_lateral_len
    spacing = min_lateral_len / 2.0

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    # Split tips into optimal and random sets
    all_tips = [n for n in critical_nodes if n != base_node]
    random.shuffle(all_tips)
    n_optimal = int(len(all_tips) * optimal_fraction)
    optimal_tips = set(all_tips[:n_optimal])
    random_tips = set(all_tips[n_optimal:])

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    # ---- Phase 1: Connect optimal tips using the fixed-spacing binary algorithm ----
    optimal_out = set(optimal_tips)

    while optimal_out:
        for u in unpaired_nodes:
            if not H.has_node(u):
                continue
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            invalid_neighbors = []
            closest_neighbor = None
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v) or v not in optimal_out:
                    invalid_neighbors.append(v)
                else:
                    closest_neighbor = v
                    break
            for invalid in invalid_neighbors:
                closest_neighbors[u].remove(invalid)

            if closest_neighbor is None:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[closest_neighbor]["pos"]
            edge_length = point_dist(p1, p2)
            total_root_length = edge_length
            total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]
            cost = pareto_cost(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                alpha=alpha,
            )
            insort(best_edges, (cost, u, closest_neighbor))

        if not best_edges:
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0:
                    nude_zone = min_lateral_len
                    if H.nodes[u]["dist_to_axis_tip"] < nude_zone:
                        continue

            option_a_valid = False
            option_b_valid = False

            if u == base_node:
                option_a_valid = True
            else:
                old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                new_axis_total = old_axis_total + full_dist
                if new_axis_total > 0 and full_dist >= min_lateral_len:
                    if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            best_edges2 = []
            unpaired_nodes = {u, v}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]

            out_nodes.remove(v)
            optimal_out.discard(v)

            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint

                neighbors = []
                for out_node in out_nodes:
                    out_coord = G.nodes[out_node]["pos"]
                    dist = point_dist(midpoint, out_coord)
                    neighbors.append((dist, out_node))
                neighbors = sorted(neighbors)
                closest_neighbors[midpoint_node] = [n for _, n in neighbors]

                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)
                children_count[midpoint_node] = 0

            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            added_nodes += 1

        if not edge_accepted:
            break

    # ---- Phase 2: Connect remaining (random) tips to random Steiner points ----
    # Collect all available Steiner points (nodes with children_count < 2)
    for tip in random_tips:
        if tip not in out_nodes:
            continue

        # Find all nodes that can accept more children
        available_parents = [n for n in H.nodes()
                             if n != base_node
                             and children_count.get(n, 0) < 2
                             and not H.nodes[n].get('is_tip', False)]

        if not available_parents:
            # Fallback: use base
            if children_count.get(base_node, 0) < 1:
                available_parents = [base_node]
            else:
                continue

        # Pick a random parent
        parent = random.choice(available_parents)

        p_parent = np.array(H.nodes[parent]['pos'])
        p_tip = np.array(G.nodes[tip]['pos'])
        dist = point_dist(p_parent, p_tip)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(parent, tip, weight=dist)
        H.nodes[tip]["distance_to_base"] = H.nodes[parent]["distance_to_base"] + dist

        children_count[parent] = children_count.get(parent, 0) + 1
        children_count[tip] = 0
        out_nodes.discard(tip)

    return H


"""
G=get_graph_from_json("C:\\Users\\delivori\\Documents\\jsonArchis\\_set1_day1_20240429-125122_010_plantB_day1.json")
G2=get_graph_from_json("C:\\Users\\delivori\\Documents\\jsonArchis\\_set1_day1_20240429-125122_010_plantD_day1.json")

"""


# Legacy (exploration.py): Take two nodes from available_nodes, create a new internal node connecting them.
def add_node_and_edges_pb(H, available_nodes, next_id, min_orders, shrunk_hull,
                          min_distance, max_order, dist_matrix, proximity_bias=0):
    """
    Take two nodes from available_nodes, create a new internal node connecting them.
    """
    base_node = 0

    # Final case: only base and one other node remain
    if len(available_nodes) == 2 and base_node in available_nodes:
        other = [n for n in available_nodes if n != base_node][0]
        if not H.has_edge(base_node, other):
            H.add_edge(base_node, other)
        return H, next_id, min_orders, dist_matrix

    candidates = [n for n in available_nodes if n != base_node]

    if len(candidates) < 2:
        if len(candidates) == 1:
            if not H.has_edge(base_node, candidates[0]):
                H.add_edge(base_node, candidates[0])
            available_nodes.discard(candidates[0])
        return H, next_id, min_orders, dist_matrix

    # Build valid pairs using precomputed distances
    valid_pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            n1, n2 = candidates[i], candidates[j]
            o1, o2 = min_orders[n1], min_orders[n2]
            min_order_new = min(max(o1, o2 + 1), max(o2, o1 + 1))
            if min_order_new <= max_order:
                dist = dist_matrix.get((n1, n2))
                if dist is None:
                    dist = np.linalg.norm(H.nodes[n1]['pos'] - H.nodes[n2]['pos'])
                    dist_matrix[(n1, n2)] = dist
                    dist_matrix[(n2, n1)] = dist
                valid_pairs.append((dist, n1, n2, min_order_new))

    if not valid_pairs:
        return None, None, None, dist_matrix

    # Sort by distance and select based on proximity_bias
    valid_pairs.sort(key=lambda x: x[0])

    if proximity_bias >= 1.0:
        top_k = min(5, len(valid_pairs))
        chosen = valid_pairs[np.random.randint(top_k)]
    elif proximity_bias <= 0.0:
        chosen = valid_pairs[np.random.randint(len(valid_pairs))]
    else:
        top_k = max(1, int(len(valid_pairs) * proximity_bias))
        chosen = valid_pairs[np.random.randint(top_k)]

    nodedist, n1, n2, min_order_new = chosen

    # Create new internal node
    pos1 = H.nodes[n1]['pos']
    pos2 = H.nodes[n2]['pos']
    midpoint = (pos1 + pos2) / 2
    base_pos = H.nodes[0]['pos']
    upward_bias = 0.3
    biased_center = midpoint + upward_bias * (base_pos - midpoint)
    std = nodedist / 6
    candidate = np.array([
        np.random.normal(biased_center[0], std),
        np.random.normal(biased_center[1], std)
    ])

    # Constrain to hull
    path = MplPath(shrunk_hull.points[shrunk_hull.vertices])
    if not path.contains_points([candidate])[0]:
        best = None
        best_d = float('inf')
        for simplex in shrunk_hull.simplices:
            for k in range(len(simplex)):
                p1 = shrunk_hull.points[simplex[k]]
                p2 = shrunk_hull.points[simplex[(k + 1) % len(simplex)]]
                vec = p2 - p1
                t = np.dot(candidate - p1, vec) / np.dot(vec, vec)
                t = np.clip(t, 0, 1)
                proj = p1 + t * vec
                d = np.linalg.norm(candidate - proj)
                if d < best_d:
                    best_d = d
                    best = proj
        candidate = best

    # Bounce from collisions
    for _ in range(20):
        too_close = False
        for n in H.nodes():
            d = np.linalg.norm(candidate - H.nodes[n]['pos'])
            if d < min_distance:
                if d < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = np.array([np.cos(angle), np.sin(angle)]) * 2 * min_distance
                else:
                    direction = (candidate - H.nodes[n]['pos']) / d
                    push = direction * 2 * min_distance
                candidate = candidate + push
                too_close = True
                break
        if not too_close:
            break

    new_order = max(min_orders[n1], min_orders[n2]) + 1

    new_node_id = next_id
    H.add_node(new_node_id, pos=candidate, is_internal=True, root_deg=new_order)
    H.add_edge(new_node_id, n1)
    H.add_edge(new_node_id, n2)

    available_nodes.discard(n1)
    available_nodes.discard(n2)
    available_nodes.add(new_node_id)

    min_orders[new_node_id] = min_order_new
    del min_orders[n1], min_orders[n2]

    # Update distance matrix
    for old_node in (n1, n2):
        keys_to_remove = [k for k in dist_matrix if old_node in k]
        for k in keys_to_remove:
            del dist_matrix[k]

    new_pos = H.nodes[new_node_id]['pos']
    for n in available_nodes:
        if n != new_node_id and n != base_node:
            dist = np.linalg.norm(new_pos - H.nodes[n]['pos'])
            dist_matrix[(new_node_id, n)] = dist
            dist_matrix[(n, new_node_id)] = dist

    return H, next_id + 1, min_orders, dist_matrix


# Legacy (exploration.py): Generate random trees bottom-up with proximity bias.
def generate_trees_from_tips_bottomup_pb(G_input, n_trees=500, proximity_bias=0,
                                         max_order_factor=1, verbose=True):
    """
    Generate random trees bottom-up with proximity bias.
    """
    critical_nodes = get_critical_nodes(G_input)
    base_node = 0

    positions = {}
    for node in critical_nodes:
        pos = G_input.nodes[node]['pos']
        positions[node] = np.array([pos[0], pos[1]]) if len(pos) == 3 else np.array(pos)

    tip_points = np.array([positions[n] for n in critical_nodes if n != base_node])
    hull = ConvexHull(tip_points)
    hull_centroid = np.mean(hull.points[hull.vertices], axis=0)
    shrink_factor = np.sqrt(0.85)
    shrunk_vertices = hull_centroid + shrink_factor * (hull.points[hull.vertices] - hull_centroid)
    shrunk_hull = ConvexHull(shrunk_vertices)
    shrunk_with_base = np.vstack([shrunk_hull.points[shrunk_hull.vertices], positions[base_node].reshape(1, 2)])
    shrunk_hull = ConvexHull(shrunk_with_base)

    min_distance = get_min_lateral_length(G_input)
    if min_distance <= 0:
        min_distance = 10.0

    max_order_input = get_max_root_order(G_input)
    max_order = int((max_order_input + 1) * max_order_factor)

    if verbose:
        print(f"Input max root_deg: {max_order_input}, Allowed: {max_order}")
        print(f"Proximity bias: {proximity_bias}")
        print(f"Generating {n_trees} trees (bottom-up)...")

    random_trees = []
    costs = []
    trees_generated = 0
    attempts = 0
    max_attempts = n_trees * 100

    while trees_generated < n_trees and attempts < max_attempts:
        attempts += 1

        H = nx.Graph()
        H.add_node(base_node, pos=positions[base_node], is_base=True, root_deg=0)
        for tip in critical_nodes:
            if tip != base_node:
                H.add_node(tip, pos=positions[tip], is_tip=True, root_deg=0)

        available_nodes = set(critical_nodes)
        next_id = max(critical_nodes) + 1
        min_orders = {n: 0 for n in available_nodes if n != base_node}

        # Build initial distance matrix
        dist_matrix = {}
        cand_list = list(available_nodes - {base_node})
        for i in range(len(cand_list)):
            for j in range(i + 1, len(cand_list)):
                n1, n2 = cand_list[i], cand_list[j]
                dist = np.linalg.norm(H.nodes[n1]['pos'] - H.nodes[n2]['pos'])
                dist_matrix[(n1, n2)] = dist
                dist_matrix[(n2, n1)] = dist

        # Main merging loop
        while len(available_nodes) > 2 or (len(available_nodes) == 2 and base_node not in available_nodes):
            if not can_complete_within_order(list(min_orders.values()), max_order):
                H = None
                break

            result = add_node_and_edges_pb(
                H, available_nodes, next_id, min_orders,
                shrunk_hull, min_distance, max_order,
                dist_matrix, proximity_bias
            )
            if result[0] is None:
                H = None
                break
            H, next_id, min_orders, dist_matrix = result

        if H is None:
            continue

        # Final connection to base
        result = add_node_and_edges_pb(
            H, available_nodes, next_id, min_orders,
            shrunk_hull, min_distance, max_order,
            dist_matrix, proximity_bias
        )
        if result[0] is None:
            continue
        H, next_id, min_orders, dist_matrix = result

        for u, v in H.edges():
            if 'weight' not in H[u][v]:
                H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])

        if nx.is_tree(H) and nx.is_connected(H):
            try:
                mactual, sactual = graph_costs(H,get_critical_nodes(H))
                costs.append((mactual, sactual))
                random_trees.append(H)
                trees_generated += 1
                if verbose and trees_generated % 10 == 0:
                    print(f"  Generated {trees_generated} trees")
            except Exception as e:
                if verbose and attempts <= 3:
                    print(f"  Attempt {attempts}: Error - {e}")
        else:
            if verbose and attempts <= 3:
                print(f"  Attempt {attempts}: Invalid tree")

    if verbose:
        print(f"\nGenerated {len(random_trees)} trees after {attempts} attempts")

    return costs, random_trees


# Legacy (exploration.py): Check if a node's current position satisfies minimum distance constraints.
def is_valid_position(tree, node, min_length):
    """
    Check if a node's current position satisfies minimum distance constraints.

    Args:
        tree: NetworkX graph
        node: Node to check
        min_length: Minimum allowed distance to neighbors and tips

    Returns:
        True if position is valid, False otherwise
    """
    p_node = np.array(tree.nodes[node]['pos'])

    # Distance to neighbors (edges)
    for neighbor in tree.neighbors(node):
        p_neighbor = np.array(tree.nodes[neighbor]['pos'])
        if point_dist(p_node, p_neighbor) < min_length:
            return False

    # Distance to all tips
    for n in tree.nodes():
        if tree.nodes[n].get('is_tip', False) and n != node:
            p_tip = np.array(tree.nodes[n]['pos'])
            if point_dist(p_node, p_tip) < min_length:
                return False

    return True


# Legacy (exploration.py): Optimize tree by moving internal nodes, respecting minimum length constraints.
def local_optimize_tree_pareto_set(H, alpha, step_size=50.0, n_restarts=5, n_random_nodes=5,
                                   max_displacement=200.0, max_pareto_size=100,
                                   min_length_factor=0.5, verbose=False):
    """
    Optimize tree by moving internal nodes, respecting minimum length constraints.

    Returns:
        pareto_set: List of (tree, wiring, travel) that dominate the initial tree
        incomparable_set: List of (tree, wiring, travel) incomparable with the initial tree
    """
    # Ensure edge weights exist
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])
    update_all_distances(H)

    min_length = get_min_lateral_length(H, min_length_factor)

    directions = [
        np.array([0, -1]), np.array([1, -1]), np.array([1, 0]), np.array([1, 1]),
        np.array([0, 1]), np.array([-1, 1]), np.array([-1, 0]), np.array([-1, -1]),
    ]
    directions = [d / np.linalg.norm(d) for d in directions]

    critical_nodes = get_critical_nodes(H)
    base_node = 0
    internal_nodes = [n for n in H.nodes()
                      if n != base_node
                      and not H.nodes[n].get('is_tip', False)
                      and not H.nodes[n].get('is_base', False)]

    init_wiring, init_travel = graph_costs(H, critical_nodes=critical_nodes)

    if not internal_nodes:
        return [(H.copy(), init_wiring, init_travel)], []

    pareto_set = []
    incomparable_set = []

    best_cost = pareto_cost(init_wiring, init_travel, alpha)
    max_steps = int(max_displacement / step_size)

    for restart in range(n_restarts):
        if verbose:
            print(f"  Restart {restart + 1}/{n_restarts}")

        np.random.shuffle(internal_nodes)
        selected_nodes = internal_nodes[:n_random_nodes]
        dirs = directions.copy()
        np.random.shuffle(dirs)

        for idx, node in enumerate(selected_nodes):
            original_pos = np.array(H.nodes[node]['pos'])
            if verbose:
                print(f"    Node {node} ({idx + 1}/{len(selected_nodes)})...", flush=True)

            for direction in dirs:
                current_pos = original_pos.copy()

                for step in range(max_steps):
                    new_pos = current_pos + direction * step_size
                    displacement = np.linalg.norm(new_pos - original_pos)

                    if displacement > max_displacement:
                        break

                    H.nodes[node]['pos'] = tuple(new_pos)

                    # Reject invalid positions: stop moving in this direction
                    if not is_valid_position(H, node, min_length):
                        H.nodes[node]['pos'] = tuple(current_pos)
                        break

                    for u, v in H.edges():
                        H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])
                    update_all_distances(H)

                    new_wiring, new_travel = graph_costs(H, critical_nodes=critical_nodes)

                    if new_wiring < init_wiring and new_travel < init_travel:
                        add_to_pareto_set(pareto_set, H, new_wiring, new_travel)
                    elif (new_wiring < init_wiring and new_travel >= init_travel) or \
                            (new_wiring >= init_wiring and new_travel < init_travel):
                        add_to_pareto_set(incomparable_set, H, new_wiring, new_travel)

                    new_cost = pareto_cost(new_wiring, new_travel, alpha)

                    if new_cost < best_cost:
                        best_cost = new_cost
                        current_pos = new_pos.copy()
                    else:
                        H.nodes[node]['pos'] = tuple(current_pos)
                        break

        all_found = pareto_set + incomparable_set
        if all_found:
            best_idx = min(range(len(all_found)),
                           key=lambda i: pareto_cost(all_found[i][1], all_found[i][2], alpha))
            H = all_found[best_idx][0].copy()

    return pareto_set, incomparable_set


# Legacy (exploration.py): Add (tree, wiring, travel) to target_set if non-dominated within that set.
def add_to_pareto_set(target_set, tree, wiring, travel):
    """
    Add (tree, wiring, travel) to target_set if non-dominated within that set.
    Removes any points in the set that are dominated by the new point.
    """
    # Remove points dominated by the new point
    target_set[:] = [p for p in target_set
                     if not (wiring <= p[1] and travel <= p[2]
                             and (wiring < p[1] or travel < p[2]))]

    # Check if new point is dominated by any remaining point
    for existing in target_set:
        _, w_ex, t_ex = existing
        if w_ex <= wiring and t_ex <= travel and (w_ex < wiring or t_ex < travel):
            return False

    target_set.append((tree.copy(), wiring, travel))
    return True


# Legacy (exploration.py): Update distance_to_base for all nodes using BFS.
def update_all_distances(H):
    """Update distance_to_base for all nodes using BFS."""
    base_node = 0
    H.nodes[base_node]['distance_to_base'] = 0

    queue = [base_node]
    visited = {base_node}

    while queue:
        u = queue.pop(0)
        for v in H.neighbors(u):
            if v not in visited:
                visited.add(v)
                H.nodes[v]['distance_to_base'] = (
                        H.nodes[u]['distance_to_base'] + H[u][v]['weight']
                )
                queue.append(v)


# Legacy (exploration.py): Explore the neighborhood of a tree by moving internal nodes randomly,
def local_chaotize_tree_pareto_set(H, step_size=50.0, n_restarts=5,
                                   max_displacement=200.0, n_random_nodes=5,
                                   verbose=False):
    """
    Explore the neighborhood of a tree by moving internal nodes randomly,
    keeping ALL configurations visited.
    Each configuration stores (tree, wiring, travel, steps_taken).
    """
    # Ensure edge weights exist
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])
    update_all_distances(H)

    directions = [
        np.array([0, -1]), np.array([1, -1]), np.array([1, 0]), np.array([1, 1]),
        np.array([0, 1]), np.array([-1, 1]), np.array([-1, 0]), np.array([-1, -1]),
    ]
    directions = [d / np.linalg.norm(d) for d in directions]

    critical_nodes = get_critical_nodes(H)
    base_node = 0
    internal_nodes = [n for n in H.nodes()
                      if n != base_node
                      and not H.nodes[n].get('is_tip', False)
                      and not H.nodes[n].get('is_base', False)]

    actual_n_random = min(n_random_nodes, len(internal_nodes))
    max_steps = int(max_displacement / step_size)

    all_configs = []
    total_steps = 0  # Global step counter

    # Add initial configuration
    init_wiring, init_travel = graph_costs(H, critical_nodes=critical_nodes)
    all_configs.append((H.copy(), init_wiring, init_travel, 0))

    for restart in range(n_restarts):
        if verbose:
            print(f"  Restart {restart + 1}/{n_restarts}")

        np.random.shuffle(internal_nodes)
        selected_nodes = internal_nodes[:actual_n_random]
        dirs = directions.copy()
        np.random.shuffle(dirs)

        for node in selected_nodes:
            original_pos = np.array(H.nodes[node]['pos'])

            for direction in dirs:
                current_pos = original_pos.copy()

                for step in range(max_steps):
                    new_pos = current_pos + direction * step_size
                    displacement = np.linalg.norm(new_pos - original_pos)

                    if displacement > max_displacement:
                        break

                    H.nodes[node]['pos'] = tuple(new_pos)

                    for u, v in H.edges():
                        H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])
                    update_all_distances(H)

                    new_wiring, new_travel = graph_costs(H, critical_nodes=critical_nodes)
                    total_steps += 1
                    if random.random() < 0.1:  # 10% chance to keep this config
                        all_configs.append((H.copy(), new_wiring, new_travel, total_steps))

                    if verbose and step == 0:
                        print(f"    Node {node}: displacement={displacement:.0f}px, "
                              f"wiring={new_wiring:.0f}, travel={new_travel:.0f}")

                    current_pos = new_pos.copy()

        # Reset to a random configuration for the next restart
        if all_configs:
            idx = np.random.randint(len(all_configs))
            H = all_configs[idx][0].copy()

    return all_configs


# Legacy (exploration.py): Return True if node u is at least min_lateral_len away from every tip
def _geometric_nude_zone_ok(H, u, min_lateral_len, is_tip_check=True,
                             is_branch_check=True):
    """
    Return True if node u is at least min_lateral_len away from every tip
    (if is_tip_check) and every branch point (if is_branch_check) currently
    in the tree H.  Uses cached positions; callers should invalidate the
    cache when H changes.

    This is the geometric analogue of the axis-based nude-zone check: instead
    of measuring distance to the two ends of "u's own axis", we measure
    distance to any endpoint of any axis that could be intersected by a new
    branch at u.
    """
    p_u = np.asarray(H.nodes[u]["pos"])
    for n, data in H.nodes(data=True):
        if n == u:
            continue
        if is_tip_check and data.get("is_tip", False):
            if point_dist(p_u, data["pos"]) < min_lateral_len:
                return False
        if is_branch_check and H.degree(n) >= 3:
            if point_dist(p_u, data["pos"]) < min_lateral_len:
                return False
    return True


# Legacy (exploration.py): Geometric (axis-free) version of
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random_geometric(
        G, alpha=None, min_length_factor=0.5, spacing_factor=1.0,
        optimization_rounds=0):
    """
    Geometric (axis-free) version of
    pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random.

    Constraints enforced (identical to the axis-based version):
      * tip preservation            - tips are never used as internal nodes
      * binary branching            - base <= 1 child, others <= 2 children
      * minimum edge length         - every new edge >= min_lateral_len
      * nude zones                  - a branching node must be >= min_lateral_len
                                      from every tip and every other branch point

    The axis-based nude-zone check is replaced by a direct geometric check
    against all tips and all branch points in the current tree.  This is
    strictly more conservative (it also blocks near-merge of two distinct
    axes) but requires no axis metadata.

    The first `optimization_rounds` connections are chosen greedily by
    Pareto cost; the rest are chosen uniformly at random from valid
    candidates.  If alpha is None it is drawn uniformly from [0, 1].

    Notes
    -----
    Trees produced are statistically similar to the axis-based version but
    not identical: the geometric check rejects a small number of choices
    that the axis-based check permits (specifically, choices that would
    bring two different axes closer than min_lateral_len).  This is the
    intended comparison to test whether the axis metadata is functionally
    redundant.
    """
    if alpha is None:
        alpha = np.random.uniform(0, 1)
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}
    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    # Cache: set of (tip positions, branch-point positions).  Invalidated
    # whenever a node is added or a node's degree crosses 2 -> 3.
    tip_positions = []       # list of np.arrays
    branch_positions = []    # list of np.arrays

    def _node_pos(n):
        return np.asarray(H.nodes[n]["pos"])

    def _register_new_node(n):
        """Update geometric caches when n is added to H."""
        if H.nodes[n].get("is_tip", False):
            tip_positions.append(_node_pos(n))
        # A node becomes a branch point only when its degree reaches 3; new
        # nodes start at degree <= 2, so nothing to do here yet.

    def _register_branching(n):
        """Call when n's degree has just become 3."""
        branch_positions.append(_node_pos(n))

    def _nude_zone_ok(u):
        """Geometric nude-zone check for a candidate branching node u."""
        p_u = _node_pos(u)
        for p in tip_positions:
            if point_dist(p_u, p) < min_lateral_len:
                return False
        for p in branch_positions:
            if point_dist(p_u, p) < min_lateral_len:
                return False
        return True

    while added_nodes < len(critical_nodes):
        use_random = (added_nodes - 1) >= optimization_rounds

        # --- candidate generation -------------------------------------------------
        if use_random:
            shuffled_unpaired = list(unpaired_nodes)
            np.random.shuffle(shuffled_unpaired)
            for u in shuffled_unpaired:
                if H.nodes[u].get("is_tip", False):
                    continue
                if not H.has_node(u):
                    continue
                if u != base_node and children_count.get(u, 0) >= 2:
                    continue
                if u == base_node and children_count.get(u, 0) >= 1:
                    continue

                u_has_child = children_count.get(u, 0) > 0
                if u != base_node and u_has_child and not _nude_zone_ok(u):
                    continue

                p1 = H.nodes[u]["pos"]
                valid_tips = []
                for v in out_nodes:
                    p2 = G.nodes[v]["pos"]
                    if point_dist(p1, p2) < min_lateral_len:
                        continue
                    valid_tips.append(v)

                if valid_tips:
                    tip = valid_tips[np.random.randint(len(valid_tips))]
                    insort(best_edges, (np.random.random(), u, tip))
        else:
            for u in unpaired_nodes:
                if H.nodes[u].get("is_tip", False):
                    continue
                if not H.has_node(u):
                    continue
                if u != base_node and children_count.get(u, 0) >= 2:
                    continue
                if u == base_node and children_count.get(u, 0) >= 1:
                    continue

                u_has_child = children_count.get(u, 0) > 0
                if u != base_node and u_has_child and not _nude_zone_ok(u):
                    continue

                best_tip = None
                best_cost = float("inf")
                p1 = H.nodes[u]["pos"]
                for v in out_nodes:
                    p2 = G.nodes[v]["pos"]
                    edge_length = point_dist(p1, p2)
                    if edge_length < min_lateral_len:
                        continue
                    total_travel = edge_length + H.nodes[u]["distance_to_base"]
                    cost = pareto_cost(edge_length, total_travel, alpha)
                    if cost < best_cost:
                        best_cost = cost
                        best_tip = v
                if best_tip is not None:
                    insort(best_edges, (best_cost, u, best_tip))

        if not best_edges:
            break

        # --- accept one edge ------------------------------------------------------
        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)
            if full_dist < min_lateral_len:
                continue

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child and not _nude_zone_ok(u):
                continue

            edge_accepted = True

            # Invalidate stale best_edges entries for this tip.
            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            # Add the tip.
            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            out_nodes.remove(v)
            _register_new_node(v)

            old_deg_u = children_count.get(u, 0)
            children_count[u] = old_deg_u + 1
            children_count[v] = 0

            # If u just became a branch point (2 -> 3), register it.
            if u != base_node and old_deg_u == 1:
                _register_branching(u)

            # Fixed-spacing Steiner points along u -> v.
            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []
            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint
                children_count[midpoint_node] = 1
                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)
                # Steiner points are not tips and not branch points.

            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            added_nodes += 1

        if not edge_accepted:
            break

    return H







if __name__ == "__main__":
    from ariadne_roots.main import get_graph_from_json

    simpgraphs = []
    folder_path = "C:\\Users\\delivori\\Documents\\jsonArchis"
    json_files = list(Path(folder_path).glob("*.json"))

    graphs = []
    for filepath in json_files:
        G = get_graph_from_json(str(filepath))
        graphs.append(G)
        print(f"Loaded: {filepath.name}")

    graphs.pop(0)

    for G in graphs:
        simpgraphs.append(straighten_tree(G))


# Legacy (exploration.py): Debug comparison between fixed-spacing and continuous Steiner algorithms.
def debug_compare_steiner_algorithms(G, alpha=0.5, min_length_factor=0.5, spacing_factor=1.0):
    """
    Debug comparison between fixed-spacing and continuous Steiner algorithms.
    Prints step-by-step costs and returns both trees for visual inspection.
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']),
                                           np.array(H.nodes[v]['pos']))

    critical_nodes = get_critical_nodes(H)

    print(f"{'=' * 60}")
    print(f"DEBUG COMPARISON: alpha={alpha}, min_length_factor={min_length_factor}")
    print(f"{'=' * 60}")

    # Original graph costs
    orig_w, orig_t = graph_costs(H, critical_nodes=critical_nodes)
    print(f"\nOriginal graph: wiring={orig_w:.1f}, travel={orig_t:.1f}")

    # Fixed-spacing
    print(f"\n--- Fixed-Spacing ---")
    tree_fixed = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
        H, alpha=alpha, min_length_factor=min_length_factor, spacing_factor=spacing_factor
    )
    if tree_fixed is not None:
        fw, ft = graph_costs(tree_fixed, critical_nodes=critical_nodes)
        n_tips_fixed = len([n for n in tree_fixed.nodes() if tree_fixed.nodes[n].get('is_tip', False)])
        n_internal_fixed = len([n for n in tree_fixed.nodes()
                                if not tree_fixed.nodes[n].get('is_tip', False)
                                and not tree_fixed.nodes[n].get('is_base', False)])
        print(f"  Result: wiring={fw:.1f}, travel={ft:.1f}")
        print(f"  Nodes: {tree_fixed.number_of_nodes()} ({n_tips_fixed} tips, {n_internal_fixed} internal)")
    else:
        fw, ft = float('inf'), float('inf')
        print(f"  FAILED to build tree")

    # Continuous
    print(f"\n--- Continuous ---")
    tree_cont = pareto_steiner_fast_continuous_preserve_tips(
        H, alpha=alpha, min_length_factor=min_length_factor, spacing_factor=spacing_factor
    )
    if tree_cont is not None:
        cw, ct = graph_costs(tree_cont, critical_nodes=critical_nodes)
        n_tips_cont = len([n for n in tree_cont.nodes() if tree_cont.nodes[n].get('is_tip', False)])
        n_internal_cont = len([n for n in tree_cont.nodes()
                               if not tree_cont.nodes[n].get('is_tip', False)
                               and not tree_cont.nodes[n].get('is_base', False)])
        print(f"  Result: wiring={cw:.1f}, travel={ct:.1f}")
        print(f"  Nodes: {tree_cont.number_of_nodes()} ({n_tips_cont} tips, {n_internal_cont} internal)")
    else:
        cw, ct = float('inf'), float('inf')
        print(f"  FAILED to build tree")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"  Wiring:  fixed={fw:.1f}, continuous={cw:.1f} (diff={cw - fw:+.1f}, {(cw / fw - 1) * 100:+.1f}%)")
    print(f"  Travel:  fixed={ft:.1f}, continuous={ct:.1f} (diff={ct - ft:+.1f}, {(ct / ft - 1) * 100:+.1f}%)")

    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    draw_tree(axes[0], H, f"Original\n(w={orig_w:.0f}, t={orig_t:.0f})")

    if tree_fixed is not None:
        draw_tree(axes[1], tree_fixed, f"Fixed-Spacing\n(w={fw:.0f}, t={ft:.0f})")
    else:
        axes[1].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[1].transAxes)

    if tree_cont is not None:
        draw_tree(axes[2], tree_cont, f"Continuous\n(w={cw:.0f}, t={ct:.0f})")
    else:
        axes[2].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    plt.show()

    return tree_fixed, tree_cont


# Legacy (visualisation.py): Plot Pareto front comparing two random tree generators.
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


# Legacy (visualisation.py): print_tree
def print_tree(G, node=0, parent=None, indent=0):
    pos = G.nodes[node].get('pos', [0,0])
    ntype = 'B' if node==0 else ('T' if G.nodes[node].get('is_tip',False) else 'I')
    print(f"{'  '*indent}{node} ({ntype}) [{pos[0]:.0f},{pos[1]:.0f}]")
    for child in G.neighbors(node):
        if child != parent:
            print_tree(G, child, node, indent+1)


# Legacy (visualisation.py): Master plot: Pareto front comparison surrounded by example tree graphs.
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


# Legacy (visualisation.py): Master plot comparing random spanning trees, top-down, and bottom-up
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


# Legacy (visualisation.py): Compare top-down vs bottom-up with varying proximity bias.
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


# Legacy (visualisation.py): Compare original Pareto front with locally optimized trees.
def visualize_local_optimization(G, steiner_function, n_alphas=10,
                                 step_size=50.0, n_restarts=3,
                                 n_random_nodes=5, dest=None):
    """
    Compare original Pareto front with locally optimized trees.
    Solid circles = trees dominating their origin.
    Transparent circles = trees incomparable with their origin.
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Compute original Pareto front
    print("Computing original Pareto front...")
    front, actual = pareto_front_generic(H, steiner_function)
    cleaned_front = clean_pareto_front(front)

    # Sample alphas for local optimization
    alphas = np.linspace(0.05, 0.9, n_alphas)
    colors = plt.cm.tab10(np.linspace(0, 1, n_alphas))

    dominating_points = {alpha: [] for alpha in alphas}
    incomparable_points = {alpha: [] for alpha in alphas}
    origin_points = {}

    for alpha in alphas:
        print(f"Local optimization for alpha={alpha:.4f}...")
        if alpha == 0:
            tree = satellite_tree(H)
        else:
            tree = steiner_function(H, alpha)

        if tree is not None:
            w, t = graph_costs(tree, critical_nodes=critical_nodes)
            origin_points[alpha] = (w, t)

        pareto_set, incomparable_set = local_optimize_tree_pareto_set(
            tree, alpha, step_size=step_size, n_restarts=n_restarts,
            n_random_nodes=n_random_nodes, verbose=False
        )

        for _, wiring, travel in pareto_set:
            dominating_points[alpha].append((wiring, travel))

        for _, wiring, travel in incomparable_set:
            incomparable_points[alpha].append((wiring, travel))

        print(f"  Dominating: {len(pareto_set)}, Incomparable: {len(incomparable_set)}")

    # Scaling
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

    # Plot cleaned Pareto front
    cleaned_x = [scale_data(v[0]) for v in cleaned_front.values()]
    cleaned_y = [scale_data(v[1]) for v in cleaned_front.values()]
    ax.plot(cleaned_x, cleaned_y, '-', color='black', linewidth=2,
            label='Cleaned Pareto Front', zorder=2)

    # Plot origin points
    if origin_points:
        ox = [scale_data(p[0]) for p in origin_points.values()]
        oy = [scale_data(p[1]) for p in origin_points.values()]
        ax.scatter(ox, oy, marker='o', color='black', s=40, zorder=3,
                   label='Origin points', edgecolors='white', linewidth=0.5)

    # Plot actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='x', color='black', s=100, linewidth=2, zorder=5,
               label='Actual Tree')

    # Plot local optimization results
    for i, alpha in enumerate(alphas):
        # Solid circles for dominating points
        dom_pts = dominating_points[alpha]
        if dom_pts:
            dom_x = [scale_data(p[0]) for p in dom_pts]
            dom_y = [scale_data(p[1]) for p in dom_pts]
            ax.scatter(dom_x, dom_y, color=colors[i], s=20, alpha=0.7,
                       zorder=4, edgecolors='none')

        # Transparent circles for incomparable points
        inc_pts = incomparable_points[alpha]
        if inc_pts:
            inc_x = [scale_data(p[0]) for p in inc_pts]
            inc_y = [scale_data(p[1]) for p in inc_pts]
            ax.scatter(inc_x, inc_y, facecolors='none', edgecolors=colors[i],
                       s=30, alpha=0.8, zorder=4, linewidth=1.5)

        # Dashed lines from origin to all points
        if alpha in origin_points:
            origin_x = scale_data(origin_points[alpha][0])
            origin_y = scale_data(origin_points[alpha][1])

            for px, py in zip(dom_x, dom_y) if dom_pts else []:
                ax.plot([origin_x, px], [origin_y, py],
                        color=colors[i], linewidth=0.5, alpha=0.3, linestyle='--')
            for px, py in zip(inc_x, inc_y) if inc_pts else []:
                ax.plot([origin_x, px], [origin_y, py],
                        color=colors[i], linewidth=0.5, alpha=0.3, linestyle='--')

        # Legend entry
        if dom_pts or inc_pts:
            ax.scatter([], [], color=colors[i], s=20, alpha=0.7,
                       label=f'α={alpha:.4f}')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax.set_title(f"Pareto Front with Local Optimization ({steiner_function.__name__})",
                 fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_x = cleaned_x + [scale_data(actual[0])]
    all_y = cleaned_y + [scale_data(actual[1])]
    for alpha in alphas:
        if alpha in origin_points:
            all_x.append(scale_data(origin_points[alpha][0]))
            all_y.append(scale_data(origin_points[alpha][1]))
        for p in dominating_points[alpha]:
            all_x.append(scale_data(p[0]))
            all_y.append(scale_data(p[1]))
        for p in incomparable_points[alpha]:
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


# Legacy (visualisation.py): Take a single tree, run local optimization across multiple alpha values,
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


# Legacy (visualisation.py): Visualize how trees drift away from the Pareto front under random perturbations.
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


# Legacy (visualisation.py): Print tree with degree and children count for each node.
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







import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


# Legacy (animating.py): Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
        G, alpha, min_length_factor=1 / 2, spacing_factor=1.0, verbose=False, record=False):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.

    Steiner points are placed at fixed spacing along edges. Points within
    min_lateral_len of existing branch points or tips are excluded.

    If record=True, returns (H, frames) where frames is a list of intermediate
    tree states after each edge addition.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1
    iteration = 1

    frames = []
    if record:
        frames.append(H.copy())

    while added_nodes < len(critical_nodes):
        for u in unpaired_nodes:
            if H.nodes[u].get("is_tip", False):
                continue
            if not H.has_node(u):
                continue
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            best_tip = None
            best_cost = float('inf')

            for v in out_nodes:
                p1 = H.nodes[u]["pos"]
                p2 = G.nodes[v]["pos"]
                edge_length = point_dist(p1, p2)

                if edge_length < min_lateral_len:
                    continue

                u_has_child = children_count.get(u, 0) > 0
                if u != base_node and u_has_child:
                    if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                        continue
                    if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                        continue

                total_root_length = edge_length
                total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]

                cost = pareto_cost(
                    total_root_length=total_root_length,
                    total_travel_distance=total_travel_distance,
                    alpha=alpha,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_tip = v

            if best_tip is not None:
                insort(best_edges, (best_cost, u, best_tip))

        if not best_edges:
            if verbose:
                print(f"  Step {iteration}: no best_edges, breaking")
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child:
                if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                    continue
                if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                    continue

            option_a_valid = False
            option_b_valid = False

            if u == base_node:
                option_a_valid = True
            else:
                if u_has_child:
                    old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                    new_axis_total = old_axis_total + full_dist
                    if new_axis_total > 0 and full_dist >= min_lateral_len:
                        if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                            option_a_valid = True
                else:
                    if full_dist >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            if verbose:
                print(f"  Step {iteration}: u={u} -> tip {v}, option={chosen_option}, "
                      f"edge_len={full_dist:.1f}, cost={cost:.1f}")

            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            out_nodes.remove(v)

            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint
                children_count[midpoint_node] = 1
                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)

            line_nodes = [u] + midpoint_nodes + [v]
            for i in range(len(line_nodes) - 1):
                n1 = line_nodes[i]
                n2 = line_nodes[i + 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        H[n1][n2]["weight"] + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            # Remove nodes too close to the new branch point u
            current = u
            while True:
                parents = [n for n in H.neighbors(current)
                           if H.nodes[n].get("distance_to_base", float('inf'))
                           < H.nodes[current].get("distance_to_base", float('inf'))]
                if not parents:
                    break
                current = parents[0]
                if node_dist(H, current, u) < min_lateral_len:
                    unpaired_nodes.discard(current)
                else:
                    break

            for child in list(H.neighbors(u)):
                if H.nodes[child].get("distance_to_base", 0) > H.nodes[u].get("distance_to_base", 0):
                    current = child
                    while True:
                        if node_dist(H, current, u) < min_lateral_len:
                            unpaired_nodes.discard(current)
                            grandchildren = [n for n in H.neighbors(current)
                                             if H.nodes[n].get("distance_to_base", 0)
                                             > H.nodes[current].get("distance_to_base", 0)]
                            if not grandchildren:
                                break
                            current = grandchildren[0]
                        else:
                            break

            added_nodes += 1
            iteration += 1

            if record:
                frames.append(H.copy())

        if not edge_accepted:
            if verbose:
                print(f"  Step {iteration}: no edge accepted, breaking")
            break

    if record:
        return H, frames
    return H


# Legacy (animating.py): Animate tree construction frames with fixed zoom level throughout.
def animate_tree_construction(frames, G=None, dest=None, fps=2, figsize=(8, 10)):
    """
    Animate tree construction frames with fixed zoom level throughout.
    Tips from the input graph G are shown as faint gray dots from the start.

    Args:
        frames: List of NetworkX graphs (intermediate tree states)
        G: Original input graph. If provided, its tips are shown as background.
        dest: Path to save GIF. If None, displays animation.
        fps: Frames per second
        figsize: Figure size
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Background tips from input graph
    background_tips = []
    if G is not None:
        critical_nodes = get_critical_nodes(G)
        background_tips = [n for n in critical_nodes if n != 0]
        bg_pos = {n: G.nodes[n]['pos'] for n in background_tips}

    # Compute global bounds from the final frame
    final_tree = frames[-1]
    all_pos = [final_tree.nodes[n]['pos'] for n in final_tree.nodes()]
    xs = [p[0] for p in all_pos]
    ys = [p[1] for p in all_pos]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_buf = 0.1 * (x_max - x_min)
    y_buf = 0.1 * (y_max - y_min)

    def update(i):
        ax.clear()

        # Draw background tips (input)
        if background_tips:
            bg_x = [bg_pos[n][0] for n in background_tips]
            bg_y = [bg_pos[n][1] for n in background_tips]
            ax.scatter(bg_x, bg_y, c='lightgray', s=20, zorder=1,
                       edgecolors='gray', linewidth=0.3, alpha=0.5,
                       label='Input tips')

        # Draw current tree
        draw_tree_simple(ax, frames[i], f"Step {i + 1}/{len(frames)}")

        ax.set_xlim(x_min - x_buf, x_max + x_buf)
        ax.set_ylim(y_min - y_buf, y_max + y_buf)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, repeat=True)

    if dest is not None:
        ani.save(dest, writer='pillow', fps=fps, dpi=100)
        plt.close(fig)
        print(f"GIF saved to {dest}")
    else:
        plt.show()

    return ani


# Legacy notebook demo (disabled so this module can be imported):
# tree, frames = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
#     graphs[2], alpha=0.5, record=True
# )
# ani = animate_tree_construction(frames, G=graphs[2], fps=2)
# from IPython.display import HTML
# HTML(ani.to_jshtml())


# Legacy (animating.py): Continuous Pareto Steiner tree that preserves tip integrity.
def pareto_steiner_fast_analytical_preserve_tips(G, alpha, min_length_factor=1 / 2,
                                                 spacing_factor=1.0, verbose=False, record=False):
    """
    Continuous Pareto Steiner tree that preserves tip integrity.
    Uses analytical solution for optimal attachment point.

    If record=True, returns (H, frames) where frames is a list of intermediate
    tree states after each edge addition.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["is_base"] = True

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    frames = []
    if record:
        frames.append(H.copy())

    # Step 1: Connect base to the closest tip
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    if dist < min_length:
        if record:
            return None, frames
        return None

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.nodes[closest_tip]["is_tip"] = True
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)
    children_count = {base_node: 1, closest_tip: 0}

    if verbose:
        print(f"  Step 1: base -> tip {closest_tip}, dist={dist:.1f}")

    if record:
        frames.append(H.copy())

    iteration = 2
    while out_nodes:
        best_cost = float('inf')
        best_action = None
        best_cost_small = float('inf')
        best_action_small = None

        for u, v in H.edges():
            # Orient edge so u is closer to base
            if H.nodes[u].get("distance_to_base", float('inf')) > H.nodes[v].get("distance_to_base", float('inf')):
                u, v = v, u

            p_u = np.array(H.nodes[u]['pos'])
            p_v = np.array(H.nodes[v]['pos'])

            for tip in out_nodes:
                cost, point, attach_type = find_optimal_t_on_analytical(
                    p_u=p_u,
                    p_v=p_v,
                    p_tip=G.nodes[tip]['pos'],
                    dist_to_base_u=H.nodes[u]['distance_to_base'],
                    alpha=alpha,
                    min_length=min_length
                )

                if attach_type == 'edge':
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (u, v, tip, point)
                elif attach_type == 'small_edge':
                    if cost < best_cost_small:
                        best_cost_small = cost
                        best_action_small = (u, v, tip, point)

        if best_action is not None:
            u, v, tip, best_point = best_action
            if verbose:
                print(f"  Step {iteration}: edge({u},{v}) -> tip {tip}, cost={best_cost:.1f}")
        elif best_action_small is not None:
            u, v, tip, best_point = best_action_small
            if verbose:
                print(f"  Step {iteration}: small_edge({u},{v}) -> tip {tip}, cost={best_cost_small:.1f}")
        else:
            if verbose:
                print(f"  Step {iteration}: no valid action, breaking")
            break

        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)

        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        children_count[new_node] = 2
        children_count[tip] = 0

        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(
            H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v
        )

        out_nodes.remove(tip)
        iteration += 1

        if record:
            frames.append(H.copy())

    if record:
        return H, frames
    return H


# Legacy (animating.py): Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
        G, alpha=None, min_length_factor=0.5, spacing_factor=1.0,
        optimization_rounds=0, verbose=False, record=False):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.

    For the first `optimization_rounds` iterations, uses Pareto-optimal tip selection
    via optimal_steiner_step. After that, switches to random tip selection among
    valid candidates.

    If record=True, returns (H, frames) where frames is a list of intermediate
    tree states after each edge addition.
    """
    if alpha is None:
        alpha = np.random.uniform(0, 1)
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    frames = []
    if record:
        frames.append(H.copy())

    while added_nodes < len(critical_nodes):
        # Determine mode based on iteration count
        use_random = (added_nodes - 1) >= optimization_rounds

        if use_random:
            # RANDOM MODE: shuffle unpaired nodes, pick random valid tip for each
            shuffled_unpaired = list(unpaired_nodes)
            np.random.shuffle(shuffled_unpaired)

            if use_random:
                # RANDOM MODE: pick a random valid (u, v) pair directly
                shuffled_unpaired = list(unpaired_nodes)
                np.random.shuffle(shuffled_unpaired)
                shuffled_out = list(out_nodes)
                np.random.shuffle(shuffled_out)

                # Phase 1: find a valid pair
                found_pair = False
                for u in shuffled_unpaired:
                    if H.nodes[u].get("is_tip", False):
                        continue
                    if not H.has_node(u):
                        continue
                    if u != base_node and children_count.get(u, 0) >= 2:
                        continue
                    if u == base_node and children_count.get(u, 0) >= 1:
                        continue

                    u_has_child = children_count.get(u, 0) > 0
                    if u != base_node and u_has_child:
                        if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                            continue
                        if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                            continue

                    p1 = H.nodes[u]["pos"]

                    for v in shuffled_out:
                        p2 = G.nodes[v]["pos"]
                        edge_length = point_dist(p1, p2)

                        if edge_length < min_lateral_len:
                            continue

                        insort(best_edges, (np.random.random(), u, v))
                        found_pair = True
                        break

                    if found_pair:
                        break

                # Phase 2: fallback — pick any (u, v) regardless of constraints
                if not found_pair:
                    if verbose:
                        print("  WARNING: No valid pair found, falling back to unconstrained pair")

                    for u in shuffled_unpaired:
                        if H.nodes[u].get("is_tip", False):
                            continue
                        if u != base_node and children_count.get(u, 0) >= 2:
                            continue
                        if u == base_node and children_count.get(u, 0) >= 1:
                            continue
                        if shuffled_out:
                            insort(best_edges, (np.random.random(), u, shuffled_out[0]))
                            found_pair = True
                            break

                if not found_pair:
                    if verbose:
                        print("  ERROR: No pair possible at all")
                    break
        else:
            # OPTIMAL MODE: use Pareto-optimal tip selection
            for u in unpaired_nodes:
                if H.nodes[u].get("is_tip", False):
                    continue
                if not H.has_node(u):
                    continue
                if u != base_node and children_count.get(u, 0) >= 2:
                    continue
                if u == base_node and children_count.get(u, 0) >= 1:
                    continue

                best_tip = None
                best_cost = float('inf')

                for v in out_nodes:
                    p1 = H.nodes[u]["pos"]
                    p2 = G.nodes[v]["pos"]
                    edge_length = point_dist(p1, p2)

                    if edge_length < min_lateral_len:
                        continue

                    u_has_child = children_count.get(u, 0) > 0
                    if u != base_node and u_has_child:
                        if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                            continue
                        if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                            continue

                    total_root_length = edge_length
                    total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]

                    cost = pareto_cost(
                        total_root_length=total_root_length,
                        total_travel_distance=total_travel_distance,
                        alpha=alpha,
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_tip = v

                if best_tip is not None:
                    insort(best_edges, (best_cost, u, best_tip))

        if not best_edges:
            if verbose:
                print(f"  No best_edges. Remaining tips: {len(out_nodes)}")
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child:
                if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                    continue
                if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                    continue

            option_a_valid = False
            option_b_valid = False

            if u == base_node:
                option_a_valid = True
            else:
                if u_has_child:
                    old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                    new_axis_total = old_axis_total + full_dist
                    if new_axis_total > 0 and full_dist >= min_lateral_len:
                        if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                            option_a_valid = True
                else:
                    if full_dist >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            if verbose:
                print(f"  Step: u={u} -> tip {v}, option={chosen_option}, "
                      f"edge_len={full_dist:.1f}")

            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            out_nodes.remove(v)

            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint
                children_count[midpoint_node] = 1
                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)

            line_nodes = [u] + midpoint_nodes + [v]
            for i in range(len(line_nodes) - 1):
                n1 = line_nodes[i]
                n2 = line_nodes[i + 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        H[n1][n2]["weight"] + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            # Remove nodes too close to the new branch point u
            current = u
            while True:
                parents = [n for n in H.neighbors(current)
                           if H.nodes[n].get("distance_to_base", float('inf'))
                           < H.nodes[current].get("distance_to_base", float('inf'))]
                if not parents:
                    break
                current = parents[0]
                if node_dist(H, current, u) < min_lateral_len:
                    unpaired_nodes.discard(current)
                else:
                    break

            for child in list(H.neighbors(u)):
                if H.nodes[child].get("distance_to_base", 0) > H.nodes[u].get("distance_to_base", 0):
                    current = child
                    while True:
                        if node_dist(H, current, u) < min_lateral_len:
                            unpaired_nodes.discard(current)
                            grandchildren = [n for n in H.neighbors(current)
                                             if H.nodes[n].get("distance_to_base", 0)
                                             > H.nodes[current].get("distance_to_base", 0)]
                            if not grandchildren:
                                break
                            current = grandchildren[0]
                        else:
                            break

            added_nodes += 1

            if record:
                frames.append(H.copy())

        if not edge_accepted:
            if verbose:
                print(f"  No edge accepted, breaking")
            break

    if record:
        return H, frames
    return H


# Legacy notebook demo (disabled so this module can be imported):
# tree, frames = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
#     graphs[2], alpha=0.5, record=True, optimization_rounds=5
# )
# ani = animate_tree_construction(frames, G=graphs[2], fps=2)
# from IPython.display import HTML
# HTML(ani.to_jshtml())


# Legacy (benchmarking.py): Time a steiner function over multiple runs.
def benchmark_steiner(steiner_fn, G, alpha=0.5, n_runs=10, **kwargs):
    """
    Time a steiner function over multiple runs.
    Returns (mean_time, std_time).
    """
    times = []
    for _ in range(n_runs):
        # Warm-up run (not timed) to avoid cold cache effects
        _ = steiner_fn(G, alpha=alpha, **kwargs)

        start = time.perf_counter()
        _ = steiner_fn(G, alpha=alpha, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


# Legacy benchmark demo (disabled so this module can be imported):
# mean_1, std_1 = benchmark_steiner(
#     pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto, G, n_runs=10
# )
# mean_2, std_2 = benchmark_steiner(
#     pareto_steiner_fast_analytical_preserve_tips, G, n_runs=10
# )
# print(f"Fixed-spacing: {mean_1:.4f} +/- {std_1:.4f} s")
# print(f"Analytical:    {mean_2:.4f} +/- {std_2:.4f} s")
# print(f"Speedup: {mean_1/mean_2:.2f}x")


# Legacy (jazzsta_exploration.py): Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017.
def random_tree(G,n=1000):
    """Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017.
    Only consider the critical nodes (and root node) of G.

    Args:
        G (nx.Graph): The graph to compute the random trees for
    Returns:
        costs (list): A list of (mactual, sactual) tuples for each random tree which
            represent the wiring cost and conduction delay of the tree.
    """
    random.seed(a=None)
    random_trees = []
    costs = []

    for i in range(n):
        R = nx.Graph()
        G_critical_nodes = get_critical_nodes(G)

        while len(G_critical_nodes) > 0:
            index = random.randrange(len(G_critical_nodes))
            g = G_critical_nodes[index]

            if len(R.nodes) > 0:
                r_index = random.randrange(len(R.nodes))
                r = list(R.nodes)[r_index]
                R.add_node(g, pos=G.nodes.data()[g]["pos"])
                R.add_edge(r, g, weight=node_dist(R, r, g))
            else:
                R.add_node(g, pos=G.nodes.data()[g]["pos"])

            del G_critical_nodes[index]

        random_trees.append(R)

    for R in random_trees:
        mactual, sactual = graph_costs(R)
        costs.append((mactual, sactual))

    return costs, random_trees


# Legacy (jazzsta_exploration.py): Calculate the maximum root order from graph attributes.
def get_max_root_order(G):
    """Calculate the maximum root order from graph attributes.
    Ariadne convention: primary = 0, first laterals = 1, etc.
    """
    max_order = 0
    for node, attrs in G.nodes(data=True):
        order = attrs.get('root_deg', 0)
        if order > max_order:
            max_order = order
    return max_order


# Legacy (jazzsta_exploration.py): Take an input graph G and add N-2 internal nodes placed within the
def place_internal_nodes_only(G_input, min_distance=1.0,
                              verbose=True):
    """
    Take an input graph G and add N-2 internal nodes placed within the
    expanded bounding box of the critical nodes, with bias toward the barycenter.
    Places internal nodes at half the Y-distance from base to tip barycenter.
    Uses halved standard deviation on X axis for tighter clustering.
    """
    from ariadne_roots.pareto_functions import get_critical_nodes

    critical_nodes = get_critical_nodes(G_input)
    base_node = 0

    positions = {}
    for node in critical_nodes:
        pos = G_input.nodes[node]['pos']
        positions[node] = np.array([pos[0], pos[1]]) if len(pos) == 3 else np.array(pos)

    # Assert base is at (0, 0)
    base_pos = positions[base_node]
    assert base_pos[0] == 0 and base_pos[1] == 0, \
        f"Base node must be at (0, 0), but found at ({base_pos[0]}, {base_pos[1]})"

    critical_nodes_sorted = sorted(critical_nodes, key=lambda n: G_input.nodes[n]['pos'][1])
    assert critical_nodes[0]==critical_nodes_sorted[0], f"Base node must be naming convention error"

    N = len(critical_nodes)
    n_internal = N - 2

    if verbose:
        print(f"Input: {N} critical nodes (1 base + {N - 1} tips)")
        print(f"Need to place: {n_internal} internal nodes")
        print(f"Base node: {base_node} at {positions[base_node]}")

    all_points = np.array(list(positions.values()))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)

    # Exclude primary tip from barycenter calculation
    tip_nodes = [n for n in critical_nodes if n != base_node]
    primary_tip = max(tip_nodes, key=lambda n: positions[n][1])
    tip_points = np.array([positions[n] for n in critical_nodes if n != base_node])
    hull = ConvexHull(tip_points)

    # Compute the centroid of the hull vertices
    hull_centroid = np.mean(hull.points[hull.vertices], axis=0)

    # Shrink the hull to 75% of its area by moving vertices toward the centroid
    # Shrinking by factor sqrt(0.75) ≈ 0.866 for area scaling
    shrink_factor = np.sqrt(0.75)  # ≈ 0.866
    shrunk_vertices = hull_centroid + shrink_factor * (hull.points[hull.vertices] - hull_centroid)

    # Build the shrunk hull while keeping base root area connected.
    shrunk_hull = ConvexHull(shrunk_vertices)
    shrunk_with_base = np.vstack([
        shrunk_hull.points[shrunk_hull.vertices],
        base_pos.reshape(1, 2)
    ])
    shrunk_hull = ConvexHull(shrunk_with_base)

    # Compute barycenter of tips (excluding primary tip)
    points_without_primary = np.array([positions[n] for n in critical_nodes
                                       if n != primary_tip])
    tip_barycenter = np.mean(points_without_primary, axis=0)

    # Place internal nodes at HALF the distance from base to tip barycenter
    base_pos = positions[base_node]
    barycenter = base_pos + (tip_barycenter - base_pos) / 2

    if verbose:
        print(f"Tip barycenter (excl. primary): [{tip_barycenter[0]:.2f}, {tip_barycenter[1]:.2f}]")
        print(f"Internal node barycenter (α/2): [{barycenter[0]:.2f}, {barycenter[1]:.2f}]")

    G_new = nx.Graph()
    G_new.add_node(critical_nodes_sorted[0], pos=positions[critical_nodes_sorted[0]])
    G_new.nodes[0]['is_base'] = (0 == base_node)
    G_new.nodes[0]['is_tip'] = (0 != base_node)
    next_id = max(critical_nodes_sorted) + 1

    for node in critical_nodes_sorted[1:-1]:
        G_new.add_node(node, pos=positions[node])
        G_new.nodes[node]['is_base'] = (node == base_node)
        G_new.nodes[node]['is_tip'] = (node != base_node)
        std_x = (max_coords[0] - min_coords[0]) / 12
        std_y = abs(G_new.nodes[node]['pos'][1]) / 12
        candidate = np.array([
            np.random.normal(barycenter[0], std_x),
            np.random.normal(G_new.nodes[node]['pos'][1]/ 2, std_y)
        ])

        # Constrain_to_shrunk_hull if outside (checking using winding number)
        path = Path(shrunk_hull.points[shrunk_hull.vertices])
        if not path.contains_points([candidate])[0]:
            # Outside — find closest point on hull boundary
            min_dist = float('inf')
            best_point = None
            for simplex in shrunk_hull.simplices:
                for i in range(len(simplex)):
                    p1 = shrunk_hull.points[simplex[i]]
                    p2 = shrunk_hull.points[simplex[(i + 1) % len(simplex)]]
                    vec = p2 - p1
                    t = np.dot(candidate - p1, vec) / np.dot(vec, vec)
                    t = np.clip(t, 0, 1)
                    proj_point = p1 + t * vec
                    dist = np.linalg.norm(candidate - proj_point)
                    if dist < min_dist:
                        min_dist = dist
                        best_point = proj_point
            candidate = best_point
        print(f"  Tip {node} (Y={G_new.nodes[node]['pos'][1]:.0f}) -> internal {next_id - 1} (Y={candidate[1]:.0f})")

        too_close = False
        if np.linalg.norm(candidate - G_new.nodes[node]['pos']) < min_distance:
            too_close = True
            max_bounces = 20
            bounces = 0

            while too_close and bounces < max_bounces:
                too_close = False
                for existing_node in G_new.nodes():
                    existing_pos = G_new.nodes[existing_node ]['pos']
                    dist = np.linalg.norm(candidate - existing_pos)
                    if dist < min_distance:
                        # Bounce away: push candidate in random direction by 2*min_distance
                        if dist < 1e-6:
                            # Coincident points - push in purely random direction
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = np.array([np.cos(angle), np.sin(angle)]) * 2 * min_distance
                        else:
                            # Push away from the existing node along the line connecting them
                            direction = (candidate - existing_pos) / dist
                            push = direction * 2 * min_distance

                        candidate = candidate + push
                        candidate = np.clip(candidate, min_coords, max_coords)
                        too_close = True
                        bounces += 1
                        break  # Recheck from the beginning after each bounce

        G_new.add_node(next_id, pos=candidate, is_internal=True)
        next_id += 1

    if verbose:
        print(f"\nResult: Graph with {G_new.number_of_nodes()} nodes")

    return G_new


# Legacy (jazzsta_exploration.py): Take an input graph G and add N-2 internal nodes placed within the
def place_internal_nodes_only_tips(G_input, min_distance=1.0, expansion=1.0,
                              downward_bias=True, barycenter_weight=0.7,
                              verbose=True):
    """
    Take an input graph G and add N-2 internal nodes placed within the
    expanded bounding box of the critical nodes, with bias toward the barycenter.
    """
    from ariadne_roots.pareto_functions import get_critical_nodes

    critical_nodes = get_critical_nodes(G_input)
    base_node = 0

    positions = {}
    for node in critical_nodes:
        pos = G_input.nodes[node]['pos']
        positions[node] = np.array([pos[0], pos[1]]) if len(pos) == 3 else np.array(pos)

    N = len(critical_nodes)
    n_internal = N - 2

    if verbose:
        print(f"Input: {N} critical nodes (1 base + {N - 1} tips)")
        print(f"Need to place: {n_internal} internal nodes")
        print(f"Base node: {base_node} at {positions[base_node]}")

    all_points = np.array(list(positions.values()))
    min_coords = np.min(all_points, axis=0) - expansion
    max_coords = np.max(all_points, axis=0) + expansion
    barycenter = np.mean(all_points, axis=0)

    if verbose:
        print(f"Bounding box: x=[{min_coords[0]:.2f}, {max_coords[0]:.2f}], "
              f"y=[{min_coords[1]:.2f}, {max_coords[1]:.2f}]")
        print(f"Barycenter: [{barycenter[0]:.2f}, {barycenter[1]:.2f}]")

    G_new = nx.Graph()
    for node in critical_nodes:
        G_new.add_node(node, pos=positions[node])
        G_new.nodes[node]['is_base'] = (node == base_node)
        G_new.nodes[node]['is_tip'] = (node != base_node)

    next_id = max(critical_nodes) + 1
    internal_count = 0
    max_attempts = n_internal * 1000
    attempts = 0
    base_pos = positions[base_node]

    while internal_count < n_internal and attempts < max_attempts:
        if np.random.random() < barycenter_weight:
            std_x = (max_coords[0] - min_coords[0]) / 6
            std_y = (max_coords[1] - min_coords[1]) / 6
            candidate = np.array([
                np.random.normal(barycenter[0], std_x),
                np.random.normal(barycenter[1], std_y)
            ])
            candidate = np.clip(candidate, min_coords, max_coords)
        else:
            candidate = np.array([
                np.random.uniform(min_coords[0], max_coords[0]),
                np.random.uniform(min_coords[1], max_coords[1])
            ])

        if downward_bias:
            if candidate[1] < base_pos[1]:
                if np.random.random() < 0.8:
                    attempts += 1
                    continue

        too_close = False
        for node in G_new.nodes():
            if np.linalg.norm(candidate - G_new.nodes[node]['pos']) < min_distance:
                too_close = True
                break

        if too_close:
            attempts += 1
            continue

        G_new.add_node(next_id, pos=candidate, is_internal=True)
        internal_count += 1
        next_id += 1
        attempts = 0

    if verbose:
        print(f"\nResult: Graph with {G_new.number_of_nodes()} nodes")
        print(f"  Internal nodes: {internal_count}")
        if internal_count < n_internal:
            print(f"  WARNING: Only placed {internal_count} of {n_internal} internal nodes")

    return G_new


# Legacy (jazzsta_exploration.py): Add ONE axis to the graph G.
def add_axis(G, max_order, randomness=0.3, verbose=False):
    """
    Add ONE axis to the graph G.
    Ariadne convention: primary = 0, first laterals = 1, etc.
    """
    base_node = 0

    # Determine parent
    if G.degree(base_node) == 0:
        parent = base_node
        parent_order = 0
        new_order = 0
        is_primary = True
    else:
        is_primary = False
        parent = None
        visited = set()
        queue = [base_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node != base_node and G.nodes[node].get('is_internal', False):
                if G.degree(node) < 3:
                    parent = node
                    break

            for neighbor in G.neighbors(node):
                if neighbor not in visited and 'root_deg' in G.nodes[neighbor]:
                    queue.append(neighbor)

        if parent is None:
            if verbose:
                print("    No unbranched internal node found")
            return False

        parent_order = G.nodes[parent]['root_deg']
        children = list(G.neighbors(parent))
        child_orders = [G.nodes[c].get('root_deg', 0) for c in children]
        new_order = parent_order if parent_order not in child_orders else parent_order + 1

    if new_order > max_order:
        if verbose:
            print(f"    Cannot add axis: new_order {new_order} > max_order {max_order}")
        return False

    parent_pos = G.nodes[parent]['pos']

    available_tips = [n for n in G.nodes()
                      if G.nodes[n].get('is_tip', False) and G.degree(n) == 0]

    if not available_tips:
        if verbose:
            print("    No available tips")
        return False

    # Select tip
    if is_primary:
        chosen_tip = max(available_tips, key=lambda t: G.nodes[t]['pos'][1])
    else:
        scored_tips = []
        for tip in available_tips:
            tip_pos = G.nodes[tip]['pos']
            tip_y = tip_pos[1]
            dist = np.linalg.norm(tip_pos - parent_pos)
            growth = tip_y - parent_pos[1]
            score = dist / (1 + growth * 0.01) if growth > 0 else dist * (1 - growth * 0.1)
            scored_tips.append((score, tip))

        scored_tips.sort(key=lambda x: x[0])
        k = min(5, len(scored_tips))
        idx = np.random.randint(k) if np.random.random() < randomness else 0
        chosen_tip = scored_tips[idx][1]

    tip_pos = G.nodes[chosen_tip]['pos']

    available_internals = [n for n in G.nodes()
                           if G.nodes[n].get('is_internal', False) and G.degree(n) == 0]

    if verbose:
        print(
            f"    Parent {parent} (order={parent_order}) at Y={parent_pos[1]:.0f}, tip {chosen_tip} at Y={tip_pos[1]:.0f}")
        print(f"    Available internals: {len(available_internals)}")

    if not available_internals:
        G.add_edge(parent, chosen_tip)
        G.nodes[chosen_tip]['root_deg'] = new_order
        return True

    # Filter and score internals: must be between parent and tip in Y, prefer those
    # that keep the path oriented downward (monotonically increasing Y)
    axis_vec = tip_pos - parent_pos
    axis_len = np.linalg.norm(axis_vec)

    if axis_len > 1e-6:
        axis_dir = axis_vec / axis_len

        # Score internals by how well they keep the path straight downward
        scored_internals = []
        for node in available_internals:
            node_pos = G.nodes[node]['pos']
            node_y = node_pos[1]

            # Must be between parent and tip in Y (with margin)
            min_y = min(parent_pos[1], tip_pos[1])
            max_y = max(parent_pos[1], tip_pos[1])
            if not (min_y - 50 < node_y < max_y + 50):
                continue

            # Project onto axis
            vec = node_pos - parent_pos
            proj = np.dot(vec, axis_dir)

            # Distance from the direct line
            if proj <= 0:
                proj_point = parent_pos
            elif proj >= axis_len:
                proj_point = tip_pos
            else:
                proj_point = parent_pos + proj * axis_dir

            dist_to_axis = np.linalg.norm(node_pos - proj_point)

            # Score: prefer nodes close to the direct line, well-spaced
            if 0.1 * axis_len < proj < 0.9 * axis_len:
                score = dist_to_axis
            else:
                score = dist_to_axis + axis_len * 0.3

            scored_internals.append((score, proj, node))

        # If we have good candidates, use them
        if scored_internals:
            scored_internals.sort(key=lambda x: x[1])  # Sort by projection

            # Take a subset: at most half, evenly spaced
            n_available = len(scored_internals)
            n_take = max(1, min(n_available, n_available // 2))

            if n_take == 1:
                indices = [n_available // 2]
            else:
                indices = np.linspace(0, n_available - 1, n_take, dtype=int)

            valid_internals = [scored_internals[i][2] for i in indices]
        else:
            # Fallback: Y-based selection
            valid_internals = []
            min_y = min(parent_pos[1], tip_pos[1])
            max_y = max(parent_pos[1], tip_pos[1])
            for node in available_internals:
                node_y = G.nodes[node]['pos'][1]
                if min_y - 50 < node_y < max_y + 50:
                    valid_internals.append(node)
            if not valid_internals:
                valid_internals = available_internals
    else:
        valid_internals = available_internals

    # Sort by Y for proper chain ordering
    valid_internals.sort(key=lambda n: G.nodes[n]['pos'][1])

    if verbose:
        print(f"    Using {len(valid_internals)} internal nodes")

    # Build chain
    used = []
    prev = parent

    for node in valid_internals:
        G.add_edge(prev, node)
        G.nodes[node]['root_deg'] = new_order
        used.append(node)
        prev = node

    G.add_edge(prev, chosen_tip)
    G.nodes[chosen_tip]['root_deg'] = new_order

    if verbose:
        print(f"    Added axis with {len(used)} internal nodes, new_order={new_order}")

    return True


# Legacy (jazzsta_exploration.py): Add edges for ONE parent node.
def add_one_branch(G, max_order, randomness=0.3, verbose=False):
    """
    Add edges for ONE parent node.
    - If base has no child: connect base to one internal (order 0)
    - If parent has 1 child: add one lateral child (order + 1)
    - If parent has 0 children: add continuation child (same order)
    Only adds ONE edge per call to keep it simple.
    """
    base_node = 0

    # Ensure base has root_deg
    if 'root_deg' not in G.nodes[base_node]:
        G.nodes[base_node]['root_deg'] = 0
        G.nodes[base_node]['is_base'] = True

    # Case 1: Base has no child — connect to first internal
    if G.degree(base_node) == 0:
        available = [n for n in G.nodes()
                     if G.nodes[n].get('is_internal', False) and G.degree(n) == 0]
        if not available:
            return False
        child = min(available,
                    key=lambda n: np.linalg.norm(G.nodes[n]['pos'] - G.nodes[base_node]['pos']))
        G.add_edge(base_node, child)
        G.nodes[child]['root_deg'] = 0
        if verbose:
            print(f"    Base -> {child} (order 0)")
        return True

    # Case 2: Find an internal node that needs children
    parents = []
    for n in G.nodes():
        if n == base_node:
            continue
        if not G.nodes[n].get('is_internal', False):
            continue
        if 'root_deg' not in G.nodes[n]:
            continue
        if G.degree(n) >= 3:
            continue
        parents.append((G.nodes[n]['pos'][1], n))  # (Y, node)

    if not parents:
        return False

    # Pick the parent with lowest Y (closest to base)
    parents.sort(key=lambda x: x[0])
    _, parent = parents[0]

    parent_order = G.nodes[parent]['root_deg']
    parent_pos = G.nodes[parent]['pos']

    # Count current children (neighbors that aren't the parent)
    children = [n for n in G.neighbors(parent) if not G.nodes[n].get('is_base', False)]

    # Determine what order the next child should have
    if len(children) == 0:
        child_order = parent_order  # First child: continuation
    else:
        child_order = parent_order + 1  # Second child: lateral

    # If lateral would exceed max_order, use continuation instead
    if child_order > max_order:
        child_order = parent_order

    # Gather candidates: tips first, then internals
    candidates = []

    for t in G.nodes():
        if not G.nodes[t].get('is_tip', False):
            continue
        if G.degree(t) > 0:
            continue
        t_pos = G.nodes[t]['pos']
        dist = np.linalg.norm(t_pos - parent_pos)
        if t_pos[1] < parent_pos[1]:
            dist *= 0.5  # Priority for missed tips
        candidates.append((dist, t, 'tip'))

    for i in G.nodes():
        if not G.nodes[i].get('is_internal', False):
            continue
        if G.degree(i) > 0:
            continue
        i_pos = G.nodes[i]['pos']
        dist = np.linalg.norm(i_pos - parent_pos)
        candidates.append((dist, i, 'internal'))

    if not candidates:
        return False

    candidates.sort(key=lambda x: x[0])

    # Pick with randomness
    k = min(5, len(candidates))
    idx = np.random.randint(k) if np.random.random() < randomness else 0
    _, child, typ = candidates[idx]

    G.add_edge(parent, child)
    G.nodes[child]['root_deg'] = child_order

    if verbose:
        print(f"    {parent} (order={parent_order}) -> {child} ({typ}, order={child_order})")

    return True


# Legacy (jazzsta_exploration.py): Generate multiple random trees from a single input graph.
def generate_random_trees_from_input(G_input, n_trees=1000,
                                     min_distance=1.0, expansion=1.0,
                                     barycenter_weight=0.7, randomness=0.3,
                                     max_order_increase=0.2, verbose=True):
    """
    Generate multiple random trees from a single input graph.
    """
    from ariadne_roots.pareto_functions import graph_costs, get_critical_nodes, point_dist

    max_order_input = get_max_root_order(G_input)
    max_order = max(2, int(max_order_input * (1 + max_order_increase)))

    if verbose:
        print(f"Input max root_deg: {max_order_input}")
        print(f"Allowed max root_deg: {max_order}")
        print(f"Generating {n_trees} trees...")

    random_trees = []
    costs = []
    trees_generated = 0
    attempts = 0
    max_attempts = n_trees * 20

    while trees_generated < n_trees and attempts < max_attempts:
        attempts += 1

        H = place_internal_nodes_only(G_input, min_distance=min_distance, verbose=False)

        if H is None:
            continue

        try:
            while True:
                success = add_axis(H, max_order, randomness=randomness, verbose=False)
                if not success:
                    break

            for u, v in H.edges():
                H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])

            tips_total = [n for n in H.nodes() if H.nodes[n].get('is_tip', False)]
            tips_connected = sum(1 for n in tips_total if H.degree(n) > 0)

            if verbose and attempts <= 2:
                print(f"    Attempt {attempts} debug:")
                print(f"      tips_connected={tips_connected}/{len(tips_total)}")
                is_conn = nx.is_connected(H)
                is_tree = nx.is_tree(H)
                print(f"      is_connected={is_conn}")
                print(f"      is_tree={is_tree}")
                print(f"      nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")
                try:
                    m, s = graph_costs(H)
                    print(f"      costs=({m:.0f}, {s:.0f})")
                except Exception as e:
                    print(f"      graph_costs error: {e}")
                if not is_tree:
                    try:
                        cycle = nx.find_cycle(H)
                        print(f"      cycle found: {cycle[:3]}...")
                    except:
                        print(f"      no cycle found but not a tree")
                if not is_conn:
                    comps = list(nx.connected_components(H))
                    print(f"      components: {len(comps)}")

            if tips_connected == len(tips_total):
                mactual, sactual = graph_costs(H)
                costs.append((mactual, sactual))
                random_trees.append(H)
                trees_generated += 1
                if verbose and trees_generated % 10 == 0:
                    print(f"  Generated {trees_generated} trees")

        except Exception as e:
            if verbose and attempts % 50 == 0:
                print(f"  Error: {e}")
            continue

    if verbose:
        print(f"\nGenerated {len(random_trees)} trees after {attempts} attempts")

    return costs, random_trees


# Legacy (jazzsta_exploration.py): Generate multiple random trees from a single input graph.
def generate_random_trees_from_input_tips(G_input, n_trees=1000,
                                     min_distance=1.0, expansion=1.0,
                                     barycenter_weight=0.7, randomness=0.3,
                                     max_order_increase=0.2, verbose=True):
    """
    Generate multiple random trees from a single input graph.
    """
    from ariadne_roots.pareto_functions import graph_costs, get_critical_nodes, point_dist

    max_order_input = get_max_root_order(G_input)
    # Ariadne convention: primary=0, laterals=1,2,...
    # With max_order_input=1 (most plants), max_order becomes max(2, 1)=2
    # This allows orders 0 (primary), 1 (first laterals), 2 (second laterals)
    max_order = max(2, int(max_order_input * (1 + max_order_increase)))

    if verbose:
        print(f"Input max root_deg: {max_order_input}")
        print(f"Allowed max root_deg: {max_order}")
        print(f"Generating {n_trees} trees...")

    random_trees = []
    costs = []
    trees_generated = 0
    attempts = 0
    max_attempts = n_trees * 20

    while trees_generated < n_trees and attempts < max_attempts:
        attempts += 1

        H = place_internal_nodes_only_tips(G_input, min_distance=min_distance,
                                      expansion=expansion, downward_bias=True,
                                      barycenter_weight=barycenter_weight, verbose=False)

        if H is None:
            continue

        try:
            while True:
                success = add_axis(H, max_order, randomness=randomness, verbose=False)
                if not success:
                    break

            for u, v in H.edges():
                H[u][v]['weight'] = point_dist(H.nodes[u]['pos'], H.nodes[v]['pos'])

            tips_total = [n for n in H.nodes() if H.nodes[n].get('is_tip', False)]
            tips_connected = sum(1 for n in tips_total if H.degree(n) > 0)

            if tips_connected == len(tips_total):
                mactual, sactual = graph_costs(H)
                costs.append((mactual, sactual))
                random_trees.append(H)
                trees_generated += 1
                if verbose and trees_generated % 10 == 0:
                    print(f"  Generated {trees_generated} trees")

        except Exception as e:
            if verbose and attempts % 50 == 0:
                print(f"  Error: {e}")
            continue

    if verbose:
        print(f"\nGenerated {len(random_trees)} trees after {attempts} attempts")

    return costs, random_trees


# Legacy (jazzsta_exploration.py): Perform Pareto-related calculations with all three random generators.
def pareto_calcs(H, n_random=100):
    """Perform Pareto-related calculations with all three random generators."""
    front, actual = pareto_front(H)
    mactual, sactual = actual

    tradeoff_info = calculate_tradeoff(front, actual)
    plant_alpha, plant_scaling = distance_from_front(front, actual)

    print("Generating original random trees...")
    randoms_original = random_tree(H, n_random)[0]
    print(f"  Got {len(randoms_original)} original random trees")

    print("Generating internal barycenter random trees...")
    randoms_geometric = generate_random_trees_from_input(H, n_trees=n_random, verbose=False)[0]
    print(f"  Got {len(randoms_geometric)} internal barycenter random trees")

    print("Generating tip barycenter random trees...")
    randoms_geotips = generate_random_trees_from_input_tips(H, n_trees=n_random, verbose=False)[0]
    print(f"  Got {len(randoms_geotips)} tip barycenter random trees")

    if len(randoms_original) > 0:
        mrand_original = np.mean([x[0] for x in randoms_original])
        srand_original = np.mean([x[1] for x in randoms_original])
    else:
        mrand_original, srand_original = 0.0, 0.0

    if len(randoms_geometric) > 0:
        mrand_geometric = np.mean([x[0] for x in randoms_geometric])
        srand_geometric = np.mean([x[1] for x in randoms_geometric])
    else:
        mrand_geometric, srand_geometric = 0.0, 0.0

    if len(randoms_geotips) > 0:
        mrand_geotips = np.mean([x[0] for x in randoms_geotips])
        srand_geotips = np.mean([x[1] for x in randoms_geotips])
    else:
        mrand_geotips, srand_geotips = 0.0, 0.0

    if len(randoms_original) > 0:
        rand_alpha, rand_scaling = distance_from_front(front, (mrand_original, srand_original))
    else:
        rand_alpha, rand_scaling = 0.0, 0.0

    results = {
        "Total root length": mactual,
        "Travel distance": sactual,
        "alpha": plant_alpha,
        "scaling distance to front": plant_scaling,
        "Total root length (random)": mrand_original,
        "Travel distance (random)": srand_original,
        "Total root length (internal barycenter)": mrand_geometric,
        "Travel distance (internal barycenter)": srand_geometric,
        "Total root length (tip barycenter)": mrand_geotips,
        "Travel distance (tip barycenter)": srand_geotips,
        "alpha (random)": rand_alpha,
        "scaling (random)": rand_scaling,
    }

    results.update(tradeoff_info)
    return results, front, randoms_original, randoms_geometric, randoms_geotips


# Legacy (jazzsta_exploration.py): Report basic root metrics for a given graph.
def analyze(G, enable_3d=True, n_random=100):
    """Report basic root metrics for a given graph."""
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

    results, front, randoms_original, randoms_geometric, randoms_geotips = pareto_calcs(H, n_random)

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

    return results, front, randoms_original, randoms_geometric, randoms_geotips, \
           results_3d, front_3d, randoms_3d


# Legacy (jazzsta_exploration.py): Plot Pareto front with all three random tree populations.
def plot_all_three(front, actual, randoms_original, randoms_geometric, randoms_geotips,
                   mrand_original, srand_original,
                   mrand_geometric, srand_geometric,
                   mrand_geotips, srand_geotips, dest=None):
    """Plot Pareto front with all three random tree populations."""
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

    scaled_front_x = [scale_data(x[0]) for x in front.values()]
    scaled_front_y = [scale_data(x[1]) for x in front.values()]
    scaled_actual_x = scale_data(actual[0])
    scaled_actual_y = scale_data(actual[1])

    scaled_original_x = [scale_data(x[0]) for x in randoms_original]
    scaled_original_y = [scale_data(x[1]) for x in randoms_original]
    scaled_geometric_x = [scale_data(x[0]) for x in randoms_geometric]
    scaled_geometric_y = [scale_data(x[1]) for x in randoms_geometric]
    scaled_geotips_x = [scale_data(x[0]) for x in randoms_geotips]
    scaled_geotips_y = [scale_data(x[1]) for x in randoms_geotips]

    scaled_mrand_original = scale_data(mrand_original)
    scaled_srand_original = scale_data(srand_original)
    scaled_mrand_geometric = scale_data(mrand_geometric)
    scaled_srand_geometric = scale_data(srand_geometric)
    scaled_mrand_geotips = scale_data(mrand_geotips)
    scaled_srand_geotips = scale_data(srand_geotips)

    for i, (x, y) in enumerate(zip(scaled_original_x, scaled_original_y)):
        plt.plot(x, y, marker="+", color="green", markersize=2.5, zorder=0.5,
                 markeredgewidth=0.5, alpha=0.5,
                 label='Original Random' if i == 0 else '')

    for i, (x, y) in enumerate(zip(scaled_geometric_x, scaled_geometric_y)):
        plt.plot(x, y, marker="o", color="blue", markersize=3.0, zorder=0.6,
                 markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                 label='Internal Barycenter' if i == 0 else '')

    for i, (x, y) in enumerate(zip(scaled_geotips_x, scaled_geotips_y)):
        plt.plot(x, y, marker="s", color="purple", markersize=3.0, zorder=0.7,
                 markeredgewidth=0.5, alpha=0.5, fillstyle='none',
                 label='Tip Barycenter' if i == 0 else '')

    plt.plot(scaled_front_x, scaled_front_y, marker="s", linestyle="-",
             markeredgecolor="black", color="orange", label='Pareto Front', zorder=2)
    plt.plot(scaled_actual_x, scaled_actual_y, marker="x", markersize=14,
             zorder=4, markeredgewidth=2.0, color="black", label='Actual Tree')
    plt.plot(scaled_mrand_original, scaled_srand_original, marker="+", color="red",
             markersize=14, zorder=3, markeredgewidth=2.0, label='Original Mean')
    plt.plot(scaled_mrand_geometric, scaled_srand_geometric, marker="o", color="cyan",
             markersize=12, zorder=3, markeredgewidth=2.0, label='Internal Bary. Mean')
    plt.plot(scaled_mrand_geotips, scaled_srand_geotips, marker="s", color="magenta",
             markersize=12, zorder=3, markeredgewidth=2.0, label='Tip Bary. Mean')

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=15)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=15)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_title("Pareto Front: Comparison of Random Tree Generators", fontsize=16)
    ax.grid(True, alpha=0.3)

    all_x = scaled_front_x + scaled_original_x + scaled_geometric_x + scaled_geotips_x + \
            [scaled_actual_x, scaled_mrand_original, scaled_mrand_geometric, scaled_mrand_geotips]
    all_y = scaled_front_y + scaled_original_y + scaled_geometric_y + scaled_geotips_y + \
            [scaled_actual_y, scaled_srand_original, scaled_srand_geometric, scaled_srand_geotips]

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


# Legacy (jazzsta_exploration.py): Calculate axis limits with buffer.
def calculate_plot_buffer(min_val, max_val, buffer_percent=0.2):
    """Calculate axis limits with buffer."""
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    buffer = range_val * buffer_percent
    return min_val - buffer, max_val + buffer


# Legacy (jazzsta_exploration.py): comparaplot
def comparaplot(G, n_random=500):
    results, front, randoms_original, randoms_geometric, randoms_geotips, \
        results_3d, front_3d, randoms_3d = analyze(G, enable_3d=False, n_random=n_random)

    plot_all_three(
        front,
        [results["Total root length"], results["Travel distance"]],
        randoms_original,
        randoms_geometric,
        randoms_geotips,
        results["Total root length (random)"],
        results["Travel distance (random)"],
        results["Total root length (internal barycenter)"],
        results["Travel distance (internal barycenter)"],
        results["Total root length (tip barycenter)"],
        results["Travel distance (tip barycenter)"]
    )

"""
G=get_graph_from_json("C:\\Users\\delivori\\Documents\\jsonArchis\\_set1_day1_20240429-125122_010_plantB_day1.json")


"""


# Legacy (oldfrontexperimentations.py): Given a graph G and a value 0 <= alpha <= 1, compute the Pareto-optimal tree connecting
def pareto_steiner_fast_fixed_spacing(G, alpha, min_length_factor=1.0,spacing_factor=1.0):
    """Given a graph G and a value 0 <= alpha <= 1, compute the Pareto-optimal tree connecting
    the root base_node to all of the lateral root tips of G

    The algorithm attempts to optimize alpha * D + (1 - alpha) * W

    D is the conduction delay: the sum of the lengths of the shortest paths from every
    lateral root tip to the root base_node of the network

    W is the wiring cost: the total length of the tree

    The algorithm uses a greedy approach: always take the edge that will reduce the
    pareto cost of the tree by the smallest amount

    When alpha = 0, the cost is (1 - alpha) * W = W, so the algorithm will try to minimize
    the wiring cost

    When alpha = 1, the cost is alpha * D = D, so the algorithm will try to minimize the
    conduction delay.
    """
    assert 0 <= alpha <= 1

    # assume the base_node is node 0
    base_node = 0

    H = nx.Graph()

    H.add_node(base_node)
    # every node will keep track of its distance to the base_node
    H.nodes[base_node]["distance_to_base"] = 0
    base_pos = G.nodes[base_node]["pos"]
    H.nodes[base_node]["pos"] = base_pos
    added_nodes = 1

    critical_nodes = get_critical_nodes(G)

    # critical nodes that have currently been added to the tree
    in_nodes = set([base_node])

    # critical nodes that have not yet been added to the tree
    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    graph_total_root_length = 0
    graph_total_travel_distance = 0

    """
    closest_neighbors is a dictionary. The keys are nodes that are currently in the tree.
    The value associated with each node u is list of nodes that need to  be added to the
    tree, sorted in order of how close each node is to u.
    """
    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    """
    unpaired_nodes contains the set of nodes for which we need to (re)-compute the closest
    node that has not been added to the tree. Gonna make it really really better
    """
    unpaired_nodes = set([base_node])

    # keeps track of what the id of the next node added to the tree should be.
    node_index = max(critical_nodes) + 1

    steps = 0

    """
    The optimization this algorithm performs is that we don't need to consider every
    combination (u, v) where u is in the tree and v is not in the tree. We only need to
    consider combinations where u is in the tree, and v is the closest node to u that has
    not been added to the tree. This is because any other combination would never be the
    optimal greedy choice.

    best_edges stores all of these potentially optimal edges to add
    """
    best_edges = []

    while added_nodes < len(critical_nodes):
        assert len(out_nodes) > 0
        best_edge = None
        best_total_root_length = None
        best_total_travel_distance = None
        best_cost = float("inf")

        best_choice = None
        best_midpoint = None

        # go through nodes for which we need to (re)-compute its closest neighbor outside the tree
        for u in unpaired_nodes:
            assert H.has_node(u)
            assert "distance_to_base" in H.nodes[u]

            invalid_neighbors = []
            closest_neighbor = None
            """
            Go through u's closest neighbors and find the closest one that has not been
            added to the tree
            """
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                else:
                    # once we find the closest neighbor not in the tree,
                    closest_neighbor = v
                    break

            """
            if any of the closest neighbors are no longer valid partners, remove them from
            the list of closest neighbors
            """
            for invalid_neighbor in invalid_neighbors:
                closest_neighbors[u].remove(invalid_neighbor)

            assert closest_neighbor != None
            assert not H.has_node(closest_neighbor)

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[closest_neighbor]["pos"]

            # compute hypothetical cost of connecting u to its closest neighbor
            edge_length = point_dist(p1, p2)
            total_root_length = edge_length
            total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]
            cost = pareto_cost(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                alpha=alpha,
            )

            # add this candidate edge to the list of best edges
            # insort maintains the list in sorted order based on cost
            insort(best_edges, (cost, u, closest_neighbor))

        # We will add the candidate edge with the smallest cost
        # because we maintained these edges in sorted order, it's O(1) to get the next edge to add
        cost, u, v = best_edges.pop(0)

        # go through all the candidate edges and see which ones are no longer valid
        best_edges2 = []
        # u and v are now unpaired, we need to find their respective closest neighbors outside of the tree
        unpaired_nodes = set([u, v])
        for cost, x, y in best_edges:
            # if another node had v as its closet neighbor, we need to find its next-closest neighbor going forward
            if y == v:
                unpaired_nodes.add(x)
            else:
                best_edges2.append((cost, x, y))
        best_edges = best_edges2

        assert H.has_node(u)
        assert not H.has_node(v)
        H.add_node(v)
        H.nodes[v]["pos"] = G.nodes[v]["pos"]
        # v is now in the tree, not outside the tree
        in_nodes.add(v)
        out_nodes.remove(v)

        # connect u to v, adding several midpoints along the way
        p1 = H.nodes[u]["pos"]
        p2 = H.nodes[v]["pos"]
        spacing = get_min_lateral_length(G, spacing_factor)
        midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
        midpoint_nodes = []

        # add a new node for every midpoint being added along the u-v line segment
        for midpoint in midpoints:
            midpoint_node = node_index
            node_index += 1
            H.add_node(midpoint_node)
            H.nodes[midpoint_node]["pos"] = midpoint

            # get the distance from the midpoint  to all nodes that need to be added to t he tree
            neighbors = []
            for out_node in out_nodes:
                out_coord = G.nodes[out_node]["pos"]
                dist = point_dist(midpoint, out_coord)
                neighbors.append((dist, out_node))

            # add the newly-added midpoint node to closest_neighbors
            neighbors = sorted(neighbors)
            closest_neighbors[midpoint_node] = []
            for dist, neighbor in neighbors:
                closest_neighbors[midpoint_node].append(neighbor)

            midpoint_nodes.append(midpoint_node)

            # midpoint_node is unpaired, we need to add it to the candidate edges in the next iteration
            unpaired_nodes.add(midpoint_node)

        # connect all of the points in the line segment: u, v, and all the nodes in between
        line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
        for i in range(-1, -len(line_nodes), -1):
            n1 = line_nodes[i]
            n2 = line_nodes[i - 1]
            H.add_edge(n1, n2)
            H[n1][n2]["weight"] = node_dist(H, n1, n2)
            assert "distance_to_base" in H.nodes[n1]
            H.nodes[n2]["distance_to_base"] = (
                node_dist(H, n2, u) + H.nodes[u]["distance_to_base"]
            )

        added_nodes += 1
    return H


# Legacy (oldfrontexperimentations.py): Pareto-optimal Steiner tree that preserves original tips and enforces
def pareto_steiner_fast_preserve_tips(G, alpha, min_length_factor=1.0, spacing_factor=1.0):
    """
    Pareto-optimal Steiner tree that preserves original tips and enforces
    nude tip segments equal to minimum lateral length of G, and binary tree constraints.
    Deterministically prefers continuation (Option A) over lateral (Option B)
    when both are valid, because Option A can unlock future branching sites.

    Args:
        G: Original graph
        alpha: Pareto weight (0 = satellite, 1 = steiner)

    Returns:
        H: Steiner tree preserving tip constraints
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)


    # ---- Build Steiner tree ----
    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    in_nodes = {base_node}
    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        # Recompute best edges for all unpaired nodes that can still accept children
        for u in unpaired_nodes:
            if not H.has_node(u):
                continue
            # Binary tree constraint
            if u != base_node and H.degree(u) >= 3:
                continue
            if u == base_node and H.degree(u) >= 1:
                continue

            invalid_neighbors = []
            closest_neighbor = None
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                else:
                    closest_neighbor = v
                    break
            for invalid in invalid_neighbors:
                closest_neighbors[u].remove(invalid)

            if closest_neighbor is None:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[closest_neighbor]["pos"]
            edge_length = point_dist(p1, p2)
            total_root_length = edge_length
            total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]
            cost = pareto_cost(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                alpha=alpha,
            )
            insort(best_edges, (cost, u, closest_neighbor))

        if not best_edges:
            break

        # Try edges until one satisfies constraints
        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            # Binary tree constraint (again, in case state changed)
            if u != base_node and H.degree(u) >= 3:
                continue
            if u == base_node and H.degree(u) >= 1:
                continue

            # Check if u is a valid branching point (nude zone)
            u_has_child = len([n for n in H.neighbors(u) if n != base_node]) > 0
            if u != base_node and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0:
                    nude_zone = min_lateral_len
                    if H.nodes[u]["dist_to_axis_tip"] < nude_zone:
                        continue  # u is in the nude zone, cannot branch

            # ---- Evaluate options ----
            option_a_valid = False
            option_b_valid = False

            # Option A: Continuation (v extends u's axis)
            if u == base_node:
                option_a_valid = True
            else:
                old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                new_axis_total = old_axis_total + full_dist
                if new_axis_total > 0 and full_dist >= min_lateral_len:
                    # The old continuation becomes a lateral – check its length
                    if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                        option_a_valid = True

            # Option B: Lateral (v is a new lateral from u)
            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue  # Neither valid, try next candidate

            # Deterministic choice: prefer Option A (continuation)
            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            # Cleanup best_edges for next iteration
            best_edges2 = []
            unpaired_nodes = {u, v}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            # ---- Add node v to the tree ----
            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            in_nodes.add(v)
            out_nodes.remove(v)

            # Add Steiner midpoints
            midpoints = steiner_points(p1, p2, npoints=STEINER_MIDPOINTS)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint

                neighbors = []
                for out_node in out_nodes:
                    out_coord = G.nodes[out_node]["pos"]
                    dist = point_dist(midpoint, out_coord)
                    neighbors.append((dist, out_node))
                neighbors = sorted(neighbors)
                closest_neighbors[midpoint_node] = [n for _, n in neighbors]

                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)

            # Connect the chain: v -> midpoints -> u
            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            # ---- Apply axis labeling ----
            if chosen_option == 'A':
                # Continuation: all new nodes belong to v's axis
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                # Lateral: new axis starts at u
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            added_nodes += 1

        if not edge_accepted:
            # No valid edge found; should not happen under normal circumstances
            break

    return H


# Legacy (oldfrontexperimentations.py): Same as pareto_steiner_fast_preserve_tips, but uses fixed-spacing Steiner points
def pareto_steiner_fast_preserve_tips_fixed_spacing(G, alpha, min_length_factor=1.0, spacing_factor=1.0):
    """
    Same as pareto_steiner_fast_preserve_tips, but uses fixed-spacing Steiner points
    based on half the minimum lateral length instead of a fixed number of midpoints.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    in_nodes = {base_node}
    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        for u in unpaired_nodes:
            if H.nodes[u].get("is_tip", False):
                continue
            if not H.has_node(u):
                continue
            if u != base_node and H.degree(u) >= 3:
                continue
            if u == base_node and H.degree(u) >= 1:
                continue

            invalid_neighbors = []
            closest_neighbor = None
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                else:
                    closest_neighbor = v
                    break
            for invalid in invalid_neighbors:
                closest_neighbors[u].remove(invalid)

            if closest_neighbor is None:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[closest_neighbor]["pos"]
            edge_length = point_dist(p1, p2)
            total_root_length = edge_length
            total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]
            cost = pareto_cost(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                alpha=alpha,
            )
            insort(best_edges, (cost, u, closest_neighbor))

        if not best_edges:
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            if u != base_node and H.degree(u) >= 3:
                continue
            if u == base_node and H.degree(u) >= 1:
                continue

            u_has_child = len([n for n in H.neighbors(u) if n != base_node]) > 0
            if u != base_node and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0:
                    nude_zone = min_lateral_len
                    if H.nodes[u]["dist_to_axis_tip"] < nude_zone:
                        continue

            option_a_valid = False
            option_b_valid = False

            if u == base_node:
                option_a_valid = True
            else:
                old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                new_axis_total = old_axis_total + full_dist
                if new_axis_total > 0 and full_dist >= min_lateral_len:
                    if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            in_nodes.add(v)
            out_nodes.remove(v)

            # ---- USE FIXED SPACING INSTEAD OF FIXED NUMBER ----
            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint

                neighbors = []
                for out_node in out_nodes:
                    out_coord = G.nodes[out_node]["pos"]
                    dist = point_dist(midpoint, out_coord)
                    neighbors.append((dist, out_node))
                neighbors = sorted(neighbors)
                closest_neighbors[midpoint_node] = [n for _, n in neighbors]

                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)

            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                    node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            added_nodes += 1

        if not edge_accepted:
            break

    return H


# Legacy (oldfrontexperimentations.py): Fixed-spacing Steiner tree with tip preservation AND binary tree constraint.
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary(G, alpha, min_length_factor=1.0, spacing_factor=1.0):
    """
    Fixed-spacing Steiner tree with tip preservation AND binary tree constraint.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    in_nodes = {base_node}
    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    # Track children counts for binary tree constraint
    children_count = {base_node: 0}

    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        for u in unpaired_nodes:
            if H.nodes[u].get("is_tip", False):
                continue
            if not H.has_node(u):
                continue
            # Binary tree constraint
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            invalid_neighbors = []
            closest_neighbor = None
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                else:
                    closest_neighbor = v
                    break
            for invalid in invalid_neighbors:
                closest_neighbors[u].remove(invalid)

            if closest_neighbor is None:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[closest_neighbor]["pos"]
            edge_length = point_dist(p1, p2)
            total_root_length = edge_length
            total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]
            cost = pareto_cost(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                alpha=alpha,
            )
            insort(best_edges, (cost, u, closest_neighbor))

        if not best_edges:
            print(f"  No best_edges. Remaining tips: {len(out_nodes)}")
            print(f"  Unpaired nodes: {len(unpaired_nodes)}")
            for u in unpaired_nodes:
                print(f"    Node {u}: children={children_count.get(u,0)}, degree={H.degree(u)}")
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            # Binary tree constraint
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0:
                    nude_zone = min_lateral_len
                    if H.nodes[u]["dist_to_axis_tip"] < nude_zone:
                        continue

            option_a_valid = False
            option_b_valid = False

            if u == base_node:
                option_a_valid = True
            else:
                old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                new_axis_total = old_axis_total + full_dist
                if new_axis_total > 0 and full_dist >= min_lateral_len:
                    if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            in_nodes.add(v)
            out_nodes.remove(v)

            # Update children counts
            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint

                neighbors = []
                for out_node in out_nodes:
                    out_coord = G.nodes[out_node]["pos"]
                    dist = point_dist(midpoint, out_coord)
                    neighbors.append((dist, out_node))
                neighbors = sorted(neighbors)
                closest_neighbors[midpoint_node] = [n for _, n in neighbors]

                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)
                children_count[midpoint_node] = 1

            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                    node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            added_nodes += 1

        if not edge_accepted:
            break

    return H


# Legacy (oldfrontexperimentations.py): Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(G, alpha, min_length_factor=1/2,spacing_factor=1.0):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
    Selects tips based on total Pareto cost (not just Euclidean proximity).
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G,min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        for u in unpaired_nodes:
            if H.nodes[u].get("is_tip", False):  # ADD THIS
                continue
            if not H.has_node(u):
                continue
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            # Search ALL remaining tips and pick the one with best Pareto cost
            best_tip = None
            best_cost = float('inf')

            for v in out_nodes:
                p1 = H.nodes[u]["pos"]
                p2 = G.nodes[v]["pos"]
                edge_length = point_dist(p1, p2)

                total_root_length = edge_length
                total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]

                cost = pareto_cost(
                    total_root_length=total_root_length,
                    total_travel_distance=total_travel_distance,
                    alpha=alpha,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_tip = v

            if best_tip is not None:
                insort(best_edges, (best_cost, u, best_tip))

        if not best_edges:
            print(f"  No best_edges. Remaining tips: {len(out_nodes)}")
            print(f"  Unpaired nodes: {len(unpaired_nodes)}")
            for u in unpaired_nodes:
                print(f"    Node {u}: children={children_count.get(u,0)}, degree={H.degree(u)}")
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            full_dist = point_dist(p1, p2)

            u_has_child = children_count.get(u, 0) > 0
            if u != base_node and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0:
                    if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                        continue  # Too close to downstream tip
                    if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                        continue  # Too close to upstream branch point

            option_a_valid = False  # Continuation
            option_b_valid = False  # Lateral

            if u == base_node:
                option_a_valid = True
            else:
                old_axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                new_axis_total = old_axis_total + full_dist
                if new_axis_total > 0 and full_dist >= min_lateral_len:
                    if H.nodes[u]["dist_to_axis_tip"] >= min_lateral_len:
                        option_a_valid = True

            if u != base_node and u_has_child:
                if full_dist >= min_lateral_len:
                    option_b_valid = True

            if not option_a_valid and not option_b_valid:
                continue

            chosen_option = 'A' if option_a_valid else 'B'
            edge_accepted = True

            best_edges2 = []
            unpaired_nodes = {u}
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            out_nodes.remove(v)

            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            midpoints = steiner_points_fixed_spacing(p1, p2, spacing)
            midpoint_nodes = []

            for midpoint in midpoints:
                midpoint_node = node_index
                node_index += 1
                H.add_node(midpoint_node)
                H.nodes[midpoint_node]["pos"] = midpoint
                children_count[midpoint_node] = 1

                neighbors = []
                for out_node in out_nodes:
                    out_coord = G.nodes[out_node]["pos"]
                    dist = point_dist(midpoint, out_coord)
                    neighbors.append((dist, out_node))
                neighbors = sorted(neighbors)
                # We don't need closest_neighbors anymore since we search all tips each time
                # But Steiner points still need to be able to connect later
                if midpoint_node not in children_count:
                    children_count[midpoint_node] = 0

                midpoint_nodes.append(midpoint_node)
                unpaired_nodes.add(midpoint_node)

            line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
            for i in range(-1, -len(line_nodes), -1):
                n1 = line_nodes[i]
                n2 = line_nodes[i - 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        node_dist(H, n2, n1) + H.nodes[n1]["distance_to_base"]
                )

            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            added_nodes += 1

        if not edge_accepted:
            break

    return H


# Legacy (oldfrontexperimentations.py): Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
        G, alpha, min_length_factor=1 / 2, spacing_factor=1.0, verbose=False):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_lateral_len = get_min_lateral_length(G, min_length_factor)
    spacing = min_lateral_len * spacing_factor

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["axis_id"] = "axis_base"
    H.nodes[base_node]["dist_from_axis_start"] = 0.0
    H.nodes[base_node]["dist_to_axis_tip"] = 0.0

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    unpaired_nodes = {base_node}
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        if verbose:
            print(f"\n  Iteration {added_nodes}: {len(out_nodes)} tips remaining, "
                  f"{len(unpaired_nodes)} unpaired nodes")

        for u in unpaired_nodes:
            if H.nodes[u].get("is_tip", False):
                continue
            if not H.has_node(u):
                continue
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            best_tip = None
            best_cost = float('inf')
            best_edge_length = None

            for v in out_nodes:
                p1 = H.nodes[u]["pos"]
                p2 = G.nodes[v]["pos"]
                edge_length = point_dist(p1, p2)

                if edge_length < min_lateral_len:
                    if verbose:
                        print(f"    u={u} -> tip {v}: edge_length={edge_length:.1f} < min={min_lateral_len:.1f}, skip")
                    continue

                u_has_child = children_count.get(u, 0) > 0
                if u != base_node and u_has_child:
                    if H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                        if verbose:
                            print(
                                f"    u={u} -> tip {v}: nude zone (dist_to_axis_tip={H.nodes[u]['dist_to_axis_tip']:.1f}), skip")
                        continue
                    if H.nodes[u]["dist_from_axis_start"] < min_lateral_len:
                        if verbose:
                            print(
                                f"    u={u} -> tip {v}: too close to upstream branch (dist_from_axis_start={H.nodes[u]['dist_from_axis_start']:.1f}), skip")
                        continue

                total_root_length = edge_length
                total_travel_distance = edge_length + H.nodes[u]["distance_to_base"]

                cost = pareto_cost(
                    total_root_length=total_root_length,
                    total_travel_distance=total_travel_distance,
                    alpha=alpha,
                )

                if verbose:
                    print(f"    u={u} -> tip {v}: edge_len={edge_length:.1f}, "
                          f"travel={total_travel_distance:.1f}, cost={cost:.1f}")

                if cost < best_cost:
                    best_cost = cost
                    best_tip = v
                    best_edge_length = edge_length

            if best_tip is not None:
                insort(best_edges, (best_cost, u, best_tip))
                if verbose:
                    print(f"    => u={u} best tip={best_tip}, edge_len={best_edge_length:.1f}, cost={best_cost:.1f}")

        if not best_edges:
            if verbose:
                print(f"  No best_edges. Remaining tips: {len(out_nodes)}")
            break

        edge_accepted = False
        while best_edges and not edge_accepted:
            cost, u, v = best_edges.pop(0)

            if verbose:
                print(f"  Trying edge: u={u} -> tip={v}, cost={cost:.1f}")

            # ... rest of the edge acceptance logic unchanged ...

            if edge_accepted and verbose:
                print(f"  --> Accepted: u={u} -> tip={v}, option={chosen_option}, "
                      f"edge_len={full_dist:.1f}")

        if not edge_accepted:
            if verbose:
                print(f"  No edge accepted this iteration")
            break

    return H


# Legacy (oldfrontexperimentations.py): Continuous version of the Pareto Steiner tree algorithm.
def pareto_steiner_fast_continuous(G, alpha, min_length_factor=1.0,spacing_factor=1.0):
    """
    Continuous version of the Pareto Steiner tree algorithm.
    Attaches each tip at the optimal point along an existing edge,
    or directly to an existing node if that's optimal.

    Args:
        G: Original graph
        alpha: Pareto weight (0 = satellite, 1 = steiner)

    Returns:
        H: Pareto-optimal continuous Steiner tree
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G)
    if min_length <= 0:
        min_length = 10.0

    # Initialize tree with base
    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    # Step 1: Connect base to the closest tip
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)

    # Iteratively add remaining tips
    while out_nodes:
        best_cost = float('inf')
        best_action = None  # Will be tuple describing what to do

        # For each existing edge and each remaining tip,
        # find the optimal attachment
        for u, v in H.edges():
            for tip in out_nodes:
                cost, point, _, _, _, attach_type = optimal_attachment_point_ternary(
                    G, H, (u, v), tip, alpha, min_length
                )
                if cost < best_cost:
                    best_cost = cost
                    best_action = (u, v, tip, point, attach_type)

        if best_action is None:
            break

        u, v, tip, best_point, attach_type = best_action

        if attach_type == 'u':
            # Attach tip directly to node u
            dist_u_tip = point_dist(H.nodes[u]['pos'], G.nodes[tip]['pos'])
            H.add_node(tip)
            H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
            H.add_edge(u, tip, weight=dist_u_tip)
            H.nodes[tip]["distance_to_base"] = (
                    H.nodes[u]["distance_to_base"] + dist_u_tip
            )

        elif attach_type == 'v':
            # Attach tip directly to node v
            dist_v_tip = point_dist(H.nodes[v]['pos'], G.nodes[tip]['pos'])
            H.add_node(tip)
            H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
            H.add_edge(v, tip, weight=dist_v_tip)
            H.nodes[tip]["distance_to_base"] = (
                    H.nodes[v]["distance_to_base"] + dist_v_tip
            )

        else:  # attach_type == 'edge'
            # Insert a new internal node on the edge (u, v)
            new_node = node_index
            node_index += 1
            H.add_node(new_node, pos=tuple(best_point))

            # Remove old edge
            H.remove_edge(u, v)

            # Add split edges and connection to tip
            dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
            dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
            dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

            H.add_edge(u, new_node, weight=dist_u_new)
            H.add_edge(new_node, v, weight=dist_new_v)

            H.add_node(tip)
            H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
            H.add_edge(new_node, tip, weight=dist_new_tip)

            # Update distances
            dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
            H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
            H.nodes[tip]["distance_to_base"] = (
                    H.nodes[new_node]["distance_to_base"] + dist_new_tip
            )

            # Update v and all nodes downstream
            update_distances_downstream(
                H, v,
                H.nodes[new_node]["distance_to_base"] + dist_new_v
            )

        out_nodes.remove(tip)

    return H


# Legacy (oldfrontexperimentations.py): Find the optimal point to attach a tip to an existing edge using ternary search.
def optimal_attachment_point_ternary(G, H, edge, tip, alpha, min_length):
    """
    Find the optimal point to attach a tip to an existing edge using ternary search.
    Considers: attaching at u, at v, or somewhere along the edge.

    Returns:
        (cost, best_point, u, v, tip, attach_type)
        where attach_type is 'u', 'v', or 'edge'
    """
    u, v = edge
    p_u = np.array(H.nodes[u]['pos'])
    p_v = np.array(H.nodes[v]['pos'])
    p_tip = np.array(G.nodes[tip]['pos'])

    edge_vec = p_v - p_u
    edge_len = np.linalg.norm(edge_vec)

    best_cost = float('inf')
    best_point = None
    best_type = None

    # Option 1: Attach directly at u
    dist_u_tip = np.linalg.norm(p_tip - p_u)
    if dist_u_tip >= min_length:
        wiring = dist_u_tip
        travel = dist_u_tip + H.nodes[u].get('distance_to_base', 0)
        cost = pareto_cost(wiring, travel, alpha)
        if cost < best_cost:
            best_cost = cost
            best_point = p_u
            best_type = 'u'

    # Option 2: Attach directly at v
    dist_v_tip = np.linalg.norm(p_tip - p_v)
    if dist_v_tip >= min_length:
        dist_to_base_v = H.nodes[v].get('distance_to_base', 0)
        wiring = dist_v_tip
        travel = dist_v_tip + dist_to_base_v
        cost = pareto_cost(wiring, travel, alpha)
        if cost < best_cost:
            best_cost = cost
            best_point = p_v
            best_type = 'v'

    # Option 3: Attach along the edge interior using ternary search
    if edge_len > 1e-6 and edge_len >= 2 * min_length:
        edge_dir = edge_vec / edge_len
        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)

        t_min = min_length / edge_len
        t_max = (edge_len - min_length) / edge_len

        # Ternary search for optimal t
        for _ in range(20):
            t1 = t_min + (t_max - t_min) / 3
            t2 = t_max - (t_max - t_min) / 3

            point1 = p_u + t1 * edge_len * edge_dir
            point2 = p_u + t2 * edge_len * edge_dir

            d1 = np.linalg.norm(p_tip - point1)
            d2 = np.linalg.norm(p_tip - point2)

            f1 = alpha * d1 + (1 - alpha) * (d1 + dist_to_base_u + t1 * edge_len)
            f2 = alpha * d2 + (1 - alpha) * (d2 + dist_to_base_u + t2 * edge_len)

            if f1 < f2:
                t_max = t2
            else:
                t_min = t1

        t_opt = (t_min + t_max) / 2
        point = p_u + t_opt * edge_len * edge_dir
        dist_to_tip = np.linalg.norm(p_tip - point)

        if dist_to_tip >= min_length:
            wiring = dist_to_tip
            travel = dist_to_tip + dist_to_base_u + t_opt * edge_len
            cost = pareto_cost(wiring, travel, alpha)

            if cost < best_cost:
                best_cost = cost
                best_point = point
                best_type = 'edge'

    return (best_cost, best_point, u, v, tip, best_type)


# Legacy (oldfrontexperimentations.py): Find the optimal attachment point along an edge for a new tip using ternary search.
def find_optimal_t_on_edge(G, H, edge, tip, alpha, min_length, existing_tip_positions):
    """
    Find the optimal attachment point along an edge for a new tip using ternary search.

    Args:
        G: Original graph
        H: Current tree
        edge: Tuple (u, v) of the existing edge
        tip: The new tip to attach
        alpha: Pareto weight
        min_length: Minimum edge length
        existing_tip_positions: List of numpy arrays for existing tip positions

    Returns:
        (cost, best_point, u, v, tip, attach_type)
    """
    u, v = edge
    p_u = np.array(H.nodes[u]['pos'])
    p_v = np.array(H.nodes[v]['pos'])
    p_tip = np.array(G.nodes[tip]['pos'])

    edge_vec = p_v - p_u
    edge_len = np.linalg.norm(edge_vec)

    if edge_len < 1e-6:
        return (float('inf'), None, u, v, tip, 'edge')

    edge_dir = edge_vec / edge_len
    dist_to_base_u = H.nodes[u].get('distance_to_base', 0)

    # Edge too short — use midpoint
    if edge_len < 2 * min_length:
        midpoint = p_u + 0.5 * edge_vec
        dist_to_tip = np.linalg.norm(p_tip - midpoint)
        wiring = dist_to_tip
        travel = dist_to_tip + dist_to_base_u + 0.5 * edge_len
        cost = pareto_cost(wiring, travel, alpha)
        return (cost, midpoint, u, v, tip, 'small_edge')

    t_min = min_length / edge_len
    t_max = (edge_len - min_length) / edge_len

    # Ternary search for optimal t
    for _ in range(100):
        t1 = t_min + (t_max - t_min) / 3
        t2 = t_max - (t_max - t_min) / 3

        # Check if interval is already tiny
        if (t_max - t_min) * edge_len < 0.01:  # Less than 0.01 pixels
            break

        point1 = p_u + t1 * edge_len * edge_dir
        point2 = p_u + t2 * edge_len * edge_dir

        d1 = np.linalg.norm(p_tip - point1)
        d2 = np.linalg.norm(p_tip - point2)

        f1 = alpha * d1 + (1 - alpha) * (d1 + dist_to_base_u + t1 * edge_len)
        f2 = alpha * d2 + (1 - alpha) * (d2 + dist_to_base_u + t2 * edge_len)

        # Early stop if cost difference is negligible
        if abs(f1 - f2) < 1e-6:
            break

        if f1 < f2:
            t_max = t2
        else:
            t_min = t1

    t_opt = (t_min + t_max) / 2
    best_point = p_u + t_opt * edge_len * edge_dir

    # Check proximity to existing tips at the optimal point
    #for t_pos in existing_tip_positions:
    #   if np.linalg.norm(best_point - t_pos) < min_length:
    #        # Optimal point is too close to an existing tip — fall back to midpoint
    #        midpoint = p_u + 0.5 * edge_vec
    ##       dist_to_tip = np.linalg.norm(p_tip - midpoint)
    #       wiring = dist_to_tip
    #       travel = dist_to_tip + dist_to_base_u + 0.5 * edge_len
    #       cost = pareto_cost(wiring, travel, alpha)
    #       return (cost, midpoint, u, v, tip, 'small_edge')

    dist_to_tip = np.linalg.norm(p_tip - best_point)
    if dist_to_tip < min_length:
        return (float('inf'), None, u, v, tip, 'edge')

    wiring = dist_to_tip
    travel = dist_to_tip + dist_to_base_u + t_opt * edge_len
    cost = pareto_cost(wiring, travel, alpha)

    return (cost, best_point, u, v, tip, 'edge')


# Legacy (oldfrontexperimentations.py): Find the optimal attachment point along an edge for a new tip using ternary search.
def find_optimal_t_on_edge(G, H, edge, tip, alpha, min_length, existing_tip_positions, verbose=False):
    """
    Find the optimal attachment point along an edge for a new tip using ternary search.

    Args:
        G: Original graph
        H: Current tree
        edge: Tuple (u, v) of the existing edge
        tip: The new tip to attach
        alpha: Pareto weight
        min_length: Minimum edge length
        existing_tip_positions: List of numpy arrays for existing tip positions
        verbose: If True, print debug info for each evaluation

    Returns:
        (cost, best_point, u, v, tip, attach_type)
    """
    u, v = edge
    p_u = np.array(H.nodes[u]['pos'])
    p_v = np.array(H.nodes[v]['pos'])
    p_tip = np.array(G.nodes[tip]['pos'])

    edge_vec = p_v - p_u
    edge_len = np.linalg.norm(edge_vec)

    if edge_len < 1e-6:
        if verbose:
            print(f"    Edge ({u},{v}): zero length, skipping")
        return (float('inf'), None, u, v, tip, 'edge')

    edge_dir = edge_vec / edge_len
    dist_to_base_u = H.nodes[u].get('distance_to_base', 0)

    # Edge too short — use midpoint
    if edge_len < 2 * min_length:
        midpoint = p_u + 0.5 * edge_vec
        dist_to_tip = np.linalg.norm(p_tip - midpoint)
        wiring = dist_to_tip
        travel = dist_to_tip + dist_to_base_u + 0.5 * edge_len
        cost = pareto_cost(wiring, travel, alpha)
        if verbose:
            print(f"    Edge ({u},{v}): len={edge_len:.1f} < 2*min_length={2*min_length:.1f}, "
                  f"midpoint, cost={cost:.1f} (w={wiring:.1f}, t={travel:.1f})")
        return (cost, midpoint, u, v, tip, 'small_edge')

    t_min = min_length / edge_len
    t_max = (edge_len - min_length) / edge_len

    if verbose:
        print(f"    Edge ({u},{v}): len={edge_len:.1f}, t_range=[{t_min:.4f}, {t_max:.4f}], "
              f"dist_to_base_u={dist_to_base_u:.1f}")

    # Ternary search for optimal t
    for iteration in range(100):
        t1 = t_min + (t_max - t_min) / 3
        t2 = t_max - (t_max - t_min) / 3

        if (t_max - t_min) * edge_len < 0.01:
            if verbose:
                print(f"      iter {iteration}: interval < 0.01px, stopping")
            break

        point1 = p_u + t1 * edge_len * edge_dir
        point2 = p_u + t2 * edge_len * edge_dir

        d1 = np.linalg.norm(p_tip - point1)
        d2 = np.linalg.norm(p_tip - point2)

        f1 = alpha * d1 + (1 - alpha) * (d1 + dist_to_base_u + t1 * edge_len)
        f2 = alpha * d2 + (1 - alpha) * (d2 + dist_to_base_u + t2 * edge_len)

        if verbose:
            print(f"      iter {iteration}: t1={t1:.4f} (d={d1:.1f}, cost={f1:.1f}), "
                  f"t2={t2:.4f} (d={d2:.1f}, cost={f2:.1f}), "
                  f"interval_width={(t_max-t_min)*edge_len:.2f}px")

        if abs(f1 - f2) < 1e-6:
            if verbose:
                print(f"      iter {iteration}: costs converged, stopping")
            break

        if f1 < f2:
            t_max = t2
        else:
            t_min = t1

    t_opt = (t_min + t_max) / 2
    best_point = p_u + t_opt * edge_len * edge_dir

    dist_to_tip = np.linalg.norm(p_tip - best_point)
    if dist_to_tip < min_length:
        if verbose:
            print(f"    Edge ({u},{v}): dist_to_tip={dist_to_tip:.1f} < min_length={min_length:.1f}, rejecting")
        return (float('inf'), None, u, v, tip, 'edge')

    wiring = dist_to_tip
    travel = dist_to_tip + dist_to_base_u + t_opt * edge_len
    cost = pareto_cost(wiring, travel, alpha)

    if verbose:
        print(f"    Edge ({u},{v}): t_opt={t_opt:.4f}, dist_to_tip={dist_to_tip:.1f}, "
              f"wiring={wiring:.1f}, travel={travel:.1f}, cost={cost:.1f}")

    return (cost, best_point, u, v, tip, 'edge')


# Legacy (oldfrontexperimentations.py): Continuous Pareto Steiner tree that preserves tip integrity.
def pareto_steiner_fast_continuous_preserve_tips(G, alpha, min_length_factor=1/2, spacing_factor=1.0, verbose=False):
    """
    Continuous Pareto Steiner tree that preserves tip integrity.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["is_base"] = True

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    # Step 1: Connect base to the closest tip
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    if dist < min_length:
        return None

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.nodes[closest_tip]["is_tip"] = True
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)

    children_count = {base_node: 1, closest_tip: 0}

    # Precompute existing tip positions (updates each iteration)

    while out_nodes:
        # Precompute positions of tips currently in the tree
        existing_tip_positions = [
            np.array(H.nodes[n]['pos'])
            for n in H.nodes()
            if H.nodes[n].get('is_tip', False)
        ]

        best_cost = float('inf')
        best_action = None
        best_cost_small = float('inf')
        best_action_small = None

        for u, v in H.edges():
            for tip in out_nodes:
                cost, point, _, _, _, attach_type = find_optimal_t_on_edge(
                    G, H, (u, v), tip, alpha, min_length, existing_tip_positions,
                    verbose=verbose
                )

                if attach_type == 'edge':
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (u, v, tip, point)
                elif attach_type == 'small_edge':
                    if cost < best_cost_small:
                        best_cost_small = cost
                        best_action_small = (u, v, tip, point)

        if verbose and best_action is not None:
            u, v, tip, best_point = best_action
            print(f"  --> Selected edge ({u},{v}) for tip {tip}, cost={best_cost:.1f}")
        elif verbose and best_action_small is not None:
            u, v, tip, best_point = best_action_small
            print(f"  --> Selected small_edge ({u},{v}) for tip {tip}, cost={best_cost_small:.1f}")
        else:
            break

        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)

        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        children_count[new_node] = 2
        children_count[tip] = 0

        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(
            H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v
        )

        remaining = len(out_nodes) - 1
        total = len(critical_nodes) - 1
        print(f"    Connecting tips: {total - remaining}/{total} done", end="\r")
        out_nodes.remove(tip)

    print()
    return H


# Legacy (oldfrontexperimentations.py): Continuous Pareto Steiner tree with tip preservation and binary tree constraint.
def pareto_steiner_fast_continuous_preserve_tips_euclidean(G, alpha, min_length_factor=1.0, spacing_factor=1.0):
    """
    Continuous Pareto Steiner tree with tip preservation and binary tree constraint.
    Uses Euclidean nearest-neighbor for tip selection (greedy),
    then Pareto-optimal edge attachment for that tip.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length =  get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["is_base"] = True

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    # Precompute Euclidean nearest neighbors
    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    # Step 1: Connect base to the closest tip (Euclidean)
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    if dist < min_length:
        return None

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.nodes[closest_tip]["is_tip"] = True
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)

    children_count = {base_node: 1, closest_tip: 0}
    unpaired_nodes = {base_node, closest_tip}

    while out_nodes:
        existing_tip_positions = [
            np.array(H.nodes[n]['pos'])
            for n in H.nodes()
            if H.nodes[n].get('is_tip', False)
        ]

        best_cost = float('inf')
        best_action = None
        best_cost_small = float('inf')
        best_action_small = None

        # For each unpaired node, pick its Euclidean-closest tip
        for u in unpaired_nodes:
            if not H.has_node(u):
                continue
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue

            # Find Euclidean-closest unpaired tip to u
            invalid_neighbors = []
            closest_tip_to_u = None
            for i in range(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                elif v in out_nodes:
                    closest_tip_to_u = v
                    break
            for invalid in invalid_neighbors:
                closest_neighbors[u].remove(invalid)

            if closest_tip_to_u is None:
                continue

            # Find best edge to attach this tip (Pareto-optimal edge selection)
            for edge_u, edge_v in H.edges():
                cost, point, _, _, _, attach_type = find_optimal_t_on_edge(
                    G, H, (edge_u, edge_v), closest_tip_to_u, alpha, min_length,
                    existing_tip_positions
                )

                if attach_type == 'edge':
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (edge_u, edge_v, closest_tip_to_u, point)
                elif attach_type == 'small_edge':
                    if cost < best_cost_small:
                        best_cost_small = cost
                        best_action_small = (edge_u, edge_v, closest_tip_to_u, point)

        if best_action is not None:
            u, v, tip, best_point = best_action
        elif best_action_small is not None:
            u, v, tip, best_point = best_action_small
        else:
            break

        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)

        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        children_count[new_node] = 2
        children_count[tip] = 0

        # Update children_count for u (one more child)
        children_count[u] = children_count.get(u, 0) + 1

        # Remove u from unpaired if it reached max children
        if u != base_node and children_count[u] >= 2:
            unpaired_nodes.discard(u)
        elif u == base_node and children_count[u] >= 1:
            unpaired_nodes.discard(u)

        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(
            H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v
        )

        unpaired_nodes.add(new_node)
        unpaired_nodes.add(tip)

        remaining = len(out_nodes) - 1
        total = len(critical_nodes) - 1
        print(f"    Connecting tips: {total - remaining}/{total} done", end="\r")
        out_nodes.remove(tip)

    print()
    return H


# Legacy (oldfrontexperimentations.py): Given a graph G, compute the Pareto front of optimal solutions for the wiring cost and conduction delay.
def pareto_front_fixed_spacing(G):
    """Given a graph G, compute the Pareto front of optimal solutions for the wiring cost and conduction delay.

    This allows to compare how G was connected and how G could have been connected had it
    been trying to optimize wiring cost and conduction delay.

    When alpha = 0, the algorithm computes the satellite tree in linear time. This is the tree
    that connects the base node to all of the critical nodes in G.

    When alpha != 0, the algorithm computes the Pareto-optimal tree connecting the base node to all
    of the critical nodes in G. The algorithm attempts to optimize alpha * D + (1 - alpha) * W
    where D is the conduction delay and W is the wiring cost.

    Args:
        G (nx.Graph): The graph to compute the Pareto front for
    Returns:
        front (dict): A dictionary of edge_lengths, travel_distances_to_base for each alpha value on the front
        actual (tuple): The actual total_root_length, total_travel_distance of the original plant
    """
    critical_nodes = get_critical_nodes(G)

    # test: compute the actual total_root_length, total_travel_distance for the original plant
    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    # dictionary of edge_lengths, travel_distances_to_base for each alpha value on the front
    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        # if alpha = 0 compute the satellite tree in linear time
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_fixed_spacing(G, alpha)

        # compute the wiring cost and conduction delay
        # only the original critical nodes contribute to conduction delay
        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Given a graph G, compute the Pareto front of optimal solutions for the wiring cost and conduction delay.
def pareto_front_preserve_tips(G):
    """Given a graph G, compute the Pareto front of optimal solutions for the wiring cost and conduction delay.

    This allows to compare how G was connected and how G could have been connected had it
    been trying to optimize wiring cost and conduction delay.

    When alpha = 0, the algorithm computes the satellite tree in linear time. This is the tree
    that connects the base node to all of the critical nodes in G.

    When alpha != 0, the algorithm computes the Pareto-optimal tree connecting the base node to all
    of the critical nodes in G. The algorithm attempts to optimize alpha * D + (1 - alpha) * W
    where D is the conduction delay and W is the wiring cost.

    Args:
        G (nx.Graph): The graph to compute the Pareto front for
    Returns:
        front (dict): A dictionary of edge_lengths, travel_distances_to_base for each alpha value on the front
        actual (tuple): The actual total_root_length, total_travel_distance of the original plant
    """
    critical_nodes = get_critical_nodes(G)

    # test: compute the actual total_root_length, total_travel_distance for the original plant
    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    # dictionary of edge_lengths, travel_distances_to_base for each alpha value on the front
    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        # if alpha = 0 compute the satellite tree in linear time
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_preserve_tips(G, alpha)

        # compute the wiring cost and conduction delay
        # only the original critical nodes contribute to conduction delay
        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_preserve_tips_fixed_spacing(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_preserve_tips_fixed_spacing(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_preserve_tips_fixed_spacing_binary(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_preserve_tips_fixed_spacing_binary(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_preserve_tips_fixed_spacing_binary_pareto(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_continuous(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_continuous(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_continuous_preserve_tips(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_continuous_preserve_tips(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points.
def pareto_front_continuous_preserve_tips_euclidean(G):
    """Same as pareto_front_preserve_tips, but uses fixed-spacing Steiner points."""
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        H = None
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = pareto_steiner_fast_continuous_preserve_tips_euclidean(G, alpha)

        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual


# Legacy (oldfrontexperimentations.py): Find the optimal attachment point along an edge for a new tip using ternary search.
def find_optimal_t_on_edge(G, H, edge, tip, alpha, min_length, existing_tip_positions, verbose=False):
    """
    Find the optimal attachment point along an edge for a new tip using ternary search.
    """
    u, v = edge
    p_u = np.array(H.nodes[u]['pos'])
    p_v = np.array(H.nodes[v]['pos'])
    p_tip = np.array(G.nodes[tip]['pos'])

    edge_vec = p_v - p_u
    edge_len = np.linalg.norm(edge_vec)

    if edge_len < 1e-6:
        return (float('inf'), None, u, v, tip, 'edge')

    edge_dir = edge_vec / edge_len
    dist_to_base_u = H.nodes[u].get('distance_to_base', 0)

    if edge_len < 2 * min_length:
        midpoint = p_u + 0.5 * edge_vec
        dist_to_tip = np.linalg.norm(p_tip - midpoint)
        wiring = dist_to_tip
        travel = dist_to_tip + dist_to_base_u + 0.5 * edge_len
        cost = pareto_cost(wiring, travel, alpha)
        return (cost, midpoint, u, v, tip, 'small_edge')

    t_min = min_length / edge_len
    t_max = (edge_len - min_length) / edge_len

    for _ in range(100):
        t1 = t_min + (t_max - t_min) / 3
        t2 = t_max - (t_max - t_min) / 3

        if (t_max - t_min) * edge_len < 0.01:
            break

        point1 = p_u + t1 * edge_len * edge_dir
        point2 = p_u + t2 * edge_len * edge_dir

        d1 = np.linalg.norm(p_tip - point1)
        d2 = np.linalg.norm(p_tip - point2)

        f1 = alpha * d1 + (1 - alpha) * (d1 + dist_to_base_u + t1 * edge_len)
        f2 = alpha * d2 + (1 - alpha) * (d2 + dist_to_base_u + t2 * edge_len)

        if abs(f1 - f2) < 1e-6:
            break

        if f1 < f2:
            t_max = t2
        else:
            t_min = t1

    t_opt = (t_min + t_max) / 2
    best_point = p_u + t_opt * edge_len * edge_dir

    dist_to_tip = np.linalg.norm(p_tip - best_point)
    if dist_to_tip < min_length:
        return (float('inf'), None, u, v, tip, 'edge')

    wiring = dist_to_tip
    travel = dist_to_tip + dist_to_base_u + t_opt * edge_len
    cost = pareto_cost(wiring, travel, alpha)

    return (cost, best_point, u, v, tip, 'edge')


# Legacy (oldfrontexperimentations.py): Find the optimal attachment point along an edge for a new tip using
def find_optimal_t_on_edge_analytical_old(G, H, edge, tip, alpha, min_length,
                                      existing_tip_positions=None, verbose=False):
    """
    Find the optimal attachment point along an edge for a new tip using
    the analytical solution of the Pareto cost derivative.

    Same signature as find_optimal_t_on_edge for drop-in replacement.
    existing_tip_positions is accepted for compatibility but not used.
    """
    u, v = edge
    p_u = np.array(H.nodes[u]['pos'])
    p_v = np.array(H.nodes[v]['pos'])
    p_t = np.array(G.nodes[tip]['pos'])

    e_vec = p_v - p_u
    L = np.linalg.norm(e_vec)

    if L < 1e-6:
        return (float('inf'), None, u, v, tip, 'edge')

    e_hat = e_vec / L
    dist_to_base_u = H.nodes[u].get('distance_to_base', 0)

    # Edge too short for valid attachment
    if L < 2 * min_length:
        midpoint = p_u + 0.5 * e_vec
        dist_to_tip = np.linalg.norm(p_t - midpoint)
        wiring = dist_to_tip
        travel = dist_to_tip + dist_to_base_u + 0.5 * L
        cost = pareto_cost(wiring, travel, alpha)
        return (cost, midpoint, u, v, tip, 'small_edge')

    t_min = min_length / L
    t_max = 1.0 - t_min

    w_vec = p_t - p_u  # from u to tip
    w_dot_e = np.dot(w_vec, e_vec)
    w_norm_sq = np.dot(w_vec, w_vec)

    def cost_at(t):
        point = p_u + t * e_vec
        d = np.linalg.norm(p_t - point)
        wiring = d
        travel = d + dist_to_base_u + t * L
        return pareto_cost(wiring, travel, alpha), d

    candidates = [t_min, t_max]  # boundary points

    if alpha == 1.0:
        # Pure wiring cost: perpendicular projection onto edge line
        t_perp = w_dot_e / (L * L)
        t_perp = np.clip(t_perp, t_min, t_max)
        candidates.append(t_perp)

    elif alpha == 0.0:
        # Pure travel cost: minimum is at smallest feasible t
        # Already have t_min in candidates; nothing else needed
        pass

    else:
        # 0 < alpha < 1: solve quadratic from analytical derivation
        k = 2 * alpha - alpha * alpha  # 1 - (1-alpha)^2, > 0 for alpha in (0,1)

        a = L * L
        b = -2.0 * w_dot_e
        c = (w_dot_e * w_dot_e - (1 - alpha) ** 2 * L * L * w_norm_sq) / (k * L * L)

        disc = b * b - 4.0 * a * c

        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            for root in [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]:
                if t_min <= root <= t_max:
                    # Verify derivative sign: LHS of critical eq must be negative
                    point = p_u + root * e_vec
                    d_root = np.linalg.norm(p_t - point)
                    if d_root > 1e-12:
                        lhs = (L * L * root - w_dot_e) / d_root
                        rhs = -(1.0 - alpha) * L
                        if abs(lhs - rhs) < 1e-8 or lhs < 0:
                            candidates.append(root)

    # Evaluate all candidates
    best_cost = float('inf')
    best_t = None
    for t in candidates:
        if t_min - 1e-10 <= t <= t_max + 1e-10:
            t = np.clip(t, t_min, t_max)
            cost, d = cost_at(t)
            if d < min_length:
                continue
            if cost < best_cost:
                best_cost = cost
                best_t = t

    if best_t is None:
        return (float('inf'), None, u, v, tip, 'edge')

    best_point = p_u + best_t * e_vec
    dist_to_tip = np.linalg.norm(p_t - best_point)

    if dist_to_tip < min_length:
        return (float('inf'), None, u, v, tip, 'edge')

    if verbose:
        print(f"    Edge ({u},{v}): t_opt={best_t:.4f}, dist_to_tip={dist_to_tip:.1f}, cost={best_cost:.1f}")

    return (best_cost, tuple(best_point), u, v, tip, 'edge')


# Legacy (oldfrontexperimentations.py): Continuous Pareto Steiner tree that preserves tip integrity.
def pareto_steiner_fast_continuous_preserve_tips(G, alpha, min_length_factor=1/2,
                                                  spacing_factor=1.0, verbose=False):
    """
    Continuous Pareto Steiner tree that preserves tip integrity.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["is_base"] = True

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    # Step 1: Connect base to the closest tip
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    if dist < min_length:
        return None

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.nodes[closest_tip]["is_tip"] = True
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)
    children_count = {base_node: 1, closest_tip: 0}

    if verbose:
        print(f"  Step 1: base -> tip {closest_tip}, dist={dist:.1f}")
        print(f"  Remaining tips: {len(out_nodes)}")

    iteration = 2
    while out_nodes:
        existing_tip_positions = [
            np.array(H.nodes[n]['pos'])
            for n in H.nodes()
            if H.nodes[n].get('is_tip', False)
        ]

        best_cost = float('inf')
        best_action = None
        best_cost_small = float('inf')
        best_action_small = None

        for u, v in H.edges():
            for tip in out_nodes:
                cost, point, eu, ev, etip, attach_type = find_optimal_t_on_edge(
                    G, H, (u, v), tip, alpha, min_length, existing_tip_positions
                )

                if attach_type == 'edge':
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (u, v, tip, point)
                elif attach_type == 'small_edge':
                    if cost < best_cost_small:
                        best_cost_small = cost
                        best_action_small = (u, v, tip, point)

        if best_action is not None:
            u, v, tip, best_point = best_action
            if verbose:
                print(f"  Step {iteration}: edge({u},{v}) -> tip {tip}, cost={best_cost:.1f}")
        elif best_action_small is not None:
            u, v, tip, best_point = best_action_small
            if verbose:
                print(f"  Step {iteration}: small_edge({u},{v}) -> tip {tip}, cost={best_cost_small:.1f}")
        else:
            if verbose:
                print(f"  Step {iteration}: no valid action, breaking")
            break

        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)

        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        children_count[new_node] = 2
        children_count[tip] = 0

        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(
            H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v
        )

        out_nodes.remove(tip)
        iteration += 1

    return H


# Legacy (oldfrontexperimentations.py): Continuous Pareto Steiner tree that preserves tip integrity.
def pareto_steiner_fast_analytical_preserve_tips_old(G, alpha, min_length_factor=1/2,
                                                  spacing_factor=1.0, verbose=False):
    """
    Continuous Pareto Steiner tree that preserves tip integrity.
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    H.nodes[base_node]["pos"] = G.nodes[base_node]["pos"]
    H.nodes[base_node]["is_base"] = True

    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    node_index = max(critical_nodes) + 1

    # Step 1: Connect base to the closest tip
    base_pos = H.nodes[base_node]["pos"]
    closest_tip = min(out_nodes,
                      key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    if dist < min_length:
        return None

    H.add_node(closest_tip)
    H.nodes[closest_tip]["pos"] = tip_pos
    H.nodes[closest_tip]["is_tip"] = True
    H.add_edge(base_node, closest_tip, weight=dist)
    H.nodes[closest_tip]["distance_to_base"] = dist

    out_nodes.remove(closest_tip)
    children_count = {base_node: 1, closest_tip: 0}

    if verbose:
        print(f"  Step 1: base -> tip {closest_tip}, dist={dist:.1f}")
        print(f"  Remaining tips: {len(out_nodes)}")

    iteration = 2
    while out_nodes:
        existing_tip_positions = [
            np.array(H.nodes[n]['pos'])
            for n in H.nodes()
            if H.nodes[n].get('is_tip', False)
        ]

        best_cost = float('inf')
        best_action = None
        best_cost_small = float('inf')
        best_action_small = None

        for u, v in H.edges():
            for tip in out_nodes:
                cost, point, eu, ev, etip, attach_type = find_optimal_t_on_edge_analytical_old(
                    G, H, (u, v), tip, alpha, min_length, existing_tip_positions
                )

                if attach_type == 'edge':
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (u, v, tip, point)
                elif attach_type == 'small_edge':
                    if cost < best_cost_small:
                        best_cost_small = cost
                        best_action_small = (u, v, tip, point)

        if best_action is not None:
            u, v, tip, best_point = best_action
            if verbose:
                print(f"  Step {iteration}: edge({u},{v}) -> tip {tip}, cost={best_cost:.1f}")
        elif best_action_small is not None:
            u, v, tip, best_point = best_action_small
            if verbose:
                print(f"  Step {iteration}: small_edge({u},{v}) -> tip {tip}, cost={best_cost_small:.1f}")
        else:
            if verbose:
                print(f"  Step {iteration}: no valid action, breaking")
            break

        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)

        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(best_point, H.nodes[v]['pos'])
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        children_count[new_node] = 2
        children_count[tip] = 0

        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(
            H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v
        )

        out_nodes.remove(tip)
        iteration += 1

    return H


# Legacy (oldfrontexperimentations.py): Compare Pareto fronts from the analytical and continuous (ternary search)
def compare_analytical_vs_continuous(G, alphas=None, min_length_factor=0.5, dest=None):
    """
    Compare Pareto fronts from the analytical and continuous (ternary search)
    attachment point algorithms. They should be nearly identical if both are correct.

    Args:
        G: Input graph
        alphas: List of alpha values to evaluate. Default: 11 values from 0 to 1.
        min_length_factor: Minimum lateral length factor
        dest: Optional path to save the plot
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 11)

    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']),
                                           np.array(H.nodes[v]['pos']))

    critical_nodes = get_critical_nodes(H)
    orig_w, orig_t = graph_costs(H, critical_nodes=critical_nodes)

    # Compute both fronts
    front_analytical = {}
    front_continuous = {}

    for alpha in alphas:
        print(f"Alpha={alpha:.2f}")

        # Analytical
        tree_a = pareto_steiner_fast_analytical_preserve_tips(
            H, alpha=alpha, min_length_factor=min_length_factor
        )
        if tree_a is not None:
            aw, at = graph_costs(tree_a, critical_nodes=critical_nodes)
            front_analytical[alpha] = (aw, at)

        # Continuous (ternary search)
        tree_c = pareto_steiner_fast_continuous_preserve_tips(
            H, alpha=alpha, min_length_factor=min_length_factor
        )
        if tree_c is not None:
            cw, ct = graph_costs(tree_c, critical_nodes=critical_nodes)
            front_continuous[alpha] = (cw, ct)
    print("Continuous front values:")
    for a in sorted(front_continuous.keys()):
        print(f"  alpha={a:.2f}: {front_continuous[a]}")
    # Scaling
    try:
        from ariadne_roots import config
        scale_factor = config.length_scale_factor
        scale_unit = config.length_scale_unit
    except (ImportError, AttributeError):
        scale_factor = 1.0
        scale_unit = "pixels"

    def scale_data(data):
        return data * scale_factor

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax1, ax2 = axes

    # --- Left: Overlay both fronts ---
    if front_analytical:
        ax_fx = [scale_data(v[0]) for v in front_analytical.values()]
        ax_fy = [scale_data(v[1]) for v in front_analytical.values()]
        ax1.plot(ax_fx, ax_fy, 's-', color='blue', linewidth=2, markersize=8,
                 label='Analytical', zorder=3)

    if front_continuous:
        cx_fx = [scale_data(v[0]) for v in front_continuous.values()]
        cx_fy = [scale_data(v[1]) for v in front_continuous.values()]
        ax1.plot(cx_fx, cx_fy, 'o--', color='red', linewidth=2, markersize=8,
                 markerfacecolor='none', label='Continuous (ternary)', zorder=2)

    ax1.scatter(scale_data(orig_w), scale_data(orig_t),
                marker='*', color='black', s=200, zorder=5,
                label='Actual Tree', edgecolors='white')

    ax1.set_xlabel(f"Total length ({scale_unit})", fontsize=12)
    ax1.set_ylabel(f"Travel distance ({scale_unit})", fontsize=12)
    ax1.set_title("Pareto Fronts: Analytical vs Continuous", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Right: Difference plot ---
    common_alphas = sorted(set(front_analytical.keys()) & set(front_continuous.keys()))
    if common_alphas:
        dw = []
        dt = []
        for a in common_alphas:
            dw.append(front_analytical[a][0] - front_continuous[a][0])
            dt.append(front_analytical[a][1] - front_continuous[a][1])

        ax2.plot(common_alphas, dw, 's-', color='blue', linewidth=2, markersize=6,
                 label='Δ Wiring (analytical − continuous)')
        ax2.plot(common_alphas, dt, 'o-', color='red', linewidth=2, markersize=6,
                 label='Δ Travel (analytical − continuous)')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        ax2.set_xlabel("α", fontsize=12)
        ax2.set_ylabel(f"Difference ({scale_unit})", fontsize=12)
        ax2.set_title("Difference: Analytical − Continuous", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Print summary
        print(
            f"\n{'Alpha':<8} {'A_wiring':<12} {'C_wiring':<12} {'Δ_w':<12} {'A_travel':<12} {'C_travel':<12} {'Δ_t':<12}")
        print("-" * 80)
        for a in common_alphas:
            aw, at = front_analytical[a]
            cw, ct = front_continuous[a]
            print(f"{a:<8.2f} {aw:<12.1f} {cw:<12.1f} {aw - cw:<+12.1f} {at:<12.1f} {ct:<12.1f} {at - ct:<+12.1f}")

    plt.tight_layout()

    if dest is not None:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
        from pathlib import Path
        svg_dest = Path(dest).with_suffix(".svg")
        plt.savefig(svg_dest, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        plt.show()

    return fig, axes


# Moved from quantify.py: Construct a broken graph (without problematic edges)..
def make_graph_alt(target):  # pragma: no cover
    """Construct a broken graph (without problematic edges).

    Legacy function for text file format. Not used in current JSON workflow.
    """
    G = nx.Graph()
    with open(target, "r") as f:  # parse input file
        q = Queue()
        node_num = 1  # label nodes with unique identifiers
        for line in f:
            if line.startswith("##"):  # Level heading
                group_num = 0  # count nodes per level, and reset on level change, to match hierarchy info from tuples
                level = int(line.rstrip().split(": ")[1])
                continue
            else:
                info = line.rstrip().split("; ")
                if len(info) > 1:  # node has degree > 1
                    coords = tuple(int(float(i)) for i in info[0].split())[
                        0:2
                    ]  # change output coords from floats to ints
                    G.add_node(node_num, pos=coords)
                    if not q.empty():
                        parent_node = q.get()
                        # print(parent_node, level, group_num, info)
                        if (
                            level == parent_node[1][0]
                            and group_num == parent_node[1][1]
                        ):  # check that the expected and actual positions of the child match
                            G.add_edge(
                                node_num,
                                parent_node[0],
                                length=distance(
                                    G.nodes[node_num]["pos"],
                                    G.nodes[parent_node[0]]["pos"],
                                ),
                            )
                        else:
                            print(
                                f"Edge assignment failed: {parent_node}; {level}; {group_num}; {info}"
                            )
                    # place all descendants of the current node in the queue for processing in future rounds
                    children = info[1].split()
                    for child_node in children:
                        q.put(
                            (
                                node_num,
                                list(map(int, child_node.strip("[]").split(","))),
                            )
                        )  # converts each child object from list of strings to list of ints
                else:  # terminal node (degree == 1)
                    coords = tuple(int(float(i)) for i in info[0].rstrip(";").split())[
                        0:2
                    ]
                    G.add_node(node_num, pos=coords)
                    children = None
                    parent_node = q.get()
                    if level == parent_node[1][0] and group_num == parent_node[1][1]:
                        G.add_edge(
                            node_num,
                            parent_node[0],
                            length=distance(
                                G.nodes[node_num]["pos"], G.nodes[parent_node[0]]["pos"]
                            ),
                        )
                    else:
                        print("Edge assignment failed: terminal node.")
                node_num += 1
                group_num += 1
    return G


# Moved from quantify.py: Plot a Pareto front and save to .jpg..
def save_plot(path, name, title):  # pragma: no cover
    """Plot a Pareto front and save to .jpg.

    GUI function for manual visualization. Not tested in automated test suite.
    """
    G = make_graph(path)
    # check that graph is indeed a tree (acyclic, undirected, connected)
    assert nx.is_tree(G)

    edge_lengths, travel_distances_to_base, actual = pareto_front(G)
    randoms = random_tree(G)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Total length", fontsize=15)
    ax.set_ylabel("Travel distance", fontsize=15)

    plt.plot(
        edge_lengths,
        travel_distances_to_base,
        marker="s",
        linestyle="-",
        markeredgecolor="black",
    )
    plt.plot(actual[0], actual[1], marker="x", markersize=12)
    for i in randoms:
        plt.plot(i[0], i[1], marker="+", color="green", markersize=4)

    plt.show()


# Moved from quantify.py: Construct graph from file and check for errors..
def make_graph(target):  # pragma: no cover
    """Construct graph from file and check for errors.

    Legacy function for text file format. Not used in current JSON workflow.
    """
    G = nx.Graph()
    with open(target, "r") as f:  # parse input file
        q = Queue()
        node_num = 1  # label nodes with unique identifiers
        for line in f:
            if line.startswith("##"):  # Level heading
                group_num = 0  # count nodes per level, and reset on level change, to match hierarchy info from tuples
                level = int(line.rstrip().split(": ")[1])
                continue
            else:
                info = line.rstrip().split("; ")

                coords = tuple(int(float(i)) for i in info[0].split())[
                    0:2
                ]  # change output coords from floats to ints
                metadata = re.findall(
                    r"\(.+?\)|\[.+?\]", info[1]
                )  # find chunks in brackets or parenthesis

                root_metadata = metadata[-1]  # eg (PR, None)
                child_metadata = []  # eg ['[1,0]']

                for el in metadata:
                    if el != root_metadata:
                        child_metadata.append(el)

                if not child_metadata:  # terminal node, no children
                    G.add_node(node_num, pos=coords)
                    parent_node = q.get()

                    parent_level = parent_node[1][0]
                    parent_group = parent_node[1][1]

                    if level == parent_level and group_num == parent_group:
                        G.add_edge(
                            node_num,
                            parent_node[0],
                            length=distance(
                                G.nodes[node_num]["pos"], G.nodes[parent_node[0]]["pos"]
                            ),
                        )
                    else:
                        print("ERROR: edge assignment failed (terminal node)")

                else:
                    G.add_node(node_num, pos=coords)

                    if not q.empty():
                        parent_node = q.get()

                        parent_level = parent_node[1][0]
                        parent_group = parent_node[1][1]

                        if level == parent_level and group_num == parent_group:
                            G.add_edge(
                                node_num,
                                parent_node[0],
                                length=distance(
                                    G.nodes[node_num]["pos"],
                                    G.nodes[parent_node[0]]["pos"],
                                ),
                            )
                        else:
                            print("Error: edge assignment failed")

                    for child_node in child_metadata:
                        q.put(
                            (
                                node_num,
                                list(map(int, child_node.strip("[]").split(","))),
                            )
                        )

                node_num += 1
                group_num += 1

    # return "Done!" (used for csv creation)
    return G


# G = make_graph('/Users/kianfaizi/projects/ariadne/color-final_plantA_day1.txt')


# Moved from quantify.py: Calculate lateral root density..
def calc_density_LRs(G):  # pragma: no cover
    """Calculate lateral root density.

    Stub function - not yet implemented.
    """
    pass
    # add up to _n_ degrees


# Moved duplicate of quantify.distance().
def distance(p1, p2):
    """Compute 2D Euclidian distance between two (x,y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Moved duplicate of quantify.distance().
def distance(p1, p2):
    """Compute 2D Euclidian distance between two (x,y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Moved from pareto_functions.py: Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017..
def random_tree(G):
    """Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017.
    Only consider the critical nodes (and root node) of G.

    Args:
        G (nx.Graph): The graph to compute the random trees for
    Returns:
        costs (list): A list of (mactual, sactual) tuples for each random tree which
            represent the wiring cost and conduction delay of the tree.
    """
    random.seed(a=None)
    random_trees = []  # list of 1000 random trees
    costs = []

    for i in range(1000):  # 1000 random trees
        # instantiate random tree
        R = nx.Graph()
        G_critical_nodes = get_critical_nodes(G)

        while len(G_critical_nodes) > 0:
            # randomly draw 1 node from G's critical nodes
            index = random.randrange(len(G_critical_nodes))
            g = G_critical_nodes[index]

            if len(R.nodes) > 0:  # if R is not empty
                # add the new point AND a random edge
                r_index = random.randrange(len(R.nodes))  # get a random node from R
                r = list(R.nodes)[r_index]
                R.add_node(g, pos=G.nodes.data()[g]["pos"])
                R.add_edge(r, g, weight=node_dist(R, r, g))

            else:  # if R is empty
                # add the new point
                R.add_node(g, pos=G.nodes.data()[g]["pos"])

            # remove added node from candidate list and repeat
            del G_critical_nodes[index]

        random_trees.append(R)

    for R in random_trees:
        # compute costs for each R, to compare with G
        mactual, sactual = graph_costs(R)
        costs.append((mactual, sactual))

    return costs


# Legacy: convex-hull area helper used only by standalone trait tests.
def calculate_convex_hull_area(G):
    # Check if the graph has at least 3 nodes
    if len(G.nodes) < 3:
        print("The graph must have at least 3 nodes to calculate the convex hull.")
        return None

    # Get the positions of the nodes
    positions = np.array([data["pos"] for node, data in G.nodes(data=True)])

    # Calculate the convex hull
    hull = ConvexHull(positions)

    # Get the area of the convex hull
    hull_area = hull.area

    return hull_area
