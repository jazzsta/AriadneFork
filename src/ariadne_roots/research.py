"""Ariadne 0.3 research workflow.

This module follows the scientific story in *Road to Ariadne 0.3*:
straightening, tip preservation, fixed-spacing and continuous fronts,
random null models, optimization landscape, and sampling-based
classification. It is import-safe and is intended to be used in
``Ariadne0.3_Exploration.ipynb`` or an interactive Python session.
"""

import copy
import random
from bisect import insort

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

from ariadne_roots.pareto_functions import (
    DEFAULT_ALPHAS,
    STEINER_MIDPOINTS,
    _tree_hover_payload,
    clean_pareto_front,
    find_optimal_t_on_analytical,
    get_critical_nodes,
    get_min_lateral_length,
    graph_costs,
    k_nearest_neighbors,
    node_dist,
    pareto_cost,
    pareto_front,
    pareto_steiner_fast_analytical_preserve_tips_semi_random,
    point_dist,
    satellite_tree,
    steiner_points,
    straighten_tree,
    update_distances_downstream,
)
from ariadne_roots.quantify import (
    estimate_rounds_for_region,
    save_sampling_html,
)


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
        # critnodes=get_critical_nodes(R)
        mactual, sactual = graph_costs(R)#,critnodes)
        costs.append((mactual, sactual))

    return costs, random_trees


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


def place_internal_nodes_only(G_input, verbose=True):
    """
    Take an input graph G and add N-2 internal nodes placed within the
    expanded bounding box of the critical nodes, with bias toward the barycenter.
    Uses get_min_lateral_length for minimum inter-node spacing.
    """

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
    assert critical_nodes[0] == critical_nodes_sorted[0], f"Base node must be naming convention error"

    N = len(critical_nodes)
    n_internal = N - 2

    # Use get_min_lateral_length as the minimum spacing
    min_distance = get_min_lateral_length(G_input)

    if verbose:
        print(f"Input: {N} critical nodes (1 base + {N - 1} tips)")
        print(f"Need to place: {n_internal} internal nodes")
        print(f"Min distance (from laterals): {min_distance:.1f}")
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

    # Shrink the hull to 85% of its area
    shrink_factor = np.sqrt(0.85)
    shrunk_vertices = hull_centroid + shrink_factor * (hull.points[hull.vertices] - hull_centroid)

    # Build the shrunk hull while keeping base root area connected
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
        std_x = (max_coords[0] - min_coords[0])/2
        std_y = abs(G_new.nodes[node]['pos'][1])/2
        candidate = np.array([
            np.random.normal(barycenter[0], std_x),
            np.random.normal(G_new.nodes[node]['pos'][1] / 2, std_y)
        ])

        # Constrain to shrunk hull
        path = MplPath(shrunk_hull.points[shrunk_hull.vertices])
        if not path.contains_points([candidate])[0]:
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

        if verbose:
            print(
                f"  Tip {node} (Y={G_new.nodes[node]['pos'][1]:.0f}) -> internal {next_id - 1} (Y={candidate[1]:.0f})")

        # Bounce away from collisions using min_distance from lateral lengths
        max_bounces = 20
        bounces = 0
        too_close = True

        while too_close and bounces < max_bounces:
            too_close = False
            for existing_node in G_new.nodes():
                existing_pos = G_new.nodes[existing_node]['pos']
                dist = np.linalg.norm(candidate - existing_pos)
                if dist < min_distance:
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = np.array([np.cos(angle), np.sin(angle)]) * 2 * min_distance
                    else:
                        direction = (candidate - existing_pos) / dist
                        push = direction * 2 * min_distance
                    candidate = candidate + push
                    candidate = np.clip(candidate, min_coords, max_coords)
                    too_close = True
                    bounces += 1
                    break

        G_new.add_node(next_id, pos=candidate, is_internal=True)
        next_id += 1

    lastnode = critical_nodes_sorted[len(critical_nodes) - 1]
    G_new.add_node(lastnode, pos=positions[lastnode])
    G_new.nodes[lastnode]['is_base'] = (lastnode == base_node)
    G_new.nodes[lastnode]['is_tip'] = (lastnode != base_node)

    if verbose:
        print(f"\nResult: Graph with {G_new.number_of_nodes()} nodes")

    return G_new


def add_one_branch(G, max_order, min_length, randomness=0.3, verbose=False):
    """
    Add edges for ONE parent node.
    - If base has no child: connect base to one internal (order 0)
    - If parent has 1 child: add one lateral child (order + 1)
    - If parent has 0 children: add continuation child (same order)
    Prioritizes internals over tips when the parent can still branch deeper.
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
            if verbose:
                print(f"    DEBUG Case 1 fail: No unconnected internals for base")
            return False
        child = min(available,
                    key=lambda n: np.linalg.norm(G.nodes[n]['pos'] - G.nodes[base_node]['pos']))
        G.add_edge(base_node, child)
        G.nodes[child]['root_deg'] = 0
        if verbose:
            print(f"    Base -> {child} (order 0)")
        return True

    # At the start of Case 2, add:
    tips_total = [n for n in G.nodes() if G.nodes[n].get('is_tip', False)]
    tips_connected = sum(1 for n in tips_total if G.degree(n) > 0)
    if tips_connected == len(tips_total):
        # Tree is complete
        return False


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
        parents.append((G.nodes[n]['pos'][1], n))

    if not parents:
        if verbose:
            print(f"    DEBUG Case 2 fail: No available parents")
        return False

    # Pick the parent with lowest Y (closest to base, highest priority)
    parents.sort(key=lambda x: x[0])
    _, parent = parents[0]

    parent_order = G.nodes[parent]['root_deg']
    parent_pos = G.nodes[parent]['pos']

    # Count current children: neighbors that are NOT the parent
    # A child has root_deg >= parent_order (or hasn't been assigned yet)
    children = []
    for n in G.neighbors(parent):
        if G.nodes[n].get('is_base', False):
            continue
        child_deg = G.nodes[n].get('root_deg', -1)
        if child_deg >= parent_order or child_deg == -1:
            children.append(n)

    # Determine what order the next child should have
    if len(children) == 0:
        # First child
        if parent_order == max_order - 1:
            # At penultimate order: first child is LATERAL (order + 1 = max_order)
            child_order = parent_order + 1
        else:
            # Normal case: first child is continuation
            child_order = parent_order
    else:
        # Second child
        if parent_order == max_order - 1:
            # At penultimate order: second child is CONTINUATION (same order)
            child_order = parent_order
        else:
            # Normal case: second child is lateral
            child_order = parent_order + 1

    # ---- GATHER CANDIDATES WITH PRIORITY ----
    can_branch = (child_order < max_order)  # Can this child have its own laterals?

    candidates = []

    # If this child can branch further, prioritize internals over tips
    # to ensure the tree goes deep before going wide
    if can_branch:
        # Internals first (they can continue branching)
        for i in G.nodes():
            if not G.nodes[i].get('is_internal', False):
                continue
            if G.degree(i) > 0:
                continue
            i_pos = G.nodes[i]['pos']
            dist = np.linalg.norm(i_pos - parent_pos)
            candidates.append((dist, i, 'internal'))

        # Then tips
        for t in G.nodes():
            if not G.nodes[t].get('is_tip', False):
                continue
            if G.degree(t) > 0:
                continue
            t_pos = G.nodes[t]['pos']
            dist = np.linalg.norm(t_pos - parent_pos)
            if t_pos[1] < parent_pos[1]:
                dist *= 0.5  # Priority for missed tips
            candidates.append((dist * 2.0, t, 'tip'))  # Penalize tips when we can branch
    else:
        # Only tips can be children (they can't branch further)
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

    if not candidates:
        if verbose:
            print(f"    DEBUG Case 3 fail: No candidates for parent {parent} (order={parent_order})")
        return False

    # ---- PENALIZE CANDIDATES THAT CREATE NEAR-PARALLEL EDGES ----
    # Use half the minimum lateral length as the proximity threshold

    proximity_threshold = min_length  # Precomputed once

    if G.number_of_edges() > 0:
        penalized_candidates = []
        for dist, node, typ in candidates:
            node_pos = G.nodes[node]['pos']
            new_vec = node_pos - parent_pos
            new_len = np.linalg.norm(new_vec)
            if new_len < 1e-6:
                penalized_candidates.append((dist, node, typ))
                continue

            new_dir = new_vec / new_len
            new_high_y = min(parent_pos[1], node_pos[1])  # Higher endpoint (smaller Y)
            penalty = 1.0

            for u, v in G.edges():
                u_pos = G.nodes[u]['pos']
                v_pos = G.nodes[v]['pos']

                # Only check edges whose lower endpoint is below our higher endpoint
                edge_low_y = max(u_pos[1], v_pos[1])  # Lower endpoint (larger Y)
                if edge_low_y < new_high_y:
                    continue  # Edge is entirely above us — skip

                edge_vec = v_pos - u_pos
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-6:
                    continue
                edge_dir = edge_vec / edge_len

                # Direction similarity
                cosine = abs(np.dot(new_dir, edge_dir))
                if cosine < 0.85:  # Not parallel enough — skip
                    continue

                # Check spatial overlap: midpoint distance
                new_mid = (parent_pos + node_pos) / 2
                edge_mid = (u_pos + v_pos) / 2
                mid_dist = np.linalg.norm(new_mid - edge_mid)

                # Also check endpoint proximity
                d1 = np.linalg.norm(parent_pos - u_pos)
                d2 = np.linalg.norm(parent_pos - v_pos)
                d3 = np.linalg.norm(node_pos - u_pos)
                d4 = np.linalg.norm(node_pos - v_pos)
                min_endpoint_dist = min(d1, d2, d3, d4)

                # If edges are parallel AND spatially close, penalize
                if mid_dist < 2 * proximity_threshold or min_endpoint_dist < proximity_threshold:
                    penalty += cosine * 5

            penalized_candidates.append((dist * penalty, node, typ))
        candidates = penalized_candidates

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


def generate_trees_from_tips_topdown(G_input, n_trees=500, randomness=0.3,
                                     max_order_factor=1, verbose=True):
    """
    Generate multiple random trees from a single input graph.
    """

    max_order_input = get_max_root_order(G_input)
    max_order = (max_order_input + 1) * max_order_factor
    min_length = get_min_lateral_length(G_input)
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

        H = place_internal_nodes_only(G_input, verbose=False)

        if H is None:
            continue

        try:
            # Build tree by repeatedly calling add_one_branch
            while True:
                success = add_one_branch(H, max_order, min_length, randomness=randomness, verbose=(attempts <= 3))
                if not success:
                    break

            # Add edge weights
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
                    m, s = graph_costs(H,get_critical_nodes(H))
                    print(f"      costs=({m:.0f}, {s:.0f})")
                except Exception as e:
                    print(f"      graph_costs error: {e}")

            if tips_connected == len(tips_total):
                mactual, sactual = graph_costs(H,get_critical_nodes(H))
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


def add_node_and_edges(H, available_nodes, next_id, min_orders, shrunk_hull, min_distance, max_order):
    """
    Take two nodes from available_nodes, create a new internal node connecting them,
    and add it to H. Returns the updated H, next available ID, and updated min_orders.
    Returns (None, None, None) if no valid pairing exists due to order constraint.
    """
    base_node = 0

    # Final case: only base and one other node remain — just connect them
    if len(available_nodes) == 2 and base_node in available_nodes:
        other = [n for n in available_nodes if n != base_node][0]
        if not H.has_edge(base_node, other):
            H.add_edge(base_node, other)
        return H, next_id, min_orders

    # Pick two random nodes from available_nodes (excluding base if present)
    candidates = [n for n in available_nodes if n != base_node]

    if len(candidates) < 2:
        if len(candidates) == 1:
            if not H.has_edge(base_node, candidates[0]):
                H.add_edge(base_node, candidates[0])
            available_nodes.discard(candidates[0])
        return H, next_id, min_orders

    # Try up to 400 random pairs without repetition until we find one that satisfies order constraint
    n_pairs = min(400, len(candidates) * (len(candidates) - 1) // 2)
    all_indices = np.arange(len(candidates))
    valid_pair_found = False

    for _ in range(n_pairs):
        idx1, idx2 = np.random.choice(all_indices, size=2, replace=False)
        n1, n2 = candidates[idx1], candidates[idx2]
        o1, o2 = min_orders[n1], min_orders[n2]
        min_order_new = min(max(o1, o2 + 1), max(o2, o1 + 1))
        if min_order_new <= max_order:
            valid_pair_found = True
            break

    if not valid_pair_found:
        return None, None, None

    best_dist = np.linalg.norm(H.nodes[n1]['pos'] - H.nodes[n2]['pos'])

    # Create new internal node at the midpoint with randomness
    pos1 = H.nodes[n1]['pos']
    pos2 = H.nodes[n2]['pos']
    midpoint = (pos1 + pos2) / 2
    base_pos = H.nodes[0]['pos']
    upward_bias = 0.3
    biased_center = midpoint + upward_bias * (base_pos - midpoint)
    std = best_dist / 6
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

    # Compute the actual order for this internal node
    new_order = max(min_orders[n1], min_orders[n2]) + 1

    # Add the new internal node and edges
    H.add_node(next_id, pos=candidate, is_internal=True, root_deg=new_order)
    H.add_edge(next_id, n1)
    H.add_edge(next_id, n2)

    # Update available_nodes
    available_nodes.discard(n1)
    available_nodes.discard(n2)
    available_nodes.add(next_id)

    # Update min_orders
    min_orders[next_id] = min_order_new
    del min_orders[n1]
    del min_orders[n2]

    return H, next_id + 1, min_orders


def generate_trees_from_tips_bottomup(G_input, n_trees=500, randomness=0.3,
                                      max_order_factor=1, verbose=True):
    """
    Generate random trees bottom-up: start with tips, iteratively pair random nodes
    into a new internal node, until only base + one node remain, then connect them.
    Tracks min_possible_order for each cluster to enforce max_order constraint.
    """

    critical_nodes = get_critical_nodes(G_input)
    base_node = 0

    # Setup positions, hull, distances
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

        # Track min_possible_order for each cluster (tips start at 0)
        min_orders = {n: 0 for n in available_nodes if n != base_node}

        # Pair until only base + one other remain
        while len(available_nodes) > 2 or (len(available_nodes) == 2 and base_node not in available_nodes):
            # Early check: can this configuration possibly be completed?
            if not can_complete_within_order(list(min_orders.values()), max_order):
                print(f"    Early abort: orders={sorted(min_orders.values())}, max={max_order}")
                break
            H, next_id, min_orders = add_node_and_edges(
                H, available_nodes, next_id, min_orders,
                shrunk_hull, min_distance, max_order
            )
            if H is None:
                # Pairing failed due to order constraint
                break

        if H is None or len(available_nodes) != 2 or base_node not in available_nodes:
            continue

        # Final connection: base to the last remaining node
        H, next_id, min_orders = add_node_and_edges(
            H, available_nodes, next_id, min_orders,
            shrunk_hull, min_distance, max_order
        )

        if H is None:
            continue

        # Add edge weights
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


def can_complete_within_order(min_orders, max_order):
    orders = sorted(min_orders)

    while len(orders) > 1:
        a, b = orders[0], orders[1]

        # Option 1: Continuation (no order increase)
        cont_order = max(a, b)

        # Option 2: Lateral (one gets +1)
        if a + 1 <= b:
            lat_order = b
        else:
            lat_order = a + 1

        # Best case: choose the smaller
        new_order = min(cont_order, lat_order)

        if new_order > max_order:
            return False

        orders = sorted([new_order] + orders[2:])

    return True


def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
        G, alpha=None, min_length_factor=0.5, spacing_factor=1.0, optimization_rounds=0):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.

    For the first `optimization_rounds` iterations, uses Pareto-optimal tip selection
    via optimal_steiner_step. After that, switches to random tip selection among
    valid candidates (still respecting min_length, nude zone, and binary constraints).

    Steiner points on the existing tree are shuffled and randomly picked to make
    connections with random tips in the random phase.

    If alpha is None, a random alpha is chosen.
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

    while added_nodes < len(critical_nodes):
        # Determine mode based on iteration count
        use_random = (added_nodes - 1) >= optimization_rounds

        if use_random:
            # RANDOM MODE: shuffle unpaired nodes, pick random valid tip for each
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

                # Collect all valid tips for this u
                valid_tips = []
                for v in out_nodes:
                    p1 = H.nodes[u]["pos"]
                    p2 = G.nodes[v]["pos"]
                    edge_length = point_dist(p1, p2)

                    if edge_length < min_lateral_len:
                        continue

                    u_has_child = children_count.get(u, 0) > 0
                    if u != base_node and u_has_child:
                        axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                        if axis_total > 0 and H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
                            continue

                    valid_tips.append(v)

                if valid_tips:
                    # Pick a random valid tip
                    best_tip = valid_tips[np.random.randint(len(valid_tips))]
                    # Use random priority so best_edges order is shuffled
                    insort(best_edges, (np.random.random(), u, best_tip))
        else:
            # OPTIMAL MODE: use Pareto-optimal tip selection
            optimal_steiner_step(G, H, out_nodes, unpaired_nodes, children_count,
                                 alpha, min_lateral_len, best_edges)

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


def random_steiner_wrapper(G, n):
    """Wrapper to make the fixed-spacing Steiner function compatible with plot_random_sequence."""
    costs, trees = [], []
    for _ in range(n):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            G, alpha=alpha, optimization_rounds=0
        )
        if tree is not None:
            m, s = graph_costs(tree,get_critical_nodes(tree))
            costs.append((m, s))
            trees.append(tree)
    return costs, trees


def random_tree_normalized(G,n=1000):
    """
    Generate 1000 random spanning trees with normalized travel distances.

    Travel distance is multiplied by (c-1)/k where:
        c = number of critical nodes in the input graph G
        k = number of tips in the generated tree (excluding the root base)

    This compensates for the fact that a random tree with fewer tips than the
    original is spreading its total travel across fewer endpoints, making each
    endpoint more significant.

    Args:
        G (nx.Graph): The graph to compute the random trees for
    Returns:
        costs (list): A list of (mactual, sactual) tuples for each random tree
    """
    random.seed(a=None)
    critical_nodes_G = get_critical_nodes(G)
    c = len(critical_nodes_G)

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

        mactual, sactual = graph_costs(R, critical_nodes=critical_nodes_G)

        k = len([n for n in R.nodes() if R.degree(n) == 1 and n != 0])
        normalization_factor = (c - 1) / k if k > 0 else 1.0
        sactual_normalized = sactual * normalization_factor

        random_trees.append(R)
        costs.append((mactual, sactual_normalized))

    return costs, random_trees


def calculate_plot_buffer(min_val, max_val, buffer_percent=0.2):
    """Calculate axis limits with buffer."""
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    buffer = range_val * buffer_percent
    return min_val - buffer, max_val + buffer


def comparaplot(G, random1, random2, pareto_fun, n_random=500, dest=None):
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
    front, actual = pareto_front_generic(H,pareto_fun)

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
    front_preserve, actual_preserve = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips
    )

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

    # ---- Original RSA (first row, first column) ----

    ax = axes[0, 0]
    draw_tree_simple(ax, H, "Original RSA")

    # ---- Straightened RSA (second row, first column) ----

    ax = axes[1, 0]
    draw_tree_simple(ax, straighten_tree(H), "Straightened RSA")

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


def plot_randomization_landscape(G, n_trees=500, dest=None):
    """
    Plot trees generated with randomized alpha and optimization rounds
    using the analytical semi-random generator.
    Color = optimization level (blue = Pareto-optimal, red = fully random)
    Point size = alpha value (1 = low alpha, 4 = high alpha)
    Blue diamonds (no black perimeter) mark fully Pareto-optimal trees.
    """
    # Ensure edge weights
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            H[u][v]['weight'] = point_dist(np.array(H.nodes[u]['pos']), np.array(H.nodes[v]['pos']))

    # Compute Pareto front for reference (actual is the straightened tree)
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_analytical_preserve_tips
    )

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
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
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
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
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

        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
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

    # Plot actual tree (straightened)
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=200, linewidth=2, zorder=6,
               label='Actual Tree (straightened)', edgecolors='white')

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

        point_sizes = 10 + 70 * reg_alphas

        ax.scatter(reg_costs_x, reg_costs_y, c=colors_rgb, s=point_sizes, zorder=4,
                   edgecolors='none', alpha=0.7, label='Randomized analytical trees')

    # Plot optimal points as blue diamonds
    if opt_idx:
        opt_costs_x = [scale_data(costs[i][0]) for i in opt_idx]
        opt_costs_y = [scale_data(costs[i][1]) for i in opt_idx]
        opt_alphas = np.array([alphas_used[i] for i in opt_idx])
        opt_point_sizes = 30 + 70 * opt_alphas

        ax.scatter(opt_costs_x, opt_costs_y, marker='D', color='blue', s=opt_point_sizes,
                   zorder=7, edgecolors='none', label='Fully Pareto-optimal')

    # Colorbar for optimization rounds
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])

    n_colors = 256
    hue_values = np.linspace(0, 0.67, n_colors)  # red at 0, blue at 1
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
    all_x = front_x + [scale_data(actual[0])] + [scale_data(c[0]) for c in costs]
    all_y = front_y + [scale_data(actual[1])] + [scale_data(c[1]) for c in costs]

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
    - pareto_steiner_fast_analytical_preserve_tips_semi_random
      with optimization_rounds=0 (randomized analytical)

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
        H, pareto_steiner_fast_analytical_preserve_tips
    )

    # 1. Random spanning tree (normalized)
    print("Generating random spanning trees (normalized)...")
    costs_rand, _ = random_tree_normalized(H, n=n_random[0])
    print(f"  Got {len(costs_rand)} trees")

    # 2. Top-down
    print("Generating top-down trees...")
    costs_td, trees_td = generate_trees_from_tips_topdown(H, n_trees=n_random[1], verbose=verbose)
    print(f"  Got {len(costs_td)} trees")

    # 3. Bottom-up
    print("Generating bottom-up trees...")
    costs_bu, trees_bu = generate_trees_from_tips_bottomup(H, n_trees=n_random[2], verbose=verbose)
    print(f"  Got {len(costs_bu)} trees")

    # 4. Randomized analytical (optimization_rounds=0)
    print("Generating randomized analytical trees...")
    costs_rs = []
    for i in range(n_random[3]):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
            H, alpha=alpha, optimization_rounds=0
        )
        if i % 100 == 0:
            print(f"  Got {len(costs_rs)} randomized analytical trees")
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
    plot_cloud(costs_rs, 'v', 'purple', 'Randomized Analytical', 1)

    plot_centroid(m_rand, s_rand, 'o', 'darkgreen', 150, 'Spanning Mean', 5)
    plot_centroid(m_td, s_td, 's', 'darkblue', 120, 'Top-Down Mean', 5)
    plot_centroid(m_bu, s_bu, '^', 'darkorange', 120, 'Bottom-Up Mean', 5)
    plot_centroid(m_rs, s_rs, 'v', 'darkviolet', 120, 'Rand Analytical Mean', 5)

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


def comparaplot_two_random(
    G,
    n_random=[500, 500],
    optimization_rounds=0,
    dest=None,
    verbose=True,
):
    """
    Plot Pareto front comparing two random Steiner generators, both at
    optimization_rounds=0 (fully random):

    - pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random
      (fixed-spacing, lattice-based null model)
    - pareto_steiner_fast_analytical_preserve_tips_semi_random
      (continuous-attachment null model)

    Both generators are driven with the same shuffled alpha sequence, so the
    two clouds are paired at the population level.

    Args:
        G: Input graph
        n_random: [n_fixed, n_analytical] sample sizes
        dest: Optional path to save the plot
        verbose: Print progress

    Returns:
        fig, ax, summary dict with mean_fixed, mean_analytical, delta
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Reference Pareto front
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto
    )

    # Shared alpha stream so the two algorithms see the same alphas
    n_use = max(n_random)
    shared_alphas = np.linspace(0.0, 1.0, n_use)
    np.random.shuffle(shared_alphas)

    # 1. Fixed-spacing random (optimization_rounds)
    n_fix = n_random[0]
    if verbose:
        print(f"Generating {n_fix} fixed-spacing random trees...")
    costs_fixed = []
    for i in range(n_fix):
        tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
            H, alpha=float(shared_alphas[i]), optimization_rounds=optimization_rounds
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs_fixed.append((m, s))
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_fix}")
    if verbose:
        print(f"  Got {len(costs_fixed)} fixed-spacing trees")

    # 2. Analytical random (optimization_rounds=0)
    n_ana = n_random[1]
    if verbose:
        print(f"Generating {n_ana} analytical random trees...")
    costs_analytical = []
    for i in range(n_ana):
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
            H, alpha=float(shared_alphas[i]), optimization_rounds=optimization_rounds
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs_analytical.append((m, s))
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_ana}")
    if verbose:
        print(f"  Got {len(costs_analytical)} analytical trees")

    # Centroids
    def centroid(costs_list):
        if len(costs_list) == 0:
            return 0.0, 0.0
        return (
            float(np.mean([x[0] for x in costs_list])),
            float(np.mean([x[1] for x in costs_list])),
        )

    m_fix, s_fix = centroid(costs_fixed)
    m_ana, s_ana = centroid(costs_analytical)

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

    # Fixed-spacing cloud (green +)
    if costs_fixed:
        cfx = [scale_data(c[0]) for c in costs_fixed]
        cfy = [scale_data(c[1]) for c in costs_fixed]
        ax.scatter(cfx, cfy, marker='+', color='green', s=20, alpha=0.5,
                   linewidths=0.6, zorder=1,
                   label=f'Fixed-spacing random (n={len(cfx)})')

    # Analytical cloud (blue o, hollow)
    if costs_analytical:
        cax = [scale_data(c[0]) for c in costs_analytical]
        cay = [scale_data(c[1]) for c in costs_analytical]
        ax.scatter(cax, cay, marker='o', facecolors='none', edgecolors='blue',
                   s=22, alpha=0.5, linewidths=0.6, zorder=1,
                   label=f'Analytical random (n={len(cax)})')

    # Centroids
    ax.scatter(scale_data(m_fix), scale_data(s_fix), marker='+', color='red',
               s=180, linewidths=3.0, zorder=7,
               label='Fixed-spacing mean')
    ax.scatter(scale_data(m_ana), scale_data(s_ana), marker='o', color='cyan',
               s=160, linewidths=2.5, edgecolors='black', zorder=7,
               label='Analytical mean')

    # Arrow between the two centroids to make the gap visible
    ax.annotate(
        '',
        xy=(scale_data(m_ana), scale_data(s_ana)),
        xytext=(scale_data(m_fix), scale_data(s_fix)),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5, ls='--'),
        zorder=4,
    )
    ax.text(
        scale_data((m_fix + m_ana) / 2),
        scale_data((s_fix + s_ana) / 2),
        'gap', fontsize=10, ha='center', va='bottom',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8),
    )

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(
        f"Pareto Front: Two Fully-Random Generators "
        f"(optimization_rounds={optimization_rounds}, n={n_random[0]}/{n_random[1]})",
        fontsize=14,
    )
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Axis limits
    all_x = front_x + [scale_data(actual[0]), scale_data(m_fix), scale_data(m_ana)]
    all_y = front_y + [scale_data(actual[1]), scale_data(s_fix), scale_data(s_ana)]
    for c in costs_fixed:
        all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))
    for c in costs_analytical:
        all_x.append(scale_data(c[0])); all_y.append(scale_data(c[1]))

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

    summary = {
        "mean_fixed": (m_fix, s_fix),
        "mean_analytical": (m_ana, s_ana),
        "delta_wiring": m_ana - m_fix,
        "delta_travel": s_ana - s_fix,
        "n_fixed": len(costs_fixed),
        "n_analytical": len(costs_analytical),
    }

    if verbose:
        print("\nSummary:")
        print(f"  Fixed-spacing mean:  (w={m_fix:.2f}, t={s_fix:.2f})")
        print(f"  Analytical mean:     (w={m_ana:.2f}, t={s_ana:.2f})")
        print(f"  Delta (ana - fix):   (Δw={m_ana - m_fix:+.2f}, "
              f"Δt={s_ana - s_fix:+.2f})")

    return fig, ax, summary


def comparaplot_random_convergence(
    G,
    n_random=[500, 500],
    spacing_factor_list=(1.0, 0.5, 0.25, 0.1),
    dest=None,
    verbose=True,
):
    """
    Compare the centroids of the fixed-spacing random generator at varying
    granularities against the centroid of the analytical random generator.

    Plots only the centroids (no individual tree clouds) to show whether the
    fixed-spacing centroid converges to the analytical centroid as the
    spacing gets finer.

    Args:
        G: Input graph
        n_random: [n_fixed, n_analytical] sample sizes
        spacing_factor_list: List of spacing factors to test for the
            fixed-spacing algorithm
        dest: Optional path to save the plot
        verbose: Print progress

    Returns:
        fig, ax, summary
    """
    H = copy.deepcopy(G)
    for u, v in H.edges():
        if 'weight' not in H[u][v]:
            pos_u = H.nodes[u]['pos']
            pos_v = H.nodes[v]['pos']
            H[u][v]['weight'] = point_dist(np.array(pos_u), np.array(pos_v))

    critical_nodes = get_critical_nodes(H)

    # Reference Pareto front
    front, actual = pareto_front_generic(
        H, pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto
    )

    # Shared alpha stream (paired comparison)
    n_use = max(n_random)
    shared_alphas = np.linspace(0.0, 1.0, n_use)
    np.random.shuffle(shared_alphas)

    def centroid(costs_list):
        if len(costs_list) == 0:
            return 0.0, 0.0
        return (
            float(np.mean([x[0] for x in costs_list])),
            float(np.mean([x[1] for x in costs_list])),
        )

    # 1. Analytical random (optimization_rounds=0) -- the target
    n_ana = n_random[1]
    if verbose:
        print(f"Generating {n_ana} analytical random trees...")
    costs_analytical = []
    for i in range(n_ana):
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
            H, alpha=float(shared_alphas[i]), optimization_rounds=0
        )
        if tree is not None:
            m, s = graph_costs(tree, critical_nodes=critical_nodes)
            costs_analytical.append((m, s))
    m_ana, s_ana = centroid(costs_analytical)
    if verbose:
        print(f"  Analytical centroid: (w={m_ana:.1f}, t={s_ana:.1f})")

    # 2. Fixed-spacing random at varying granularities
    fixed_results = []
    n_fix = n_random[0]
    for spf in spacing_factor_list:
        if verbose:
            print(f"Generating {n_fix} fixed-spacing random trees "
                  f"(spacing_factor={spf})...")
        costs_fixed = []
        for i in range(n_fix):
            tree = pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto_random(
                H, alpha=float(shared_alphas[i]),
                spacing_factor=spf, optimization_rounds=0
            )
            if tree is not None:
                m, s = graph_costs(tree, critical_nodes=critical_nodes)
                costs_fixed.append((m, s))
        m_fix, s_fix = centroid(costs_fixed)
        fixed_results.append({
            "spacing_factor": spf,
            "mean": (m_fix, s_fix),
            "delta_w": m_fix - m_ana,
            "delta_t": s_fix - s_ana,
        })
        if verbose:
            print(f"  Fixed centroid: (w={m_fix:.1f}, t={s_fix:.1f})")

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

    # Pareto front
    front_x = [scale_data(x[0]) for x in front.values()]
    front_y = [scale_data(x[1]) for x in front.values()]
    ax.plot(front_x, front_y, '-', color='black', linewidth=1.5, alpha=0.5,
            label='Pareto Front', zorder=2)

    # Actual tree
    ax.scatter(scale_data(actual[0]), scale_data(actual[1]),
               marker='*', color='black', s=200, linewidth=2, zorder=5,
               label='Actual Tree', edgecolors='white')

    # Analytical target (cyan circle)
    ax.scatter(scale_data(m_ana), scale_data(s_ana), marker='o',
               color='cyan', s=180, linewidths=2.5, edgecolors='black',
               zorder=7, label='Analytical mean (target)')

    # Fixed-spacing trajectory (gradient red -> green)
    n_spf = len(fixed_results)
    colors = plt.cm.viridis(np.linspace(0.9, 0.1, n_spf))

    fix_x = [scale_data(r["mean"][0]) for r in fixed_results]
    fix_y = [scale_data(r["mean"][1]) for r in fixed_results]

    # Trajectory line
    ax.plot(fix_x, fix_y, '--', color='gray', linewidth=1.5, alpha=0.7,
            zorder=3)

    # Individual centroids
    for idx, r in enumerate(fixed_results):
        x = scale_data(r["mean"][0])
        y = scale_data(r["mean"][1])
        ax.scatter(x, y, marker='o', color=colors[idx], s=120, zorder=4,
                   edgecolors='black', linewidths=0.5,
                   label=f"spacing_factor={r['spacing_factor']}")

    # Arrow from coarsest to target
    if fixed_results:
        ax.annotate(
            '',
            xy=(scale_data(m_ana), scale_data(s_ana)),
            xytext=(fix_x[0], fix_y[0]),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2, ls=':'),
            zorder=3,
        )

    ax.set_xlabel(f"Total length ({scale_unit})", fontsize=13)
    ax.set_ylabel(f"Travel distance ({scale_unit})", fontsize=13)
    ax.set_title(
        "Convergence of Fixed-Spacing Random to Analytical Random\n"
        f"(optimization_rounds=0, n={n_random[0]}/{n_random[1]})",
        fontsize=14,
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Focus view on the centroid cluster
    if fix_x and fix_y:
        min_cx = min(min(fix_x), scale_data(m_ana))
        max_cx = max(max(fix_x), scale_data(m_ana))
        min_cy = min(min(fix_y), scale_data(s_ana))
        max_cy = max(max(fix_y), scale_data(s_ana))

        x_margin = (max_cx - min_cx) * 0.2 if max_cx > min_cx else 1000
        y_margin = (max_cy - min_cy) * 0.2 if max_cy > min_cy else 1000

        ax.set_xlim(min_cx - x_margin, max_cx + x_margin)
        ax.set_ylim(min_cy - y_margin, max_cy + y_margin)
    else:
        all_x = front_x + [scale_data(actual[0]), scale_data(m_ana)] + fix_x
        all_y = front_y + [scale_data(actual[1]), scale_data(s_ana)] + fix_y
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

    return fig, ax, fixed_results


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

    # Gray-to-black gradient: spf closest to 0 = black, furthest = light gray
    n_fixed = len(spacing_list)
    gray_scale = []
    for i in range(n_fixed):
        t = i / max(n_fixed - 1, 1)
        # Black (0,0,0) -> light gray (0.7,0.7,0.7)
        c = 0.6 * t # at t=0: 0.7 (light gray), at t=1: 0 (black)
        gray_scale.append((c, c, c))

    # Plot fixed-spacing fronts with alternating line styles
    for idx, spf in enumerate(spacing_list):
        front, _ = fronts_fixed[spf]
        fx = [scale_data(x[0]) for x in front.values()]
        fy = [scale_data(x[1]) for x in front.values()]
        linestyle = line_styles[idx % len(line_styles)]
        ax.plot(fx, fy, marker='', linestyle=linestyle, color=gray_scale[idx],
                linewidth=2.0, label=f'Fixed spf={spf}', zorder=2, alpha=0.9)

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


def pareto_steiner_fast_preserve_tips(G, alpha, min_length_factor=0.5, spacing_factor=1.0):
    """
    Original greedy Steiner algorithm with:
    - Tip preservation (tips not used as intermediate nodes)
    - Binary tree constraint (max 2 children, base max 1)
    - Minimum length from tips and branch points for Steiner point placement
    """
    assert 0 <= alpha <= 1

    base_node = 0
    critical_nodes = get_critical_nodes(G)
    min_length = get_min_lateral_length(G, min_length_factor)

    H = nx.Graph()
    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    base_pos = G.nodes[base_node]["pos"]
    H.nodes[base_node]["pos"] = base_pos

    in_nodes = set([base_node])
    out_nodes = set(critical_nodes)
    out_nodes.remove(base_node)

    children_count = {base_node: 0}

    closest_neighbors = {}
    for u in critical_nodes:
        closest_neighbors[u] = k_nearest_neighbors(
            G, u, k=None, candidate_nodes=critical_nodes[:]
        )

    unpaired_nodes = set([base_node])
    node_index = max(critical_nodes) + 1
    best_edges = []
    added_nodes = 1

    while added_nodes < len(critical_nodes):
        assert len(out_nodes) > 0

        for u in unpaired_nodes:
            assert H.has_node(u)
            assert "distance_to_base" in H.nodes[u]

            # --- NEW: binary constraint ---
            if u != base_node and children_count.get(u, 0) >= 2:
                continue
            if u == base_node and children_count.get(u, 0) >= 1:
                continue
            # --- NEW: tip preservation ---
            if H.nodes[u].get("is_tip", False):
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

            for invalid_neighbor in invalid_neighbors:
                closest_neighbors[u].remove(invalid_neighbor)

            assert closest_neighbor != None
            assert not H.has_node(closest_neighbor)

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

        cost, u, v = best_edges.pop(0)

        best_edges2 = []
        unpaired_nodes = set([u, v])
        for cost, x, y in best_edges:
            if y == v:
                unpaired_nodes.add(x)
            else:
                best_edges2.append((cost, x, y))
        best_edges = best_edges2

        assert H.has_node(u)
        assert not H.has_node(v)
        H.add_node(v)
        H.nodes[v]["pos"] = G.nodes[v]["pos"]
        H.nodes[v]["is_tip"] = True  # --- NEW ---
        in_nodes.add(v)
        out_nodes.remove(v)

        children_count[u] = children_count.get(u, 0) + 1  # --- NEW ---
        children_count[v] = 0  # --- NEW ---

        p1 = H.nodes[u]["pos"]
        p2 = H.nodes[v]["pos"]
        midpoints = steiner_points(p1, p2, npoints=STEINER_MIDPOINTS)
        midpoint_nodes = []

        for midpoint in midpoints:
            # --- NEW: min_length from u and v ---
            if point_dist(p1, midpoint) < min_length:
                continue
            if point_dist(midpoint, p2) < min_length:
                continue

            midpoint_node = node_index
            node_index += 1
            H.add_node(midpoint_node)
            H.nodes[midpoint_node]["pos"] = midpoint
            children_count[midpoint_node] = 1  # --- NEW ---

            neighbors = []
            for out_node in out_nodes:
                out_coord = G.nodes[out_node]["pos"]
                dist = point_dist(midpoint, out_coord)
                neighbors.append((dist, out_node))

            neighbors = sorted(neighbors)
            closest_neighbors[midpoint_node] = []
            for dist, neighbor in neighbors:
                closest_neighbors[midpoint_node].append(neighbor)

            midpoint_nodes.append(midpoint_node)
            unpaired_nodes.add(midpoint_node)

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

def optimal_steiner_step(G, H, out_nodes, unpaired_nodes, children_count, alpha,
                         min_lateral_len, best_edges):
    """
    Perform one optimal Steiner step: for each unpaired node in H, find the
    Pareto-optimal tip to connect to among remaining out_nodes.

    Populates best_edges with the optimal (cost, u, v) for each unpaired node.

    Args:
        G: Original graph (for tip positions)
        H: Current tree under construction
        out_nodes: Set of tips not yet connected
        unpaired_nodes: Set of nodes in H that need a new best edge computed
        children_count: Dict tracking number of children per node
        alpha: Pareto weight
        min_lateral_len: Minimum lateral length constraint
        best_edges: List to populate with (cost, u, v) tuples
    """
    for u in unpaired_nodes:
        if H.nodes[u].get("is_tip", False):
            continue
        if not H.has_node(u):
            continue
        if u != 0 and children_count.get(u, 0) >= 2:
            continue
        if u == 0 and children_count.get(u, 0) >= 1:
            continue

        best_tip = None
        best_cost = float('inf')

        for v in out_nodes:
            p1 = H.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            edge_length = point_dist(p1, p2)

            # Quick rejection: minimum length
            if edge_length < min_lateral_len:
                continue

            # Quick rejection: nude zone check
            u_has_child = children_count.get(u, 0) > 0
            if u != 0 and u_has_child:
                axis_total = H.nodes[u]["dist_from_axis_start"] + H.nodes[u]["dist_to_axis_tip"]
                if axis_total > 0 and H.nodes[u]["dist_to_axis_tip"] < min_lateral_len:
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


def sample_region(
    G,
    front,
    actual,
    n_samples=5000,
    region_factor=0.5,
    html_path=None,
    scale_factor=1.0,
    unit="",
):
    """Sample semi-random trees and classify an actual RSA.

    This is the research-facing wrapper around the UI sampling routine. All
    computations happen before the optional HTML output.
    """
    result = estimate_rounds_for_region(
        G.to_undirected(),
        n_samples=n_samples,
        region_factor=region_factor,
        plot=False,
        show=False,
        front=front,
        actual=actual,
        scale_factor=scale_factor,
        collect_trees=html_path is not None,
        verbose=False,
    )
    if html_path is not None:
        save_sampling_html(
            result,
            html_path,
            scale_factor=scale_factor,
            region_factor=region_factor,
            unit=unit,
        )
    return result


# Moved from pareto_functions.py: deterministic analytical greedy generator.
def pareto_steiner_fast_analytical_preserve_tips(G, alpha, min_length_factor=0.5,
                                                 spacing_factor=1.0, verbose=False):
    """
    Continuous Pareto Steiner tree that preserves tip integrity.
    Uses analytical solution for optimal attachment point.
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

    # Step 1: Connect base to the closest tip that satisfies the basal
    # min_length constraint.  If the globally closest tip is too close to the
    # base, fall back to the next valid tip instead of aborting.
    base_pos = H.nodes[base_node]["pos"]
    valid_first_tips = [
        t for t in out_nodes
        if point_dist(G.nodes[t]['pos'], base_pos) >= min_length
    ]
    if not valid_first_tips:
        return None

    closest_tip = min(
        valid_first_tips,
        key=lambda t: point_dist(G.nodes[t]['pos'], base_pos),
    )

    tip_pos = G.nodes[closest_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

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
                if point is None:
                    continue

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

    if out_nodes:
        return None

    return H


def random_tree_costs(G):
    """Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017.

    This preserves the original cost-only random spanning-tree null model.
    """
    random.seed(a=None)
    random_trees = []
    costs = []

    for _ in range(1000):
        R = nx.Graph()
        candidate_nodes = get_critical_nodes(G)

        while len(candidate_nodes) > 0:
            index = random.randrange(len(candidate_nodes))
            node = candidate_nodes[index]

            if len(R.nodes) > 0:
                random_index = random.randrange(len(R.nodes))
                parent = list(R.nodes)[random_index]
                R.add_node(node, pos=G.nodes[node]["pos"])
                R.add_edge(parent, node, weight=node_dist(R, parent, node))
            else:
                R.add_node(node, pos=G.nodes[node]["pos"])

            del candidate_nodes[index]

        random_trees.append(R)

    for R in random_trees:
        mactual, sactual = graph_costs(R)
        costs.append((mactual, sactual))

    return costs


def plot_graph(
    G,
    node_size_factor=20,
    node_base_size=2,
    edge_width_factor=1.0,
    base_node_color="#2E8B57",
    secondary_node_color="#A0522D",
    edge_color="#3D2B1F",
    critical_node_color="#FF0000",
    with_labels=False,
    title="Root System Graph",
    figsize=(10, 15),
    show_grid=True,
    save_path=None,
):  # pragma: no cover
    """Plots a root-system graph using node positions from their attributes, with axes and optional grid.

    Args:
        G (networkx.DiGraph): The graph to be plotted. Must have node positions stored
            as attributes. Assumes the graph is directed.
        node_size_factor (float, optional): Multiplier for node sizes based on degree.
            Defaults to 20.
        node_base_size (int, optional): Base size for all nodes.
            Defaults to 2.
        edge_width_factor (float, optional): Multiplier for edge widths.
            Defaults to 1.0.
        base_node_color (str, optional): Color for the base node (node 0).
            Defaults to "#2E8B57".
        secondary_node_color (str, optional): Color for all other nodes.
            Defaults to "#A0522D".
        edge_color (str, optional): Color of the edges. Defaults to
            "#3D2B1F".
        critical_node_color (str, optional): Color for nodes with degree == 1 (critical nodes).
            Defaults to "#FF0000".
        with_labels (bool, optional): Whether to display labels on nodes.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to "Root System Graph".
        figsize (tuple, optional): Size of the plot figure (width, height).
            Defaults to (10, 15).
        show_grid (bool, optional): Whether to display a grid. Defaults to True.
        save_path (str, optional): The file path to save the plot. Defaults to None.

    Raises:
        ValueError: If any node is missing a "pos" attribute.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            The matplotlib figure and axes objects for further customization.
    """
    # Extract positions from the nodes' "pos" attribute
    try:
        pos = nx.get_node_attributes(G, "pos")
        if len(pos) != len(G.nodes()):
            raise ValueError("Not all nodes have a 'pos' attribute.")
    except KeyError:
        raise ValueError("Nodes must have a 'pos' attribute for plotting.")

    # Define the base node explicitly as node 0
    base_node = 0
    if base_node not in G.nodes():
        raise ValueError("Node 0 (base node) is not present in the graph.")

    # Node sizes
    node_sizes = [node_base_size + G.degree(n) * node_size_factor for n in G.nodes()]

    # Assign node colors based on conditions
    node_colors = []
    for n in G.nodes():
        if n == base_node:
            node_colors.append(base_node_color)  # Base node
        elif G.degree(n) == 1:
            node_colors.append(critical_node_color)  # critical nodes
        else:
            node_colors.append(secondary_node_color)  # Other nodes

    # Edge widths   (changed so that they are minimal for better visualization)
    edge_widths = [edge_width_factor for u, v in G.edges()]

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the graph on the given axes
    nx.draw(
        G,
        pos,
        ax,
        with_labels=with_labels,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color=edge_color,
        width=edge_widths,
        arrows=True,  # Show direction since the graph is directed
    )

    # Invert the y-axis to match the coordinate system
    ax.invert_yaxis()

    # Add x and y axes to plot
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Set grid and axis labels
    if show_grid:
        ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)

    # Add title
    ax.set_title(title, fontsize=14)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="w")
        print(f"Plot saved to {save_path}")

    return fig, ax


# Moved from pareto_functions.py: Given a graph G and a value 0 <= alpha <= 1, compute the Pareto-optimal tree connecting.
def pareto_steiner_fast(G, alpha, min_length_factor=1.0,spacing_factor=1.0):
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
        midpoints = steiner_points(p1, p2, npoints=STEINER_MIDPOINTS)
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


# Moved from pareto_functions.py: Given two points p1 and p2, return evenly spaced intermediate points.
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


# Moved from pareto_functions.py: Fixed-spacing Steiner tree with tip preservation and binary tree constraint..
def pareto_steiner_fast_preserve_tips_fixed_spacing_binary_pareto(
        G, alpha, min_length_factor=1 / 2, spacing_factor=1.0, verbose=False):
    """
    Fixed-spacing Steiner tree with tip preservation and binary tree constraint.

    After each edge split, Steiner points within min_lateral_len of the new
    branch point u are removed from unpaired_nodes (both upstream and downstream).
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

            # Remove candidate edges involving this tip
            best_edges2 = []
            for c, x, y in best_edges:
                if y == v:
                    unpaired_nodes.add(x)
                else:
                    best_edges2.append((c, x, y))
            best_edges = best_edges2

            # Add tip node
            H.add_node(v)
            H.nodes[v]["pos"] = G.nodes[v]["pos"]
            H.nodes[v]["is_tip"] = True
            out_nodes.remove(v)

            children_count[u] = children_count.get(u, 0) + 1
            children_count[v] = 0

            # Create Steiner points along u -> v
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

            # Build path: u -> (Steiner points) -> v
            line_nodes = [u] + midpoint_nodes + [v]
            for i in range(len(line_nodes) - 1):
                n1 = line_nodes[i]
                n2 = line_nodes[i + 1]
                H.add_edge(n1, n2)
                H[n1][n2]["weight"] = node_dist(H, n1, n2)
                H.nodes[n2]["distance_to_base"] = (
                        H[n1][n2]["weight"] + H.nodes[n1]["distance_to_base"]
                )

            # Set axis metadata
            if chosen_option == 'A':
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = (
                                H.nodes[u]["dist_from_axis_start"] + dist_to_u
                        )
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)
            else:
                for node in line_nodes:
                    if node != u:
                        H.nodes[node]["axis_id"] = f"axis_{v}"
                        dist_to_u = node_dist(H, node, u)
                        H.nodes[node]["dist_from_axis_start"] = dist_to_u
                        H.nodes[node]["dist_to_axis_tip"] = node_dist(H, node, v)

            # Remove nodes too close to the new branch point u
            # Walk upstream
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

            # Walk downstream along each child of u
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

        if not edge_accepted:
            if verbose:
                print(f"  Step {iteration}: no edge accepted, breaking")
            break

    return H


def pareto_front_generic(
    G, steiner_function, semi_random=False, tree_payloads=None
):
    """
    Compute Pareto front using any steiner function.

    Args:
        G: Input graph
        steiner_function: Function that takes (G, alpha) and returns a tree.
            When ``semi_random`` is True it must also accept the
            ``optimization_rounds`` keyword.
        semi_random: If True, evaluate one fully optimized tree per alpha and
            four randomized trees (uniform alpha and 80-100% of the available
            optimization rounds) to refine the front. Trees that do not contain
            every original critical node are discarded.
        tree_payloads: Optional mutable dict. When provided, it is filled with
            compact geometry payloads keyed by the same alpha keys as ``front``
            so HTML reports can draw a tree for each front point.

    Returns:
        front: dict mapping alpha -> [wiring, travel]
        actual: tuple (wiring, travel) for the straightened tree
    """
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = point_dist(G.nodes[u]['pos'], G.nodes[v]['pos'])

    critical_nodes = get_critical_nodes(G)

    # Use straightened tree as the reference "actual" point
    straight = straighten_tree(G)
    actual = graph_costs(straight, critical_nodes=critical_nodes)

    front = {}

    required_nodes = set(critical_nodes)
    serialize_tree = _tree_hover_payload if tree_payloads is not None else None

    def add_tree(key, tree):
        if tree is None:
            return
        if semi_random and not required_nodes.issubset(tree.nodes):
            return
        total_root_length, total_travel_distance = graph_costs(
            tree, critical_nodes=critical_nodes
        )
        front[key] = [total_root_length, total_travel_distance]
        if serialize_tree is not None:
            tree_payloads[key] = serialize_tree(tree, None, None)

    for alpha in DEFAULT_ALPHAS:
        if semi_random:
            H = steiner_function(
                G, alpha, optimization_rounds=len(critical_nodes)
            )
            add_tree(alpha, H)

            # A few randomized runs per alpha catch points that the greedy
            # construction misses. Most decisions remain greedy (80-100% of
            # the available rounds); only the tail of the construction is
            # randomized.
            for _ in range(4):
                lambda_value = np.random.uniform(0.8, 1.0)
                alpharandom = np.random.uniform(0, 1.0)
                H_rand = steiner_function(
                    G,
                    alpharandom,
                    optimization_rounds=lambda_value * len(critical_nodes),
                )
                add_tree(alpharandom, H_rand)
        else:
            H = steiner_function(G, alpha)
            add_tree(alpha, H)

    return clean_pareto_front(front), actual



# Moved from pareto_functions.py: Compute the hypervolume (area) dominated by the Pareto front relative to a reference point..
def hypervolume(front, ref_point=None):
    """
    Compute the hypervolume (area) dominated by the Pareto front relative to a reference point.

    The reference point should be worse than any front point in both dimensions.
    If not provided, uses (max_wiring * 1.1, max_travel * 1.1) from the front.

    Args:
        front: dict mapping alpha -> [wiring, travel]
        ref_point: (wiring_ref, travel_ref) reference point

    Returns:
        hypervolume: float (larger = better front)
    """
    points = sorted(front.values(), key=lambda p: p[0])

    if ref_point is None:
        max_w = max(p[0] for p in points)
        max_t = max(p[1] for p in points)
        ref_point = (max_w * 1.1, max_t * 1.1)

    ref_w, ref_t = ref_point
    hv = 0.0

    for i, (w_i, t_i) in enumerate(points):
        # Height of this rectangle: from t_i up to ref_t, but limited by previous points
        height = ref_t - t_i
        # Width: from w_i to next point's w (or ref_w for last)
        if i < len(points) - 1:
            width = points[i + 1][0] - w_i
        else:
            width = ref_w - w_i

        hv += width * height

    return hv


# Moved from pareto_functions.py: Compute the ratio of hypervolumes of two Pareto fronts..
def compare_front_hypervolumes(front_1, front_2, ref_point=None):
    """
    Compute the ratio of hypervolumes of two Pareto fronts.

    ratio = hypervolume(front_1) / hypervolume(front_2)
    > 1 means front_1 dominates more area (better optimization).

    If ref_point is not provided, uses:
        wiring_ref = max(wiring_satellite from both fronts,
                         max wiring of either front)
        travel_ref = max(max travel of front_1, max travel of front_2)

    Args:
        front_1, front_2: dicts mapping alpha -> [wiring, travel]
        ref_point: Optional (wiring_ref, travel_ref)

    Returns:
        ratio: float
        hv_1, hv_2: individual hypervolumes
    """
    if ref_point is None:
        max_w = max(max(p[0] for p in front_1.values()),
                    max(p[0] for p in front_2.values()))
        max_t = max(max(p[1] for p in front_1.values()),
                    max(p[1] for p in front_2.values()))
        ref_point = (max_w, max_t)

    hv_1 = hypervolume(front_1, ref_point)
    hv_2 = hypervolume(front_2, ref_point)

    if hv_2 == 0:
        return float('inf') if hv_1 > 0 else 1.0, hv_1, hv_2

    return hv_1 / hv_2, hv_1, hv_2
