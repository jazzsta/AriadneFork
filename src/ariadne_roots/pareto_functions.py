import networkx as nx
import numpy as np
import random

from bisect import insort

STEINER_MIDPOINTS=10

DEFAULT_ALPHAS = np.arange(0, 1.01, 0.01)
DEFAULT_BETAS = np.arange(0, 1.01, 0.01)


def get_critical_nodes(G):
    """Given a graph G, return a list of its critical nodes (the main root base, which is the first node
    with index 0; and all root tips, which are nodes with degree 1).
    """
    critical_nodes = []
    for i in G.nodes():
        if i == 0 or G.degree(i) == 1:  # assumes root base node is 0
            critical_nodes.append(i)

    return critical_nodes


def straighten_tree(G):
    """
    Convert a digitized RSA to an abstract version with straight edges.

    Preserves the topology (parent-child relationships) and the positions of
    all branching nodes, tips, and the root base. Intermediate nodes along
    paths between branch points are removed. All edges become straight lines
    with Euclidean distances as weights.

    Args:
        G: NetworkX graph (directed or undirected) representing the digitized RSA

    Returns:
        H: NetworkX graph with same type, same branching nodes/tips/base,
           but straight edges between them
    """
    if G.is_directed():
        H = nx.DiGraph()
    else:
        H = nx.Graph()

    # Identify nodes to keep: base (0), tips (degree 1), and branch points (degree >= 3)
    nodes_to_keep = set()
    for node in G.nodes():
        deg = G.degree(node)
        if node == 0 or deg == 1 or deg >= 3:
            nodes_to_keep.add(node)

    # Copy kept nodes with their attributes
    for node in nodes_to_keep:
        H.add_node(node, **G.nodes[node])

    # Find paths between kept nodes in the original tree and add straight edges
    for u in nodes_to_keep:
        for v in nodes_to_keep:
            if u >= v:
                continue
            try:
                path = nx.shortest_path(G, source=u, target=v)
                # Only connect if no other kept nodes are in between
                interior = set(path[1:-1])
                if not interior.intersection(nodes_to_keep):
                    # Direct connection with Euclidean distance
                    pos_u = G.nodes[u]['pos']
                    pos_v = G.nodes[v]['pos']
                    weight = point_dist(pos_u, pos_v)
                    H.add_edge(u, v, weight=weight)
            except (nx.NetworkXNoPath, KeyError):
                continue

    # Copy graph-level attributes
    H.graph.update(G.graph)

    return H

def graph_costs(G, critical_nodes=None):
    """Use BFS to compute the wiring cost, conduction delay of a graph G.

    Args:
        G (nx.Graph): The graph to compute the costs for
        critical_nodes (list): The list of critical nodes to consider. If None, all
            nodes are considered.

    Returns:
        total_root_length (float): The wiring cost of the graph. Wiring cost is the total length of
            the edges in the network.
        total_travel_distance (float): The conduction delay of the graph. Conduction delay is the sum of
            the distances from each point to the root. By default, computes conduction
            delay for all nodes. If you specify a set of critical nodes, then only those
            nodes are used for computing conduction delay.
    """
    # initialize costs
    total_root_length = 0
    total_travel_distance = 0

    # dictionary that stores each node's distance to the base_node
    distance_to_base = {}
    # this method assumes node 0 is the base_node
    base_node = 0
    # node 1 has distance 0 from the base_node
    distance_to_base[base_node] = 0

    # dictionary that stores each node's parent_node in the bfs
    # this way we avoid visiting the same node twice
    parent_node = {}
    parent_node[base_node] = None

    # nodes_to_visit: nodes that have been discovered but not yet visited
    nodes_to_visit = [base_node]
    visited_nodes = set()

    # lists that store the edge lengths and the distances from the nodes to each base_node
    edge_lengths = []
    travel_distances_to_base = []
    while len(nodes_to_visit) > 0:
        # visit the next discovered but not visited node
        current_node = nodes_to_visit.pop(0)

        # if we are trying to  visit an already-visited node, => we have a cycle
        if current_node in visited_nodes:
            return float("inf"), float("inf")

        # we've visited current_node
        visited_nodes.add(current_node)

        # go through current_node's children and add the unvisited nodes to the nodes_to_visit
        for child_node in G.neighbors(current_node):
            # ignore current_node's parent_node, this was already visited in the bfs
            if child_node != parent_node[current_node]:
                edge_length = G[current_node][child_node]["weight"]
                edge_lengths.append(edge_length)

                # to get to the base_node, the child_node must go to current_node and then to the base_node
                # thus, child_node's distance_to_base = distance from child_node to current_node + distance from current_node to base
                child_distance_to_base = edge_length + distance_to_base[current_node]
                distance_to_base[child_node] = child_distance_to_base

                # if we have specified a set of critical nodes, only those nodes contribute to conduction delay
                if critical_nodes == None or child_node in critical_nodes:
                    travel_distances_to_base.append(child_distance_to_base)
                parent_node[child_node] = current_node
                nodes_to_visit.append(child_node)

    # if not every node was visited, => graph is not connected
    assert len(visited_nodes) == G.number_of_nodes()

    total_root_length = sum(sorted(edge_lengths))
    total_travel_distance = sum(sorted(travel_distances_to_base))

    return total_root_length, total_travel_distance


def graph_costs_3d_path_tortuosity(G, critical_nodes=None):
    """Use BFS to compute the wiring cost, conduction delay and total path coverage of a graph G.

    Args:
        G (nx.Graph): The graph to compute the costs for
        critical_nodes (list): The list of critical nodes to consider. If None, all
            nodes are considered.

    Returns:
        total_root_length (float): The wiring cost of the graph. Wiring cost is the total length of
            the edges in the network.
        total_travel_distance (float): The conduction delay of the graph. Conduction delay is the sum of
            the distances from each point to the base node. By default, computes conduction
            delay for all nodes. If you specify a set of critical nodes, then only those
            nodes are used for computing conduction delay.
        total_path_coverage: the sum of the tortuosity of all the root paths. The tortuosity per
            path is defined as the ratio of the actual path length to the shortest path
            length between the base node and the root tip. The total path coverage is the sum of
            the tortuosity of all the root paths.
    """
    # initialize costs
    total_root_length = 0
    total_travel_distance = 0
    total_path_coverage = 0

    # dictionary that stores each node's distance to the base_node
    distance_to_base = {}
    # this method assumes node 0 is the base_node
    base_node = 0
    # base node has distance 0 from the base_node
    distance_to_base[base_node] = 0
    # position of the base node
    base_pos = G.nodes[base_node]["pos"]

    # dictionary that stores each node's parent_node in the bfs
    # this way we avoid visiting the same node twice
    parent_node = {}
    parent_node[base_node] = None

    # nodes_to_visit: nodes that have been discovered but not yet visited
    nodes_to_visit = [base_node]
    visited_nodes = set()

    # lists that store the edge lengths, the distances from the nodes to each base_node,
    # and the straight distances from the nodes to the base_node
    edge_lengths = []
    travel_distances_to_base = []
    straight_distances_to_base = []
    while len(nodes_to_visit) > 0:
        # visit the next discovered but not visited node
        current_node = nodes_to_visit.pop(0)

        # if we are trying to  visit an already-visited node, => we have a cycle
        if current_node in visited_nodes:
            return float("inf"), float("inf"), float("inf")

        # we've visited current_node
        visited_nodes.add(current_node)

        # go through current_node's children and add the unvisited nodes to the nodes_to_visit
        for child_node in G.neighbors(current_node):
            # ignore current_node's parent_node, this was already visited in the bfs
            if child_node != parent_node[current_node]:
                edge_length = G[current_node][child_node]["weight"]
                edge_lengths.append(edge_length)

                # to get to the base_node, the child_node must go to current_node and then to the base_node
                # thus, child_node's distance_to_base = distance from child_node to current_node + distance from current_node to base
                child_distance_to_base = edge_length + distance_to_base[current_node]
                distance_to_base[child_node] = child_distance_to_base

                # if we have specified a set of critical nodes, only those nodes contribute to conduction delay
                if critical_nodes == None or child_node in critical_nodes:
                    travel_distances_to_base.append(child_distance_to_base)
                    # position of the critical node
                    child_pos = G.nodes[child_node]["pos"]
                    # straight distance from the critical node to the base node
                    straight_distance_to_base = point_dist(base_pos, child_pos)
                    straight_distances_to_base.append(straight_distance_to_base)
                parent_node[child_node] = current_node
                nodes_to_visit.append(child_node)

    # if not every node was visited, => graph is not connected
    assert len(visited_nodes) == G.number_of_nodes()

    # compute the total path coverage
    for i in range(len(travel_distances_to_base)):
        # Handle coincident nodes: if straight distance is 0, tortuosity ratio is 1.0
        # This occurs when a critical node is at the same position as the base node
        if straight_distances_to_base[i] == 0:
            total_path_coverage += 1.0
        else:
            total_path_coverage += (
                travel_distances_to_base[i] / straight_distances_to_base[i]
            )

    total_root_length = sum(sorted(edge_lengths))
    total_travel_distance = sum(sorted(travel_distances_to_base))

    return total_root_length, total_travel_distance, total_path_coverage


def slope_vector(p1, p2):
    """Given two n-dimensional points, computes the slope m between p1 and p2

    We pick the slope specifically such that p1 = p0 + 1 * m

    This way, for 0 <= t <= 1, p0 + m*t gives us a point on the line segment from p0 to p1

    this is as simple as m = p1 - p0
    """
    # points must have the same dimension
    assert len(p1) == len(p2)

    slope_vec = []
    for i in range(len(p1)):
        slope_vec.append(p2[i] - p1[i])
    return slope_vec


def delta_point(p1, slope, t):
    """Given a point p0 and a slope vector, computes p0 + t * slope

    works for any dimension
    """
    p2 = []
    assert len(p1) == len(slope)
    for i in range(len(p1)):
        p2.append(p1[i] + t * slope[i])
    return p2


def steiner_points(p1, p2, npoints=10):
    """Given two points p1 and p2, this method divides the p1-p2 line segment into several
    line segments.

    This method returns all of the intermediate points along the p1-p2 line

    npoints specifies how many intermediate points to create; the bigger this is, the
    more finely we divide the line segment
    """
    # get the slope such that p2 = p1 + m*t
    # for 0 <= t <= 1 this gives us a point along the p1-p2 line segment
    slope = slope_vector(p1, p2)

    # the more points, the smaller the gap between points
    delta = 1.0 / (npoints + 1)

    # we will vary t between 0 and 1 to get different points along the line segment
    t = delta
    midpoints = []
    while t < 1:
        p3 = delta_point(p1, slope, t)
        midpoints.append(tuple(p3))
        t += delta

    return midpoints


def pareto_cost(total_root_length, total_travel_distance, alpha):
    """Given a wiring cost, a conduction delay, and alpha, computes the overal pareto cost

    alpha tells us   how much to prioritize each objective. If it's 0, we only care about
    conduction delay. If it's 1, we only care about wiring cost

    total_root_length: the wiring cost (a.k.a. the material cost)
    total_travel_distance: the conduction delay (a.k.a. the satellite cost)
    """
    assert 0 <= alpha <= 1

    total_root_length *= alpha
    total_travel_distance *= 1 - alpha
    cost = total_root_length + total_travel_distance
    return cost


def pareto_cost_3d_path_tortuosity(
    total_root_length, total_travel_distance, total_path_coverage, alpha, beta
):
    """Computes the pareto cost.

    alpha * total_root_length + beta * total_travel_distance - gamma * total_path_coverage

    alpha + beta + gamma = 1

    When alpha = beta = 0, gamma = 1 => cost = -total_path_coverage will be minimized =>
        total_path_coverage will be maximized
    When alpha = gamma = 0, beta = 1 => cost = total_travel_distance will be minimized
    When beta = gamma = 0, alpha = 1 => cost = total_root_length will be minimized

    total_root_length: the sum of the lengths of the edges in the root network
        (a.k.a. material cost, wiring cost)
    total_travel_distance: the sum of the lengths of the shortest paths from every
        lateral root tip to the base node of the network. (a.k.a. the satellite cost,
        conduction delay)
    total_path_coverage: the sum of the tortuosity of all the root paths. The tortuosity per
        path is defined as the ratio of the actual path length to the shortest path
        length between the root and the root tip. The total path coverage is the sum of
        the tortuosity of all the root paths.
    """
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert alpha + beta <= 1, "alpha + beta must be <= 1 (gamma cannot be negative)"

    gamma = 1 - alpha - beta
    cost = (
        alpha * total_root_length
        + beta * total_travel_distance
        - gamma * total_path_coverage
    )

    return cost


def point_dist(p1, p2):
    """Euclidean distance between two different points (of any dimension)"""
    assert len(p1) == len(p2)
    sq_dist = 0
    for i in range(len(p1)):
        x1, x2 = p1[i], p2[i]
        sq_dist += (x1 - x2) ** 2
    return sq_dist**0.5


def node_dist(G, u, v):
    """Gets the euclidean distance between two different nodes in a graph

    Assumes each node has an attribute 'pos' corresponding to its coordinate
    """
    p1 = G.nodes[u]["pos"]
    p2 = G.nodes[v]["pos"]
    return point_dist(p1, p2)


def k_nearest_neighbors(G, u, k=None, candidate_nodes=None):
    """Given a graph G and a node u, this method gets u's closest neighbors in G

    The closest neighbors are determined using the euclidean distance between the nodes

    if candidate_nodes is given, the method only considers the closest neighbor out of the candida
    """
    if candidate_nodes == None:
        candidate_nodes = list(G.nodes())

    if u in candidate_nodes:
        candidate_nodes.remove(u)

    nearest_neighbors = sorted(candidate_nodes, key=lambda v: node_dist(G, u, v))
    if k != None:
        assert type(k) == int
        nearest_neighbors = nearest_neighbors[:k]
    return nearest_neighbors


def satellite_tree(G):
    """Constructs the satellite tree out of G; this is a graph in which every node is connected
    to the root base by a direct line
    """
    # assume the _node is node 0
    base_node = 0

    H = nx.Graph()

    H.add_node(base_node)
    H.nodes[base_node]["distance_to_base"] = 0
    base_pos = G.nodes[base_node]["pos"]
    H.nodes[base_node]["pos"] = base_pos

    critical_nodes = get_critical_nodes(G)

    # connect every critical node to the base_node with a direct edge
    for u in critical_nodes:
        if u == base_node:
            continue
        H.add_edge(base_node, u)
        H.nodes[u]["pos"] = G.nodes[u]["pos"]
        H[base_node][u]["weight"] = node_dist(G, base_node, u)

    return H





def get_min_lateral_length(G, factor=0.5):
    base_node = 0
    nude_lengths = []

    for node in G.nodes():
        if node == base_node:
            continue
        if G.degree(node) == 1:
            total_length = 0
            current = node
            prev = None

            while current != base_node:
                # Find the edge incident to current that doesn't go to prev
                next_node = None
                for u, v in G.edges(current):
                    neighbor = v if u == current else u
                    if neighbor != prev:
                        next_node = neighbor
                        break

                if next_node is None:
                    break

                # Get edge weight
                if 'weight' in G[current][next_node]:
                    total_length += G[current][next_node]['weight']
                else:
                    pos_current = np.array(G.nodes[current]['pos'])
                    pos_next = np.array(G.nodes[next_node]['pos'])
                    total_length += point_dist(pos_current, pos_next)

                if G.degree(next_node) > 2:
                    break

                prev = current
                current = next_node

            nude_lengths.append(total_length)

    if not nude_lengths:
        return 0.0

    return min(nude_lengths) * factor




def sample_random_attachment(G, H, out_nodes, min_length):
    """
    Sample a uniformly random valid attachment (edge, tip, lambda, point).
    Respects the exact same min_length and nude zone constraints as the
    fixed-spacing algorithm.
    """
    if not out_nodes:
        return None

    segments = []
    cumulative = 0.0

    for a, b in H.edges():
        da = H.nodes[a].get("distance_to_base", 0)
        db = H.nodes[b].get("distance_to_base", 0)
        u, v = (a, b) if da <= db else (b, a)

        p_u = np.asarray(H.nodes[u]["pos"])
        p_v = np.asarray(H.nodes[v]["pos"])
        L = point_dist(p_u, p_v)
        if L < 2 * min_length:
            continue

        # The valid interval on the edge is [min_length, L - min_length].
        # This directly enforces the nude zone against the endpoints.
        s_lo = min_length
        s_hi = L - min_length

        segments.append((cumulative, u, v, L, s_lo, s_hi))
        cumulative += s_hi - s_lo

    if cumulative > 0:
        r = np.random.uniform(0.0, cumulative)
        lo, hi = 0, len(segments) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if segments[mid][0] <= r:
                lo = mid
            else:
                hi = mid - 1

        cum_start, u, v, L, s_lo, s_hi = segments[lo]
        s = float(np.clip(s_lo + (r - cum_start), s_lo, s_hi))

        p_u = np.asarray(H.nodes[u]["pos"])
        p_v = np.asarray(H.nodes[v]["pos"])
        point = p_u + (s / L) * (p_v - p_u)

        # Pick a random tip that is far enough from the sampled point
        valid_tips = []
        for tip in out_nodes:
            if point_dist(point, G.nodes[tip]['pos']) >= min_length:
                valid_tips.append(tip)

        if valid_tips:
            tip = valid_tips[np.random.randint(len(valid_tips))]
            return u, v, tip, point

    return None


def best_small_edge_attachment(G, H, out_nodes, alpha, min_length):
    """Return the best deterministic short-edge fallback attachment.

    This mirrors the greedy algorithm's ``small_edge`` fallback and is used by
    the randomized generator when no normally constrained attachment exists.
    """
    best_cost = float("inf")
    best_action = None

    for u, v in H.edges():
        if (
            H.nodes[u].get("distance_to_base", float("inf"))
            > H.nodes[v].get("distance_to_base", float("inf"))
        ):
            u, v = v, u

        p_u = np.array(H.nodes[u]["pos"])
        p_v = np.array(H.nodes[v]["pos"])
        for tip in out_nodes:
            cost, point, attach_type = find_optimal_t_on_analytical(
                p_u=p_u,
                p_v=p_v,
                p_tip=G.nodes[tip]["pos"],
                dist_to_base_u=H.nodes[u]["distance_to_base"],
                alpha=alpha,
                min_length=min_length,
            )
            if attach_type != "small_edge" or point is None:
                continue
            if cost < best_cost:
                best_cost = cost
                best_action = (u, v, tip, point)

    return best_action


def pareto_steiner_fast_analytical_preserve_tips_semi_random(
        G, alpha=None, min_length_factor=0.5, spacing_factor=1.0,
        optimization_rounds=0, length_weighted=True, verbose=False, record=False):
    """
    Continuous (analytical) Pareto Steiner tree with tip preservation.

    Enforces the EXACT SAME nude zone constraint as the fixed-spacing algorithm:
    a new branch point must be at least min_length away from the endpoints of
    the edge it splits, and at least min_length away from the new tip.

    ``optimization_rounds`` counts greedy decisions, including the basal
    connection.  With ``optimization_rounds=0`` every connection, the basal one
    included, is drawn at random; with ``optimization_rounds=r`` the first ``r``
    connections follow the Pareto-greedy rule and the rest are random.

    If the construction cannot connect every tip, the partial tree is rejected:
    the function returns ``None`` (or ``(None, frames)`` when ``record=True``).
    """
    if alpha is None:
        alpha = np.random.uniform(0, 1)
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

    # Step 1: connect the base to a first tip.
    #
    # If the optimization budget still allows a greedy round, use the greedy
    # rule (closest tip).  Otherwise draw uniformly among the tips that are far
    # enough away, so that the null model randomizes the basal connection too.
    base_pos = H.nodes[base_node]["pos"]
    valid_first_tips = [
        t for t in out_nodes
        if point_dist(G.nodes[t]['pos'], base_pos) >= min_length
    ]
    if not valid_first_tips:
        if record:
            return None, frames
        return None

    if optimization_rounds > 0:
        first_tip = min(valid_first_tips,
                        key=lambda t: point_dist(G.nodes[t]['pos'], base_pos))
    else:
        first_tip = valid_first_tips[np.random.randint(len(valid_first_tips))]

    tip_pos = G.nodes[first_tip]['pos']
    dist = point_dist(base_pos, tip_pos)

    H.add_node(first_tip)
    H.nodes[first_tip]["pos"] = tip_pos
    H.nodes[first_tip]["is_tip"] = True
    H.add_edge(base_node, first_tip, weight=dist)
    H.nodes[first_tip]["distance_to_base"] = dist
    out_nodes.remove(first_tip)

    if record:
        frames.append(H.copy())

    iteration = 2
    while out_nodes:
        use_random = (iteration - 1) >= optimization_rounds

        if not use_random:
            # --- GREEDY MODE (Analytical) ---
            best_cost = float('inf')
            best_action = None
            best_cost_small = float('inf')
            best_action_small = None

            for u, v in H.edges():
                if H.nodes[u].get("distance_to_base", float('inf')) > H.nodes[v].get("distance_to_base", float('inf')):
                    u, v = v, u

                p_u = np.array(H.nodes[u]['pos'])
                p_v = np.array(H.nodes[v]['pos'])
                for tip in out_nodes:
                    cost, point, attach_type = find_optimal_t_on_analytical(
                        p_u=p_u, p_v=p_v, p_tip=G.nodes[tip]['pos'],
                        dist_to_base_u=H.nodes[u]['distance_to_base'],
                        alpha=alpha, min_length=min_length,
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
            elif best_action_small is not None:
                u, v, tip, best_point = best_action_small
            else:
                break
        else:
            # --- RANDOM MODE ---
            chosen = sample_random_attachment(G, H, out_nodes, min_length)
            if chosen is None:
                chosen = best_small_edge_attachment(
                    G, H, out_nodes, alpha, min_length
                )
            if chosen is None:
                break
            u, v, tip, best_point = chosen

        # --- APPLY ACTION ---
        new_node = node_index
        node_index += 1
        H.add_node(new_node, pos=tuple(best_point), is_internal=True)
        H.remove_edge(u, v)

        dist_u_new = point_dist(H.nodes[u]['pos'], best_point)
        dist_new_v = point_dist(H.nodes[v]['pos'], best_point)
        dist_new_tip = point_dist(best_point, G.nodes[tip]['pos'])

        H.add_edge(u, new_node, weight=dist_u_new)
        H.add_edge(new_node, v, weight=dist_new_v)

        H.add_node(tip)
        H.nodes[tip]["pos"] = G.nodes[tip]["pos"]
        H.nodes[tip]["is_tip"] = True
        H.add_edge(new_node, tip, weight=dist_new_tip)

        # Distances to base
        dist_to_base_u = H.nodes[u].get('distance_to_base', 0)
        H.nodes[new_node]["distance_to_base"] = dist_to_base_u + dist_u_new
        H.nodes[tip]["distance_to_base"] = H.nodes[new_node]["distance_to_base"] + dist_new_tip

        update_distances_downstream(H, v, H.nodes[new_node]["distance_to_base"] + dist_new_v)

        out_nodes.remove(tip)
        iteration += 1

        if record:
            frames.append(H.copy())

    if out_nodes:
        # A tree that does not contain every original tip would have missing
        # wiring and travel contributions, making its costs artificially low.
        if record:
            return None, frames
        return None

    if record:
        return H, frames
    return H


def pareto_steiner_fast_analytical_preserve_tips_full_greedy(
    G, alpha, **kwargs
):
    """Run the randomized analytical generator in fully greedy mode.

    ``optimization_rounds`` is set to the total number of tips plus the base,
    so every decision is greedy. This is the UI's replacement for the separate
    deterministic analytical implementation.
    """
    return pareto_steiner_fast_analytical_preserve_tips_semi_random(
        G,
        alpha,
        optimization_rounds=len(get_critical_nodes(G)),
        **kwargs,
    )


def find_optimal_t_on_analytical(p_u, p_v, p_tip, dist_to_base_u, alpha, min_length):
    """
    Find optimal attachment point along the line from p_u to p_v for tip at p_tip.

    Args:
        p_u, p_v: Endpoints of the direction line (p_u is in tree, p_v is toward tip)
        p_tip: Position of the tip to attach
        dist_to_base_u: Distance from base to u
        alpha: Pareto weight
        min_length: Minimum edge length

    Returns:
        (cost, best_point, attach_type)
    """
    p_u = np.array(p_u)
    p_v = np.array(p_v)
    p_tip = np.array(p_tip)

    e_vec = p_v - p_u
    L = np.linalg.norm(e_vec)

    if L < 1e-6:
        return (float('inf'), None, 'edge')

    e_hat = e_vec / L

    # Edge too short for valid attachment
    if L < 2 * min_length:
        midpoint = p_u + 0.5 * e_vec
        dist_to_tip = np.linalg.norm(p_tip - midpoint)
        wiring = dist_to_tip
        travel = dist_to_tip + dist_to_base_u + 0.5 * L
        cost = pareto_cost(wiring, travel, alpha)
        return (cost, tuple(midpoint), 'small_edge')

    t_min = min_length / L
    t_max = 1.0 - t_min

    w_vec = p_tip - p_u
    w_dot_e = np.dot(w_vec, e_vec)
    w_norm_sq = np.dot(w_vec, w_vec)

    def cost_at(t):
        point = p_u + t * e_vec
        d = np.linalg.norm(p_tip - point)
        wiring = d
        travel = d + dist_to_base_u + t * L
        return pareto_cost(wiring, travel, alpha), d

    candidates = [t_min, t_max]

    if alpha == 1.0:
        t_perp = w_dot_e / (L * L)
        t_perp = np.clip(t_perp, t_min, t_max)
        candidates.append(t_perp)
    elif alpha > 0.0:
        k = 2 * alpha - alpha * alpha
        a = L * L
        b = -2.0 * w_dot_e
        c = (w_dot_e * w_dot_e - (1 - alpha) ** 2 * L * L * w_norm_sq) / (k * L * L)
        disc = b * b - 4.0 * a * c

        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            for root in [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]:
                if t_min <= root <= t_max:
                    point = p_u + root * e_vec
                    d_root = np.linalg.norm(p_tip - point)
                    if d_root > 1e-12:
                        lhs = (L * L * root - w_dot_e) / d_root
                        rhs = -(1.0 - alpha) * L
                        if abs(lhs - rhs) < 1e-8 or lhs < 0:
                            candidates.append(root)

    best_cost = float('inf')
    best_t = None
    for t in candidates:
        t = np.clip(t, t_min, t_max)
        cost, d = cost_at(t)
        if d < min_length:
            continue
        if cost < best_cost:
            best_cost = cost
            best_t = t

    if best_t is None:
        return (float('inf'), None, 'edge')

    best_point = p_u + best_t * e_vec
    return (best_cost, tuple(best_point), 'edge')







def pareto_steiner_fast_3d_path_tortuosity(G, alpha, beta):
    """Given a graph G and a value 0 <= {alpha, beta} <= 1, compute the Pareto-optimal tree
    connecting the base node to all of the lateral root tips of G.

    cost = alpha * total_root_length + beta * total_travel_distance - gamma * total_path_coverage

    alpha + beta + gamma = 1

    When alpha = beta = 0, gamma = 1 => cost = -total_path_coverage will be minimized =>
        total_path_coverage will be maximized
    When alpha = gamma = 0, beta = 1 => cost = total_travel_distance will be minimized =>
        Satellite tree will be constructed
    When beta = gamma = 0, alpha = 1 => cost = total_root_length will be minimized =>
        Steiner tree will be constructed

    total_root_length: the sum of the lengths of the edges in the root network
        (a.k.a. material cost, wiring cost)
    total_travel_distance: the sum of the lengths of the shortest paths from every
        lateral root tip to the base node of the network. (a.k.a. the satellite cost,
        conduction delay)
    total_path_coverage: the sum of the tortuosity of all the root paths. The tortuosity per
        path is defined as the ratio of the actual path length to the shortest path
        length between the base node and the root tip. The total path coverage is the sum of
        the tortuosity of all the root paths.

    The algorithm uses a greedy approach: always take the edge that will reduce the
    pareto cost of the tree by the smallest amount
    """
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1

    # assume the base_node is node 0
    base_node = 0

    H = nx.Graph()

    H.add_node(base_node)
    # every node will keep track of its distance to the base
    H.nodes[base_node]["distance_to_base"] = 0
    base_pos = G.nodes[base_node]["pos"]
    H.nodes[base_node]["pos"] = base_pos
    added_nodes = 1

    # every node will keep track of the shortest path to the base_node
    H.nodes[base_node]["straight_distance_to_base"] = 0

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
    node that has not been added to the tree.
    """
    unpaired_nodes = set([base_node])

    # keeps track of what the id of the next node added to the tree should be.
    node_index = max(critical_nodes) + 1

    steps = 0

    """
    The optimization this algorithm performs is that we don't need to consider every
    combination (u, v) where u is in the tree and v is not in the tree. We only need to
    consider combinations where u is in the tree, and v is the closest node to u that has
    not been added to the  tree. This is because any other combination would never be the
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
            total_path_coverage = total_travel_distance / point_dist(base_pos, p2)
            cost = pareto_cost_3d_path_tortuosity(
                total_root_length=total_root_length,
                total_travel_distance=total_travel_distance,
                total_path_coverage=total_path_coverage,
                alpha=alpha,
                beta=beta,
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
            H.nodes[n2]["straight_distance_to_base"] = H.nodes[n2][
                "distance_to_base"
            ] / node_dist(H, n2, base_node)

        added_nodes += 1
    return H


def update_distances_downstream(H, start_node, new_dist_to_base):
    """
    Update distance_to_base for start_node and all its descendants.
    """
    old_dist = H.nodes[start_node].get('distance_to_base', 0)
    delta = new_dist_to_base - old_dist

    if abs(delta) < 1e-6:
        return

    # BFS from start_node to update all downstream nodes
    queue = [(start_node, None)]  # (node, parent)
    visited = set()

    while queue:
        node, parent = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        H.nodes[node]["distance_to_base"] += delta

        for neighbor in H.neighbors(node):
            if neighbor != parent:
                queue.append((neighbor, node))




def pareto_front_3d_path_tortuosity(G):
    """
    Given a graph G, compute the Pareto front of optimal solutions for the 3D path tortuosity

    This allows to compare how G was connected and how G could have been connected had it
    been trying to optimize wiring cost, conduction delay, and path tortuosity

    Args:
        G (nx.Graph): The graph to compute the Pareto front for
    Returns:
        front (dict): A dictionary of edge_lengths, travel_distances_to_base, and
            path_coverages for (each alpha, beta) value on the front
        actual (tuple): The actual total_root_length, total_travel_distance, and
            total_path_coverage of the original plant
    """
    critical_nodes = get_critical_nodes(G)

    # test: compute the actual total_root_length, total_travel_distance, and
    # total_path_coverage for the original plant
    mactual, sactual, pactual = graph_costs_3d_path_tortuosity(
        G, critical_nodes=critical_nodes
    )
    actual = (mactual, sactual, pactual)

    # dictionary of edge_lengths, travel_distances_to_base, and path_coverages for each
    # alpha, beta value on the front
    front = {}

    for alpha in DEFAULT_ALPHAS:
        for beta in DEFAULT_BETAS:
            # Skip invalid combinations where alpha + beta > 1 (gamma would be negative)
            if alpha + beta > 1:
                continue

            H = None
            # if alpha = 0 and beta = 1 compute the satellite tree in linear time
            if alpha == 0 and beta == 1:
                H = satellite_tree(G)
            else:
                H = pareto_steiner_fast_3d_path_tortuosity(G, alpha, beta)

            # compute the wiring cost, conduction delay and path coverage
            # only the original critical nodes contribute to conduction delay
            total_root_length, total_travel_distance, total_path_coverage = (
                graph_costs_3d_path_tortuosity(H, critical_nodes=critical_nodes)
            )
            front[(alpha, beta)] = [
                total_root_length,
                total_travel_distance,
                total_path_coverage,
            ]

    return front, actual




def random_tree_3d_path_tortuosity(G):
    """
    Given a graph G, compute 1000 random spanning trees as in Conn et al. 2017.
    Only consider the critical nodes (and root node) of G.

    Args:
        G (nx.Graph): The graph to compute the random trees for
    Returns:
        costs (list): A list of (mactual, sactual, pactual) tuples for each random tree which
            represent the wiring cost, conduction delay, and path coverage of the tree.
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
        mactual, sactual, pactual = graph_costs_3d_path_tortuosity(R)
        costs.append((mactual, sactual, pactual))

    return costs


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






def clean_pareto_front(front):
    """
    Remove dominated points from a Pareto front.
    A point is dominated if there exists another point with both lower
    wiring AND lower travel.
    """
    points = [(alpha, w, t) for alpha, (w, t) in front.items()]
    # Sort by wiring ascending
    points.sort(key=lambda x: x[1])

    cleaned = {}
    best_travel = float('inf')

    for alpha, w, t in points:
        if t < best_travel:
            best_travel = t
            cleaned[alpha] = [w, t]

    return cleaned

def _tree_hover_payload(tree, rounds, alpha):
    """Serialize a sampled tree for the interactive HTML hover panel.

    Positions and edges are stored as compact numeric arrays so thousands of
    trees can be embedded without carrying NetworkX objects or Matplotlib
    figures into the report.
    """
    nodes = list(tree.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    positions = []
    for node in nodes:
        pos = tree.nodes[node]["pos"]
        positions.append([round(float(pos[0]), 2), round(float(pos[1]), 2)])
    edges = [
        [node_index[u], node_index[v]]
        for u, v in tree.edges()
    ]
    return {
        "p": positions,
        "e": edges,
        "b": node_index.get(0, 0),
        "a": None if alpha is None else round(float(alpha), 4),
        "r": None if rounds is None else int(rounds),
    }


def random_steiner_wrapper_analytical(G, n, optimization_rounds=0):
    """Random baseline using the analytical tip-preserving Steiner generator.

    This is the 2.0 UI baseline: it uses the same continuous (analytical)
    attachment rule as the optimal front, with ``optimization_rounds=0`` so
    every connection is drawn randomly.
    """
    costs, trees = [], []
    for _ in range(n):
        alpha = np.random.uniform(0, 1)
        tree = pareto_steiner_fast_analytical_preserve_tips_semi_random(
            G, alpha=alpha, optimization_rounds=optimization_rounds
        )
        if tree is not None:
            m, s = graph_costs(tree, get_critical_nodes(tree))
            costs.append((m, s))
            trees.append(tree)
    return costs, trees
