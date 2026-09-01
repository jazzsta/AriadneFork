import networkx as nx
import numpy as np
import random

from bisect import insort


STEINER_MIDPOINTS = 10

DEFAULT_ALPHAS = np.concatenate([
    np.arange(0, 0.1, 0.001),   # 0 to 0.095: steps of 0.001
    np.arange(0.1, 0.3, 0.01),  # 0.1 to 0.29: steps of 0.01
    np.arange(0.3, 0.6, 0.02),  # 0.3 to 0.58: steps of 0.02
    np.arange(0.6, 1.01, 0.05), # 0.6 to 1.0: steps of 0.05
])
DEFAULT_ALPHAS = np.clip(DEFAULT_ALPHAS, 0.0, 1.0)
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

    return H


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



def pareto_front(G):
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
            H = pareto_steiner_fast(G, alpha)

        # compute the wiring cost and conduction delay
        # only the original critical nodes contribute to conduction delay
        total_root_length, total_travel_distance = graph_costs(
            H, critical_nodes=critical_nodes
        )
        front[alpha] = [total_root_length, total_travel_distance]

    return front, actual

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


def pareto_front_generic(G, steiner_function):
    """
    Compute Pareto front using any steiner function.

    Args:
        G: Input graph
        steiner_function: Function that takes (G, alpha) and returns a tree

    Returns:
        front: dict mapping alpha -> [wiring, travel]
        actual: tuple (wiring, travel) for the original tree
    """
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = point_dist(G.nodes[u]['pos'], G.nodes[v]['pos'])
    critical_nodes = get_critical_nodes(G)

    mactual, sactual = graph_costs(G, critical_nodes=critical_nodes)
    actual = (mactual, sactual)

    front = {}

    for alpha in DEFAULT_ALPHAS:
        if alpha == 0:
            H = satellite_tree(G)
        else:
            H = steiner_function(G, alpha)

        if H is not None:
            total_root_length, total_travel_distance = graph_costs(
                H, critical_nodes=critical_nodes
            )
            front[alpha] = [total_root_length, total_travel_distance]

    return clean_pareto_front(front), actual


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