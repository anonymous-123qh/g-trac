import math
import heapq
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from . import consts, utils


print("__name__ =", __name__)
print("__package__ =", __package__)


def enumerate_feasible_chains(
        candidates: List[Dict[str, Any]],
        max_chains: int,
) -> List[List[Dict[str, Any]]]:
    """
    Enumerate feasible chains that cover layers [0..MODEL_LAYERS-1] contiguously.
    Feasible chain condition:
      start at layer 0, pick node with (ls==0), then next node must have ls==prev_le+1, ...
      until coverage reaches MODEL_LAYERS-1.
    Multiple nodes may offer same (ls,le); that provides replication.
    """
    #build mapping: ls -> list of nodes starting at ls
    start_map: Dict[int, List[Dict[str, Any]]] = {}
    for n in candidates:
        ls, le = utils.node_layers(n)
        if ls < 0 or le < ls:
            continue
        start_map.setdefault(ls, []).append(n)

    for ls in start_map:
        #prefer high trust, then low latency
        start_map[ls].sort(key=lambda n: (-utils.get_trust(n), utils.get_lat_ms(n)))

    out: List[List[Dict[str, Any]]] = []

    #iterative Stack: each element is (expected_ls, current_chain)
    #use a stack to simulate the DFS behavior
    stack = [(0, [])]

    while stack and len(out) < max_chains:
        expected_ls, chain = stack.pop()

        # Success condition: chain covers all layers
        if expected_ls == consts.MODEL_LAYERS:
            out.append(chain)
            continue

        options = start_map.get(expected_ls, [])
        # We process options in reverse for the stack so the
        # 'best' workers (sorted to the front) are popped first.
        for n in reversed(options):
            ls, le = utils.node_layers(n)
            next_expected = le + 1

            if next_expected <= consts.MODEL_LAYERS:
                # Push a NEW chain to the stack
                stack.append((next_expected, chain + [n]))

    return out



def chain_score(chain: List[Dict[str, Any]], mode: str) -> float:
    """
    Lower is better (we minimize).
    - naive: score unused (random pick)
    - gtrac: minimize sum latency
    - gtrac_weighted: minimize negative utility combining trust and latency
    """
    if mode == "g-trac":
        return sum(utils.get_lat_ms(n) for n in chain)

    #gtrac_weighted
    alpha = 0.70  # weight on trust
    eps = 1e-6
    trust_term = sum(math.log(max(utils.get_trust(n), eps)) for n in chain)
    lat_term = sum(utils.get_lat_ms(n) for n in chain) / 100.0  # scale
    utility = alpha * trust_term - (1.0 - alpha) * lat_term
    return -utility

def select_chain(candidates: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
        Selects a sequence of workers that cover layers 0 to MODEL_LAYERS-1.
    """

    #Basic filtering: Must be alive and have valid layer ranges
    alive_candidates = [c for c in candidates if utils.is_alive(c)]
    if not alive_candidates:
        return []

    #
    #NAIVE / RANDOM
    if mode in ("naive", "rnd", "random"):
        # Simple random valid path
        return _find_random_valid_path(alive_candidates)

    #Build Graph for advanced modes
    graph, node_map = _build_routing_graph(alive_candidates)

    #G-TRAC
    # Strategy: Prune untrusted nodes, then find Lowest Latency path (Dijkstra)
    if mode =="g-trac":
        # Pruning Step: Filter out nodes with low trust
        trusted_nodes = [
            c for c in alive_candidates
            if float(c.get("trust", 0.5)) >= consts.TRUST_MIN_PER_HOP
        ]

        #Fallback: If pruning removes too many nodes (no path possible),
        nodes_to_use = trusted_nodes if len(trusted_nodes) >= 2 else alive_candidates

        graph, map_gtrac = _build_routing_graph(nodes_to_use)

        #Dijkstra for Latency
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='latency')

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    #SP (Shortest Path / Lowest Latency)
    if mode == "sp":
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='latency')

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    #MR (Max Reliability / Max Trust)
    if mode == "mr":
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='risk_cost')

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    #LARAC (Constrained Shortest Path)
    if mode == "larac":
        min_r = consts.LARAC_MIN_RELIABILITY
        path_ids = find_path_larac(graph, 'SOURCE', 'SINK', min_r)

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]

        # Fallback if constraint cannot be met: Max Reliability path
        print("[Routing] LARAC failed constraint, falling back to Max Reliability")
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='risk_cost')
        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    return []

def find_replacement_for_failed_hop(
        failed_id: str,
        failed_ls: int,
        failed_le: int,
        mode: str
) -> Optional[Dict[str, Any]]:
    """
    Find a replacement node that hosts the same (ls,le) segment.
    In tarp modes, also require trust>=threshold.
    Choose lowest EWMA latency among viable.
    """
    candidates = [c for c in consts.LOCAL_CACHE.values() if utils.is_alive(c)]
    viable: List[Dict[str, Any]] = []
    for c in candidates:
        if str(c.get("id")) == str(failed_id):
            continue
        ls, le = utils.node_layers(c)
        if (ls, le) != (failed_ls, failed_le):
            continue
        if mode in ("g-trac", "g-trac_weighted") and utils.get_trust(c) < consts.TRUST_MIN_PER_HOP:
            continue
        viable.append(c)
    if not viable:
        return None
    return min(viable, key=utils.get_lat_ms)


def find_path_larac(graph: Dict, start_node: str, end_node: str, min_reliability: float) -> Optional[List[str]]:
    """
    LARAC Heuristic for Constrained Shortest Path.
    Minimize Latency s.t. Reliability >= min_reliability
    """
    # Convert reliability (0..1) to Max Risk (sum of -log(trust))
    max_risk = -math.log(max(min_reliability, 1e-9))

    # Helper to calculate path costs
    def calc_risk(path):
        r = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in graph and v in graph[u]:
                r += graph[u][v].get('risk_cost', 0.0)
        return r

    def calc_lat(path):
        l = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in graph and v in graph[u]:
                l += graph[u][v].get('latency', 0.0)
        return l

    #Fastest path (Lambda = 0) -> Pure Latency
    p_L = _run_dijkstra(graph, start_node, end_node, weight_key='latency')
    if not p_L: return None

    risk_L = calc_risk(p_L)
    if risk_L <= max_risk:
        return p_L  # Fastest path satisfies constraint

    #Safest path (Lambda huge) -> Pure Risk
    p_R = _run_dijkstra(graph, start_node, end_node, weight_key='risk_cost')
    if not p_R: return None

    risk_R = calc_risk(p_R)
    if risk_R > max_risk:
        return None  #even safest path violates constraint

    lat_L = calc_lat(p_L)
    lat_R = calc_lat(p_R)

    #Iterate
    best_path = p_R

    #avoid division by zero
    denom = (risk_L - risk_R)
    if abs(denom) < 1e-9:
        return best_path

    lambda_val = (lat_R - lat_L) / denom

    for _ in range(10):
        #Dijkstra with custom weight: latency + lambda * risk
        def weight_fn(edge):
            return edge.get('latency', 0) + lambda_val * edge.get('risk_cost', 0)

        p_new = _run_dijkstra_custom(graph, start_node, end_node, weight_fn)
        if not p_new: break

        r_new = calc_risk(p_new)
        l_new = calc_lat(p_new)

        #check feasibility
        if r_new <= max_risk:
            best_path = p_new
            risk_R = r_new
            lat_R = l_new
        else:
            risk_L = r_new
            lat_L = l_new

        if abs(risk_L - risk_R) < 1e-6: break

        denom = (risk_L - risk_R)
        if abs(denom) < 1e-9: break
        lambda_val = (lat_R - lat_L) / denom

    return best_path


def _build_routing_graph(nodes: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
    graph = {'SOURCE': {}, 'SINK': {}}
    node_map = {}

    # 1. Pre-calculate node costs
    for n in nodes:
        wid = str(n['id'])
        node_map[wid] = n
        graph[wid] = {}

        # Costs
        lat = float(n.get("lat_ewma_ms", consts.LAT_INIT_MS))
        trust = float(n.get("trust", 0.5))
        trust = max(0.01, min(1.0, trust))

        n['_w_lat'] = lat
        n['_w_risk'] = -math.log(trust)

    # 2. SOURCE -> Nodes (Layer 0)
    for n in nodes:
        if int(n['layer_start']) == 0:
            wid = n['id']
            # Edge weight is the node's processing cost
            graph['SOURCE'][wid] = {'latency': n['_w_lat'], 'risk_cost': n['_w_risk']}

    # 3. Node -> Node (Chaining)
    target_layer_end = consts.MODEL_LAYERS - 1

    # Group by layer_start for O(N) lookup instead of O(N^2)
    by_start = {}
    for n in nodes:
        s = int(n['layer_start'])
        if s not in by_start: by_start[s] = []
        by_start[s].append(n)

    for nA in nodes:
        idA = nA['id']
        endA = int(nA['layer_end'])

        if endA == target_layer_end:
            # Connect to SINK (0 cost)
            graph[idA]['SINK'] = {'latency': 0.0, 'risk_cost': 0.0}
        else:
            # Connect to next hop
            next_start = endA + 1
            if next_start in by_start:
                for nB in by_start[next_start]:
                    idB = nB['id']
                    # Edge weight is NEXT node's cost
                    graph[idA][idB] = {'latency': nB['_w_lat'], 'risk_cost': nB['_w_risk']}

    return graph, node_map


def _run_dijkstra_custom(graph: Dict, start: str, end: str, weight_fn: Callable) -> Optional[List[str]]:
    """Dijkstra with custom weight function"""
    pq = [(0.0, start, [start])]
    visited = set()
    min_dist = {start: 0.0}

    while pq:
        cost, curr, path = heapq.heappop(pq)

        if curr == end:
            return path

        if cost > min_dist.get(curr, float('inf')):
            continue

        if curr not in graph:
            continue

        for neighbor, edge_data in graph[curr].items():
            edge_weight = weight_fn(edge_data)
            new_cost = cost + edge_weight

            if new_cost < min_dist.get(neighbor, float('inf')):
                min_dist[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return None


def _find_random_valid_path(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Bucket nodes by start layer
    by_start = {}
    for n in nodes:
        s = int(n['layer_start'])
        if s not in by_start: by_start[s] = []
        by_start[s].append(n)

    target_end = consts.MODEL_LAYERS - 1

    # 2. Manual Stack for DFS: (current_layer_to_find, current_chain_list)
    # We start looking for layer 0 with an empty chain.
    stack = [(0, [])]

    # Optional: Shuffle to ensure randomness across the whole process
    for s in by_start:
        random.shuffle(by_start[s])

    while stack:
        curr_ls, chain = stack.pop()

        # Get candidates that start at the required layer
        options = by_start.get(curr_ls, [])

        for opt in options:
            le = int(opt['layer_end'])
            new_chain = chain + [opt]

            # SUCCESS: We reached the final layer
            if le == target_end:
                return new_chain

            # CONTINUE: Push the next required layer onto the stack
            next_ls = le + 1
            if next_ls in by_start:
                stack.append((next_ls, new_chain))

    # If we exhaust the stack without returning, no path exists
    return []




def _run_dijkstra(graph: Dict, start: str, end: str, weight_key: str) -> Optional[List[str]]:
    """
    Simple wrapper for single-key Dijkstra.
    This fixes the 'unexpected keyword argument' error.
    """
    return _run_dijkstra_custom(graph, start, end, lambda e: e.get(weight_key, 1e9))
