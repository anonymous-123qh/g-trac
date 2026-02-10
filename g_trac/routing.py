import math
import heapq
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from . import consts, utils


print("__name__ =", __name__)
print("__package__ =", __package__)

def build_routing_graph(candidates: List[Dict[str, Any]]):
    nodes = {str(c['id']): c for c in candidates}
    graph = {'SOURCE': {}, 'SINK': {}}
    by_start = {}
    for c in candidates:
        ls = int(c['layer_start'])
        if ls not in by_start: by_start[ls] = []
        by_start[ls].append(c)

    for c in candidates:
        cid = str(c['id'])
        if cid not in graph: graph[cid] = {}
        lat = utils.get_lat_ms(c)
        tr = max(utils.get_trust(c), 0.0001)
        risk = -math.log(tr)

        if int(c['layer_start']) == 0:
            graph['SOURCE'][cid] = {'latency': lat, 'reliability': tr, 'risk_cost': risk}

        #using consts.MODEL_LAYERS
        if int(c['layer_end']) == consts.MODEL_LAYERS - 1:
            graph[cid]['SINK'] = {'latency': 0.0, 'reliability': 1.0, 'risk_cost': 0.0}

        next_ls = int(c['layer_end']) + 1
        if next_ls in by_start:
            for nxt in by_start[next_ls]:
                nid = str(nxt['id'])
                n_lat = utils.get_lat_ms(nxt)
                n_tr = max(utils.get_trust(nxt), 0.0001)
                n_risk = -math.log(n_tr)
                graph[cid][nid] = {'latency': n_lat, 'reliability': n_tr, 'risk_cost': n_risk}
    return graph, nodes


def run_dijkstra(graph, start_node, end_node, weight_fn):
    pq = [(0.0, start_node, [])]
    visited_costs = {}
    while pq:
        cost, u, path = heapq.heappop(pq)
        path = path + [u]
        if u == end_node: return path, cost
        if u in visited_costs and visited_costs[u] <= cost: continue
        visited_costs[u] = cost
        for v, attrs in graph.get(u, {}).items():
            w = weight_fn(attrs)
            if w == float('inf'): continue
            heapq.heappush(pq, (cost + w, v, path))
    return None, float('inf')


def find_path_larac(graph, start_node, end_node, min_reliability):
    """
    LARAC Heuristic for Constrained Shortest Path.
    Minimize Latency s.t. Reliability >= min_reliability
    """
    max_risk = -math.log(max(min_reliability, 1e-9))

    #fastest path (Lambda = 0)
    p_L, _ = run_dijkstra(graph, start_node, end_node, lambda e: e['latency'])
    if not p_L: return None

    def calc_risk(path):
        r = 0.0
        for i in range(len(path) - 1):
            r += graph[path[i]][path[i + 1]]['risk_cost']
        return r

    risk_L = calc_risk(p_L)
    if risk_L <= max_risk:
        return p_L  #fastest path satisfies constraint

    # 2. Safest path (Lambda huge)
    p_R, _ = run_dijkstra(graph, start_node, end_node, lambda e: e['risk_cost'])
    if not p_R: return None

    risk_R = calc_risk(p_R)
    if risk_R > max_risk:
        return None  #even safest path violates constraint

    def calc_lat(path):
        l = 0.0
        for i in range(len(path) - 1):
            l += graph[path[i]][path[i + 1]]['latency']
        return l

    lat_L = calc_lat(p_L)
    lat_R = calc_lat(p_R)

    # 3. Iterate
    best_path = p_R
    lambda_val = (lat_R - lat_L) / (risk_L - risk_R + 1e-9)

    for _ in range(5):  #5-10 iterations
        p_new, _ = run_dijkstra(graph, start_node, end_node, lambda e: e['latency'] + lambda_val * e['risk_cost'])
        if not p_new: break

        r_new = calc_risk(p_new)
        l_new = calc_lat(p_new)

        if r_new <= max_risk:
            best_path = p_new
            risk_R = r_new
            lat_R = l_new
        else:
            risk_L = r_new
            lat_L = l_new

        if abs(risk_L - risk_R) < 1e-6: break
        lambda_val = (lat_R - lat_L) / (risk_L - risk_R + 1e-9)

    return best_path


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

    def dfs(expected_ls: int, chain: List[Dict[str, Any]]):
        if len(out) >= max_chains:
            return
        if expected_ls == consts.MODEL_LAYERS:
            out.append(chain[:])
            return
        options = start_map.get(expected_ls, [])
        if not options:
            return
        for n in options:
            ls, le = utils.node_layers(n)
            next_expected = le + 1
            if next_expected > consts.MODEL_LAYERS:
                continue
            chain.append(n)
            dfs(next_expected, chain)
            chain.pop()

    dfs(0, [])
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
    # 1. NAIVE / RANDOM
    if mode in ("naive", "rnd", "random"):
        # Simple random valid path
        return _find_random_valid_path(alive_candidates)

    #Build Graph for advanced modes
    graph, node_map = _build_routing_graph(alive_candidates)

    # 2. G-TRAC
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

    # 3. SP (Shortest Path / Lowest Latency)
    if mode == "sp":
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='latency')

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    # 4. MR (Max Reliability / Max Trust)
    if mode == "mr":
        path_ids = _run_dijkstra(graph, 'SOURCE', 'SINK', weight_key='risk_cost')

        if path_ids and len(path_ids) > 2:
            return [node_map[wid] for wid in path_ids[1:-1]]
        return []

    # 5. LARAC (Constrained Shortest Path)
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

    # 1. Fastest path (Lambda = 0) -> Pure Latency
    p_L = _run_dijkstra(graph, start_node, end_node, weight_key='latency')
    if not p_L: return None

    risk_L = calc_risk(p_L)
    if risk_L <= max_risk:
        return p_L  # Fastest path satisfies constraint

    # 2. Safest path (Lambda huge) -> Pure Risk
    p_R = _run_dijkstra(graph, start_node, end_node, weight_key='risk_cost')
    if not p_R: return None

    risk_R = calc_risk(p_R)
    if risk_R > max_risk:
        return None  # Even safest path violates constraint (Impossible)

    lat_L = calc_lat(p_L)
    lat_R = calc_lat(p_R)

    # 3. Iterate
    best_path = p_R

    # Avoid division by zero
    denom = (risk_L - risk_R)
    if abs(denom) < 1e-9:
        return best_path

    lambda_val = (lat_R - lat_L) / denom

    for _ in range(10):
        # Dijkstra with custom weight: latency + lambda * risk
        def weight_fn(edge):
            return edge.get('latency', 0) + lambda_val * edge.get('risk_cost', 0)

        p_new = _run_dijkstra_custom(graph, start_node, end_node, weight_fn)
        if not p_new: break

        r_new = calc_risk(p_new)
        l_new = calc_lat(p_new)

        # Check feasibility
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

    target = consts.MODEL_LAYERS - 1

    def dfs(curr_layer_start):
        cands = by_start.get(curr_layer_start, [])
        random.shuffle(cands)
        for c in cands:
            le = int(c['layer_end'])
            if le == target: return [c]
            suffix = dfs(le + 1)
            if suffix: return [c] + suffix
        return None

    return dfs(0) or []


def find_replacement_for_failed_hop(candidates: List[Dict[str, Any]], failed_id: str,
                                    ls: int, le: int, mode: str) -> Optional[Dict[str, Any]]:
    replacements = [
        n for n in candidates
        if int(n["layer_start"]) == ls and int(n["layer_end"]) == le
           and str(n["id"]) != str(failed_id) and utils.is_alive(n)
    ]
    if not replacements: return None

    if mode in ("tarp", "mr"):
        replacements.sort(key=lambda x: float(x.get("trust", 0)), reverse=True)
    elif mode == "sp":
        replacements.sort(key=lambda x: float(x.get("lat_ewma_ms", 999)))
    else:
        random.shuffle(replacements)
    return replacements[0]

def _run_dijkstra(graph: Dict, start: str, end: str, weight_key: str) -> Optional[List[str]]:
    """
    Simple wrapper for single-key Dijkstra.
    This fixes the 'unexpected keyword argument' error.
    """
    return _run_dijkstra_custom(graph, start, end, lambda e: e.get(weight_key, 1e9))