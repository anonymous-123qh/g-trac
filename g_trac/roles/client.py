import requests
import time
import random
import threading
from .. import consts, utils, routing
from typing import Any, Dict, List, Optional, Tuple
import os
import csv
import math
LOCAL_CACHE = {}
ANCHOR_REACHABLE = True

#DETECT CAPABILITIES JUST IN CASE SOME NODES DOES NOT HAVE TORCH ...
try:
    import torch
    from transformers import AutoTokenizer, AutoConfig
    HAS_TORCH = True
    print("[Client] PyTorch & Transformers detected. Using HEAVY mode (Faster Server).")
except ImportError:
    HAS_TORCH = False
    print("[Client] PyTorch NOT detected. Using LIGHT mode (Text-Only).")

#Globals
LOCAL_CACHE = {}
ANCHOR_REACHABLE = True
TOKENIZER = None
GLOBAL_CONFIG = None
#CURRENT_MODEL_LAYERS = consts.MODEL_LAYERS  #updated by AutoConfig
MODEL_LAYERS = 36  #default to 36 (GPT-2 Large) if can't load config

#Helper Functions for only Client
def _chain_ids(chain: list) -> str:
    return "->".join(str(n.get("id")) for n in chain)

def _chain_layers(chain: list) -> str:
    return "->".join(f"{int(n.get('layer_start'))}-{int(n.get('layer_end'))}" for n in chain)

def _chain_trust_stats(chain: list) -> tuple:
    vals = []
    for n in chain:
        try:
            vals.append(utils.get_trust(n))
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"))
    return (min(vals), sum(vals)/len(vals))


def probe_worker_reachable(node: Dict[str, Any], timeout: float = 0.5) -> bool:
    """
    Checks if a worker is physically reachable AND serving the specific ID.
    Handles both Single-Worker and Multi-Worker responses.
    """
    try:
        url = f"http://{node['ip']}:{int(node['port'])}/status"
        r = requests.get(url, timeout=timeout)

        if r.status_code != 200:
            return False

        j = r.json()

        #logic from original probe_alive:
                #- If the worker returns a list of "peers", we must check if
                #the specific ID we want is actually in that list.
        if isinstance(j, dict) and "peers" in j and isinstance(j["peers"], list):
            want_id = str(node.get("id", ""))
            #create a set of IDs served by this physical worker
            served_ids = {str(p.get("id", "")) for p in j["peers"] if isinstance(p, dict)}

            #it is reachable only if our target ID is served here
            if want_id and want_id not in served_ids:
                return False

        return True
    except Exception:
        return False

def log_trust_snapshot(mode: str, request_id: str, chain: list, failed_id: str):
    """Logs the trust state of nodes involved in a request."""
    chain_ids = {str(n.get("id")) for n in chain}
    ids = set(chain_ids)
    if failed_id:
        ids.add(str(failed_id))

    #look up node details in LOCAL_CACHE
    for nid in ids:
        meta = LOCAL_CACHE.get(str(nid))
        if not meta:
            continue

        #define fields specific to trust log
        TRUST_FIELDS = ["run_id", "ts_unix", "mode", "request_id", "node_id", "trust", "lat_ewma_ms", "alive",
                        "layer_start", "layer_end", "in_chain", "failed_id"]

        utils.log_row(consts.TRUST_LOG_PATH, TRUST_FIELDS, {
            "run_id": consts.RUN_ID, "ts_unix": utils.unix_ts(),
            "mode": mode,
            "request_id": request_id,
            "node_id": str(nid),
            "trust": round(utils.get_trust(meta), 3),
            "lat_ewma_ms": round(utils.get_lat_ms(meta), 2),
            "alive": 1 if utils.is_alive(meta) else 0,
            "layer_start": int(meta.get("layer_start", -1)),
            "layer_end": int(meta.get("layer_end", -1)),
            "in_chain": 1 if str(nid) in chain_ids else 0,
            "failed_id": (str(failed_id) if failed_id is not None else "unknown"),
        })

def client_gossip_loop(anchor_ip, anchor_port):
    """
       Background gossip thread that periodically synchronizes local state
       with the anchor node.

       - Sends HTTP GET requests to /sync.
       - On success (HTTP 200), updates LOCAL_CACHE with anchor snapshot.
       - Maintains ANCHOR_REACHABLE flag for liveness tracking.
       - Executes every consts.GOSSIP_FREQ_SEC seconds.

       """

    global LOCAL_CACHE, ANCHOR_REACHABLE

    target_url = f"http://{anchor_ip}:{anchor_port}/sync"
    print(f"[Gossip] Thread started. Target: {target_url}", flush=True)

    while True:
        try:
            r = requests.get(f"http://{anchor_ip}:{anchor_port}/sync", timeout=3)
            if r.status_code == 200:
                LOCAL_CACHE = r.json()
                if not ANCHOR_REACHABLE:
                    print(f"[Gossip] SUCCESS! Connected to {anchor_ip}")
                ANCHOR_REACHABLE = True
            else:
                print(f"[Gossip] Connected, but HTTP Error: {r.status_code}")
                ANCHOR_REACHABLE = False
        except Exception as e:
            ANCHOR_REACHABLE = False
            print(f"[Gossip] FAILED: {e}")
        time.sleep(consts.GOSSIP_FREQ_SEC)


def run_sweep(anchor_ip, anchor_port, prompt, n_tokens=50, reps=3):
    """
        Runs an experimental sweep across multiple decision modes.

        For each mode:
          1) Reset anchor-side state (e.g., trust/latency estimators) via /reset_state.
          2) Wait briefly to allow the gossip thread to refresh LOCAL_CACHE.
          3) Execute `reps` repeated trials using either:
             - real text generation (run_client_generation) if consts.TARP_ENGINE == "real", or
             - the synthetic client request path (run_client_request) otherwise.

        """

    modes = ["naive", "g-trac", "sp", "mr", "larac"]

    for m in modes:
        try:
            requests.post(f"http://{anchor_ip}:{anchor_port}/reset_state", timeout=2)
            print(" Anchor state reset (trust + latency).")
            time.sleep(2.0)  # let gossip refresh LOCAL_CACHE
        except Exception as e:
            print(f"[Sweep] Reset failed: {e}")
            continue

        for k in range(reps):
            print(f"\n=== SWEEP mode={m} rep={k + 1}/{reps} ===")
            if consts.TARP_ENGINE == "real":
                run_client_generation(anchor_ip, anchor_port, m, prompt, max_new_tokens=n_tokens)
            else:
                run_client_request(anchor_ip, anchor_port, m)
            time.sleep(1.0)

#this for simulator
def run_client_request(anchor_ip: str, anchor_port: int, mode: str):
    """
        Simulated Request (Bytes -> Bytes).
        Used when TARP_ENGINE=sim.
        """
    if mode not in ("naive", "g-trac", "sp", "mr", "larac"):
        print("Mode must be: naive | g-trac | sp | mr | larac")
        return

    request_id = f"req-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    repair_attempted = 0
    repair_succeeded = 0
    repair_used = 0
    failed_error = ""
    failed_stage = ""
    t_req_start = utils.now_ms()

    #pass LOCAL_CACHE values as candidates
    candidates = list(LOCAL_CACHE.values())
    candidates = [c for c in candidates if utils.is_alive(c)]

    t_sel0 = time.perf_counter()
    chain = routing.select_chain(candidates, mode)
    t_sel1 = time.perf_counter()
    selection_overhead_ms = (t_sel1 - t_sel0) * 1000.0

    if not chain:
        print(
            f"{mode.upper()}: No feasible chain found. (Check workers cover layers 0..{MODEL_LAYERS - 1} contiguously; trust threshold for G-TRAC.)")
        return


    try:
        requests.post(
            f"http://{anchor_ip}:{anchor_port}/notify_active",
            json={"chain_ids": [n["id"] for n in chain]},
            timeout=0.5
        )
    except Exception:
        pass

    #sort chain by layer_start to enforce correct hop order
    chain = sorted(chain, key=lambda n: utils.node_layers(n)[0])

    #build path payload for workers
    path = []
    for n in chain:
        path.append({
            "id": n["id"],
            "ip": n["ip"],
            "port": int(n["port"]),
            "layer_start": int(n["layer_start"]),
            "layer_end": int(n["layer_end"]),
        })

    trace_id = f"trace-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    final_trace_id = trace_id

    activation = os.urandom(consts.TENSOR_SIZE_BYTES)

    entry = path[0]
    url = f"http://{entry['ip']}:{int(entry['port'])}/process"

    #print chain summary
    def fmt(n: Dict[str, Any]) -> str:
        ls, le = utils.node_layers(n)
        return f"{n['id']}[{ls}-{le}](t={utils.get_trust(n):.2f},lat={int(utils.get_lat_ms(n))}ms)"

    print("\n--- Request ---")
    print("Mode:", mode)
    print("Path:", " -> ".join(fmt(n) for n in chain))

    t0 = utils.now_ms()
    success = False
    failed_id: Optional[str] = None
    per_hop: List[Dict[str, Any]] = []

    try:
        r = requests.post(
            url,
            json={
                "trace_id": final_trace_id,
                "path": path,
                "hop_index": 0,
                "activation_b64": utils.b64e(activation),
            },
            timeout=consts.CLIENT_REQ_TIMEOUT_SEC,
        )
        elapsed = utils.now_ms() - t0
        try:
            j = r.json()
        except Exception:
            j = {"trace_id": trace_id, "failed_id": entry["id"], "error": "entry_invalid_json", "per_hop": []}

        per_hop = j.get("per_hop", []) if isinstance(j.get("per_hop", []), list) else []

        total_cpu_ms = sum(
            h.get("cpu_ms", 0.0) for h in per_hop
            if isinstance(h, dict) and isinstance(h.get("cpu_ms"), (int, float))
        )
        total_proc_ms = sum(
            h.get("proc_ms", 0.0) for h in per_hop
            if isinstance(h, dict) and isinstance(h.get("proc_ms"), (int, float))
        )

        print(f"Hop CPU time (sum): {total_cpu_ms:.2f} ms | Hop wall time (sum): {total_proc_ms:.2f} ms")

        if r.status_code == 200 and "final_activation_b64" in j:
            success = True
            print(f"SUCCESS in {elapsed:.0f} ms")
        else:
            failed_id = str(j.get("failed_id", entry["id"]))
            failed_error = str(j.get("error", "http_non_200_or_missing_final"))
            failed_stage = "pipeline"
            print(f"FAILURE in {elapsed:.0f} ms; failed_id={failed_id}; error={j.get('error')}")

            try:
                requests.post(
                    f"http://{anchor_ip}:{anchor_port}/feedback",
                    json={
                        "trace_id": final_trace_id,
                        "path_ids": [str(n["id"]) for n in chain],
                        "per_hop": per_hop,
                        "success": False,
                        "failed_id": failed_id,
                        "attribution": "failed_only",
                    },
                    timeout=2.0
                )
            except Exception:
                pass

    # except Exception:
    #    elapsed = now_ms() - t0
    #    failed_id = entry["id"]
    #    print(f"FAILURE: timeout after {elapsed:.0f} ms; failed_id={failed_id}")
    except Exception as e:
        elapsed = utils.now_ms() - t0

        #not assume entry failed. Probe it.
        failed_error = f"exception:{type(e).__name__}"
        entry_alive = probe_worker_reachable(entry, timeout=0.5)

        if not entry_alive:
            failed_id = entry["id"]
            failed_stage = "entry"
            print(f"FAILURE: timeout after {elapsed:.0f} ms; entry unreachable; failed_id={failed_id}")
        else:
            failed_id = None  # unknown/downstream
            failed_stage = "downstream_or_slow"
            print(
                f"FAILURE: timeout after {elapsed:.0f} ms; entry alive; failure likely downstream/slow; no failed_id attribution.")

    #optional one-shot hop repair
    if (not success) and consts.REPAIR_ON_FAILURE and failed_id:
        failed_node = next((n for n in chain if str(n.get("id")) == str(failed_id)), None)
        if failed_node:
            ls, le = utils.node_layers(failed_node)
            repl = routing.find_replacement_for_failed_hop(failed_id, ls, le, mode)
            if repl:
                repair_attempted = 1
                repair_used = 1
                print(f"REPAIR: replacing {failed_id}[{ls}-{le}] -> {repl['id']}[{ls}-{le}] and retrying once.")
                #replace and retry once (same trace_id ok, but new is fine)
                chain2 = [repl if str(n.get("id")) == str(failed_id) else n for n in chain]
                LOCAL_CACHE[str(repl["id"])] = repl  # ensure in cache

                #build new ordered path
                chain2 = sorted(chain2, key=lambda n: utils.node_layers(n)[0])
                path2 = [{
                    "id": n["id"], "ip": n["ip"], "port": int(n["port"]),
                    "layer_start": int(n["layer_start"]), "layer_end": int(n["layer_end"])
                } for n in chain2]
                entry2 = path2[0]
                url2 = f"http://{entry2['ip']}:{int(entry2['port'])}/process"

                trace_id2 = f"{trace_id}-repair"
                final_trace_id = trace_id2

                t1 = utils.now_ms()
                try:
                    rr = requests.post(
                        url2,
                        json={
                            "trace_id": final_trace_id,
                            "path": path2,
                            "hop_index": 0,
                            "activation_b64": utils.b64e(activation),
                        },
                        timeout=consts.CLIENT_REQ_TIMEOUT_SEC,
                    )
                    elapsed2 = utils.now_ms() - t1
                    try:
                        jj = rr.json()
                    except Exception:
                        jj = {"trace_id": trace_id2, "failed_id": entry2["id"], "error": "entry_invalid_json",
                              "per_hop": []}

                    per_hop = jj.get("per_hop", []) if isinstance(jj.get("per_hop", []), list) else []
                    total_cpu_ms = sum(
                        h.get("cpu_ms", 0.0) for h in per_hop
                        if isinstance(h, dict) and isinstance(h.get("cpu_ms"), (int, float))
                    )
                    total_proc_ms = sum(
                        h.get("proc_ms", 0.0) for h in per_hop
                        if isinstance(h, dict) and isinstance(h.get("proc_ms"), (int, float))
                    )

                    print(f"Hop CPU time (sum): {total_cpu_ms:.2f} ms | Hop wall time (sum): {total_proc_ms:.2f} ms")

                    if rr.status_code == 200 and "final_activation_b64" in jj:
                        success = True
                        repair_succeeded = 1
                        failed_id = None
                        failed_error = ""
                        failed_stage = ""
                        chain = chain2

                        print(f"REPAIR SUCCESS in {elapsed2:.0f} ms")
                    else:
                        failed_id = str(jj.get("failed_id", entry2["id"]))
                        failed_error = str(jj.get("error", "repair_http_non_200_or_missing_final"))
                        failed_stage = "repair_pipeline"
                        chain = chain2

                        print(f"REPAIR FAILURE in {elapsed2:.0f} ms; failed_id={failed_id}; error={jj.get('error')}")
                except Exception as e:
                    chain = chain2
                    failed_id = entry2["id"]
                    failed_error = f"repair_exception:{type(e).__name__}"
                    failed_stage = "repair_exception"

                    print("REPAIR FAILURE: timeout/unreachable")
            else:
                print(f"REPAIR: no replica found for layer segment [{ls}-{le}].")

    #send feedback to anchor
    try:
        chain_hops = len(path)

        # worst-case wait is dominated by forward timeouts across hops
        safe_client_timeout = (chain_hops - 1) * consts.FWD_REQ_TIMEOUT_SEC + 5.0  # + margin

        timeout_to_use = max(consts.CLIENT_REQ_TIMEOUT_SEC, safe_client_timeout)
        requests.post(
            f"http://{anchor_ip}:{anchor_port}/feedback",
            json={
                "trace_id": final_trace_id,
                "path_ids": [str(n["id"]) for n in chain],
                "per_hop": per_hop,
                "success": bool(success),
                "failed_id": failed_id,
                "attribution": "failed_only",
            },
            timeout=timeout_to_use,
        )
    except Exception:
        pass

    min_tr, mean_tr = _chain_trust_stats(chain)
    elapsed = utils.now_ms() - t_req_start

    utils.log_row(consts.REQUEST_LOG_PATH, consts.REQUEST_FIELDS, {
        "run_id": consts.RUN_ID, "ts_unix": utils.unix_ts(),
        "mode": mode, "engine": consts.TARP_ENGINE, "model": consts.MODEL_NAME,
        "request_id": request_id,
        "prompt_len_tokens": "", "target_new_tokens": "",
        "generated_new_tokens": 1 if success else 0,  # sim request produces one output blob
        "completed": 1 if success else 0,
        "request_success": 1 if success else 0,
        "request_e2e_ms": ("" if elapsed is None else round(float(elapsed), 3)),
        "selection_overhead_ms": round(selection_overhead_ms, 3),
        "repair_used": int(repair_used),
        "repair_attempted": int(repair_attempted),
        "repair_succeeded": int(repair_succeeded),
        "failed_id": failed_id if failed_id is not None else "unknown",
        "failed_error": failed_error,
        "failed_stage": failed_stage,
        "chain_ids": _chain_ids(chain),
        "chain_layers": _chain_layers(chain),
        "client_rss_mb": round(utils.client_rss_mb(), 2),
        "trust_tau": consts.TRUST_MIN_PER_HOP,
        "trust_min_in_chain": ("" if math.isnan(min_tr) else round(min_tr, 3)),
        "trust_mean_in_chain": ("" if math.isnan(mean_tr) else round(mean_tr, 3)),
    })
    log_trust_snapshot(mode, request_id, chain, failed_id)


def run_client_generation(anchor_ip: str, anchor_port: int, mode: str, prompt_text: str, max_new_tokens: int = 50):
    """
    Client-side driver for distributed, token-by-token generation over a multi-hop worker chain.

    Workflow:
      1) Reads worker metadata from LOCAL_CACHE and selects a feasible layer chain via routing.select_chain(..., mode).
      2) Notifies the anchor of the active chain (best-effort).
      3) For each token step:
         - Sends a /process request to the first worker in the chain with a 'path' describing all hops.
         - Receives the final activation (Heavy mode) or generated text (Light mode).
         - Prints tokens as they arrive and records per-token latency + per-hop metrics.
         - Sends per-step feedback to the anchor (/feedback), even on failures (best-effort).

    Modes:
      - Heavy mode (HAS_TORCH=True): encodes prompt with a local tokenizer and exchanges activations (base64 tensors).
      - Light mode (HAS_TORCH=False): sends raw text context and receives generated text (no local torch/tokenizer).

    Side effects:
      - Writes per-token traces to CSV under consts.LOG_DIR.
      - Appends a request-level summary row to consts.REQUEST_LOG_PATH.
    """

    #sampling paramters to make the prompt results more fun :D
    TEMPERATURE = 0.7  # 0.7 is balanced. 1.0 is crazy. 0.1 is robotic.
    TOP_K = 50  # Only choose from the top 50 best words.
    #-------------------------------------------------------
    if mode not in ("naive", "g-trac",  "sp", "mr", "larac"):
        print("Mode must be: naive | g-trac | sp | mr | larac")
        return
    request_id = f"req-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"

    #lazy tokenizer init
    global TOKENIZER, MODEL_LAYERS, GLOBAL_CONFIG
    # --- 1. SETUP (Heavy vs Light) ---
    if HAS_TORCH:
        # HEAVY SETUP: Initialize Tokenizer
        if TOKENIZER is None:
            TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)
        if GLOBAL_CONFIG is None:
            GLOBAL_CONFIG = AutoConfig.from_pretrained(consts.MODEL_NAME)
            total = int(getattr(GLOBAL_CONFIG, "n_layer", 0))
            if total > 0: MODEL_LAYERS = total

        # Prepare Initial Tensor
        generated = TOKENIZER.encode(prompt_text, return_tensors="pt")
        prompt_len_tokens = int(generated.shape[1])
        print(f"[Client] Heavy Mode: Prompt encoded to {prompt_len_tokens} tokens.")
    else:
        # LIGHT SETUP: Prepare String Context
        current_context = prompt_text
        prompt_len_tokens = len(prompt_text.split())  # Approx
        print(f"[Client] Light Mode: Sending raw text ({prompt_len_tokens} words).")
        if MODEL_LAYERS <= 0: MODEL_LAYERS = 36
    #if TOKENIZER is None:
    #    TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)
    consts.MODEL_LAYERS = MODEL_LAYERS
    ##Only load config one
    #if GLOBAL_CONFIG is None:
    #    print("[Client] Loading Config (one-time)...")
        #local_files_only=False allows download first time, but we cache the object in RAM
    #    GLOBAL_CONFIG = AutoConfig.from_pretrained(consts.MODEL_NAME)

        #update global MODEL_LAYERS only once
    #    total = int(getattr(GLOBAL_CONFIG, "n_layer", 0))
    #    if total > 0:
    #        MODEL_LAYERS = total
    #        print(f"[Client] Detected {MODEL_LAYERS} layers from config.")

    # Selection
    candidates = list(LOCAL_CACHE.values())
    # --- FIX: SANITIZE DATA TYPES ---
    # JSON sometimes sends numbers as strings. We must force them to ints/floats.
    clean_candidates = []
    for c in candidates:
        try:
            # Force Layers to Int
            c['layer_start'] = int(c.get('layer_start', -1))
            c['layer_end'] = int(c.get('layer_end', -1))

            # Force Trust/Port to Float/Int
            c['trust'] = float(c.get('trust', 0))
            c['port'] = int(c.get('port', 80))

            clean_candidates.append(c)
        except ValueError:
            print(f"[Client] Warning: Skipping malformed worker data: {c.get('id')}")
            continue

    candidates = clean_candidates

    #--------------------------------
    # DEBUG BLOCK
    print(f"\n[DEBUG] Target MODEL_LAYERS: 0 to {MODEL_LAYERS - 1}")

    # 1. Check raw cache
    print(f"[DEBUG] Total Workers in Cache: {len(LOCAL_CACHE)}")

    # 2. Check Liveness (Clock Sync Issue?)
    for c in candidates:
        is_alive = utils.is_alive(c)
        status = "ALIVE" if is_alive else "DEAD (Check Clocks!)"
        # Print details
        print(
            f" - Worker {c.get('id')}: Layers [{c.get('layer_start')}-{c.get('layer_end')}] | {status} | Trust={c.get('trust', 0)}")


    print(f"[DEBUG] forcing usage of all {len(candidates)} workers.")
    # -----------------------------------

    # Sort by layer start to see the sequence clearly
    candidates.sort(key=lambda x: x['layer_start'])
    print(f"\n[DEBUG] Target: 0 to {MODEL_LAYERS - 1} | Available Workers:")
    for c in candidates:
        print(f" - {c['id']}: [{c['layer_start']} - {c['layer_end']}]")
    t_sel0= time.perf_counter()
    chain = routing.select_chain(candidates, mode)
    t_sel1 = time.perf_counter()
    selection_overhead_ms = (t_sel1 - t_sel0) * 1000.0
    if not chain:
        print(
            f"{mode.upper()}: No feasible chain found. (Check workers cover layers 0..{MODEL_LAYERS - 1} contiguously; trust threshold for G-TRAC.)")
        return
    try:
        requests.post(
            f"http://{anchor_ip}:{anchor_port}/notify_active",
            json={"chain_ids": [str(n["id"]) for n in chain]},
            timeout=0.5
        )
    except Exception:
        pass
    chain = sorted(chain, key=lambda n: utils.node_layers(n)[0])

    #build path payload for workers
    path = [{
        "id": n["id"],
        "ip": n["ip"],
        "port": int(n["port"]),
        "layer_start": int(n["layer_start"]),
        "layer_end": int(n["layer_end"]),
    } for n in chain]

    #print chain summary
    def fmt(n: Dict[str, Any]) -> str:
        ls, le = utils.node_layers(n)
        return f"{n['id']}[{ls}-{le}](t={utils.get_trust(n):.2f},lat={int(utils.get_lat_ms(n))}ms)"

    print("\n--- Generation ---")
    print("Mode:", mode)
    print("Path:", " -> ".join(fmt(n) for n in chain))
    print("Prompt:", prompt_text)
    print("Output:", end=" ", flush=True)

    #generated = TOKENIZER.encode(prompt_text, return_tensors="pt")
    #prompt_len_tokens = int(generated.shape[1])


    # trace average/p99/max/min latency
    t_req0 = time.perf_counter()
    token_lat_ms: List[float] = []
    token_success: int = 0
    token_fail: int = 0

    token_cpu_ms: List[float] = []
    token_hop_wall_ms: List[float] = []  #sum of hop proc_ms per token
    token_rows: List[Dict[str, Any]] = []
    failed_id: Optional[str] = None

    for step in range(max_new_tokens):
        entry = path[0]
        url = f"http://{entry['ip']}:{int(entry['port'])}/process"
        trace_id = f"gen-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"

        t0 = utils.now_ms()
        success = False
        failed_id = None
        per_hop: List[Dict[str, Any]] = []
        new_word = ""
        stop_generation = False  #flag for EOS
        try:
            #PREPARE PAYLOAD
            payload = {
                "trace_id": trace_id,
                "path": path,
                "hop_index": 0,
                "activation_b64": ""
            }

            if HAS_TORCH:
                #HEAVY: Send Tensor Bytes
                payload["activation_b64"] = utils.tensor_to_b64(generated)
            else:
                #LIGHT: Send Raw Text
                payload["prompt"] = current_context
                payload["activation_b64"] = None

            #SEND REQUEST
            chain_hops = len(path)
            #worst-case wait is dominated by forward timeouts across hops
            safe_client_timeout = (chain_hops - 1) * consts.FWD_REQ_TIMEOUT_SEC + 5.0
            timeout_to_use = max(consts.CLIENT_REQ_TIMEOUT_SEC, safe_client_timeout)

            r = requests.post(
                url,
                json=payload,
                timeout=timeout_to_use,
            )
            elapsed = utils.now_ms() - t0
            try:
                j = r.json()
            except Exception:
                j = {"trace_id": trace_id, "failed_id": entry["id"], "error": "entry_invalid_json", "per_hop": []}

            per_hop = j.get("per_hop", []) if isinstance(j.get("per_hop", []), list) else []

            #sum CPU time and hop wall time across all workers for this token
            step_cpu_ms = sum(
                h.get("cpu_ms", 0.0) for h in per_hop
                if isinstance(h, dict) and isinstance(h.get("cpu_ms"), (int, float))
            )
            step_hop_wall_ms = sum(
                h.get("proc_ms", 0.0) for h in per_hop
                if isinstance(h, dict) and isinstance(h.get("proc_ms"), (int, float))
            )
            step_max_rss = max(
                (h.get("rss_mb", 0.0) for h in per_hop if isinstance(h.get("rss_mb"), (int, float))),
                default=0.0
            )

            #HANDLE RESPONSE
            if r.status_code == 200:
                #Case 1: heavy resposne (tensor)
                if HAS_TORCH :
                    logits = utils.b64_to_tensor(j["final_activation_b64"])  # [batch, seq, vocab]
                    next_token_logits = logits[:, -1, :]
                    #--THIS IS OPTIONAL, CAN BE REMOVED------------------------------------------------
                    #apply temperature to make sentence randomness
                    next_token_logits = next_token_logits / TEMPERATURE
                    #then use top-k filtering to keep top k options
                    if TOP_K > 0:
                        top_k_vals, _ = torch.topk(next_token_logits, TOP_K)
                        min_val = top_k_vals[:, -1]
                        # Set everything below top-k to -infinity so we don't pick them
                        next_token_logits[next_token_logits < min_val] = -float('Inf')

                    #Calculate Probabilities (Softmax)
                    probs = torch.softmax(next_token_logits, dim=-1)

                    #Sample from the distribution (Instead of Argmax)
                    next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                    # --END OF OPTIONAL PART, CAN BE REMOVED------------------------------------------------

                    #if removed the above optional block, then uncomment the line right below
                    #next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
                    next_token = torch.tensor([[next_token_id]], dtype=generated.dtype)
                    new_word = TOKENIZER.decode([next_token_id])
                    generated = torch.cat([generated, next_token], dim=1)
                    # check for EOS
                    if next_token_id == TOKENIZER.eos_token_id:
                        stop_generation = True


                #Case 2: Light response -- text
                else :
                    new_word = j.get("generated_text", "")
                    current_context += new_word

                    # EOS Check for Light Mode (String check)
                    # Note!!!!!!!!!!!!!1! might need to adjust "<|endoftext|>" depending on model
                    if "<|endoftext|>" in new_word:
                        stop_generation = True



                print(new_word, end="", flush=True)

                token_lat_ms.append(float(elapsed))
                token_cpu_ms.append(float(step_cpu_ms))
                token_hop_wall_ms.append(float(step_hop_wall_ms))  # optional
                token_success += 1
                success = True
                step_client_rss = utils.rss_mb()
                token_rows.append({
                    "token_idx": step,  # 0-based; use step+1 if you prefer 1-based
                    "trace_id": trace_id,
                    "e2e_latency_ms": float(elapsed),  #end-to-end client-perceived latency for this token
                    "sum_hop_cpu_ms": float(step_cpu_ms),  #sum of worker (user+sys) CPU across hops
                    "sum_hop_wall_ms": float(step_hop_wall_ms),  #sum of per-hop proc_ms
                    "num_hops": int(len(per_hop)),
                    "max_hop_rss_mb": float(step_max_rss),
                    "client_rss_mb": float(step_client_rss),
                })
                if stop_generation:
                    print("\n[EOS reached]")
                    # break # Uncomment if you want to stop early

            else:
                failed_id = str(j.get("failed_id", entry["id"]))
                print(f"\nFAILURE in {elapsed:.0f} ms; failed_id={failed_id}; error={j.get('error')}")
                token_fail += 1
                break

        except Exception as e:
            token_fail += 1
            elapsed = utils.now_ms() - t0
            entry_alive = probe_worker_reachable(entry, timeout=0.5)
            if not entry_alive:
                failed_id = entry["id"]
                print(f"\nFAILURE: exception after {elapsed:.0f} ms; entry unreachable; failed_id={failed_id}; ex={e}")
            else:
                failed_id = None
                print(
                    f"\nFAILURE: exception after {elapsed:.0f} ms; entry alive; failure likely downstream/slow; ex={e}")
            break

        finally:
            #Feedback each step
            try:
                requests.post(
                    f"http://{anchor_ip}:{anchor_port}/feedback",
                    json={
                        "trace_id": trace_id,
                        "path_ids": [str(n["id"]) for n in chain],
                        "per_hop": per_hop,
                        "success": bool(success),
                        "failed_id": failed_id,
                        "attribution": "failed_only",
                    },
                    timeout=max(consts.CTRL_TIMEOUT_SEC, 2.0),
                )
            except Exception:
                pass

    #write token log
    if token_rows:
        out_csv = os.path.join(consts.LOG_DIR, f"distributed_trace_{mode}_{consts.MODEL_NAME}_{int(time.time())}.csv")
        utils._ensure_header(out_csv, list(token_rows[0].keys()))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(token_rows[0].keys()))
            if f.tell() == 0: w.writeheader()
            w.writerows(token_rows)
        print(f"\n[Trace] Wrote per-token trace to: {out_csv}")
    else:
        print("\n[Trace] No token rows to write (no successful tokens).")

    # print statistic
    print("\n\n------------------------------------------------")
    print(f"\n\nRESULTS: Distributed {consts.MODEL_NAME} ({mode})")
    print("------------------------------------------------")
    print(f"Tokens requested:       {max_new_tokens}")
    print(f"Tokens succeeded:       {token_success}")
    print(f"Tokens failed:          {token_fail}")

    if token_lat_ms:
        s = sorted(token_lat_ms)
        avg = sum(s) / len(s)
        p99 = utils.percentile(s, 99.0)
        p95 = utils.percentile(s, 95.0)
        p50 = utils.percentile(s, 50.0)
        print(f"Average latency:        {avg:.2f} ms/token")
        print(f"P50 latency:            {p50:.2f} ms/token")
        print(f"P95 latency:            {p95:.2f} ms/token")
        print(f"P99 latency:            {p99:.2f} ms/token")
        print(f"Min latency:            {s[0]:.2f} ms/token")
        print(f"Max latency:            {s[-1]:.2f} ms/token")
    else:
        print("No successful tokens to report latency stats.")
    print("------------------------------------------------")
    #
    print("DEBUG token_cpu_ms len:", len(token_cpu_ms), flush=True)

    if token_cpu_ms:
        c = sorted(token_cpu_ms)
        avg_c = sum(c) / len(c)
        p99_c = utils.percentile(c, 99.0)
        p95_c = utils.percentile(c, 95.0)
        p50_c = utils.percentile(c, 50.0)
        print(f"Average CPU time:       {avg_c:.2f} ms/token (sum user+sys across hops)")
        print(f"P50 CPU time:           {p50_c:.2f} ms/token")
        print(f"P95 CPU time:           {p95_c:.2f} ms/token")
        print(f"P99 CPU time:           {p99_c:.2f} ms/token")
        print(f"Min CPU time:           {c[0]:.2f} ms/token")
        print(f"Max CPU time:           {c[-1]:.2f} ms/token")
    else:
        print("No successful tokens to report CPU stats.")

    print("\n")
    request_e2e_ms = (time.perf_counter() - t_req0) * 1000.0
    min_tr, mean_tr = _chain_trust_stats(chain)
    utils.log_row(consts.REQUEST_LOG_PATH, consts.REQUEST_FIELDS, {
        "run_id": consts.RUN_ID,
        "ts_unix": utils.unix_ts(),
        "mode": mode,
        "engine": consts.TARP_ENGINE,
        "model": consts.MODEL_NAME,
        "request_id": request_id,
        "prompt_len_tokens": prompt_len_tokens,
        "target_new_tokens": max_new_tokens,
        "generated_new_tokens": token_success,
        "completed": int(token_success == max_new_tokens),
        "request_success": int(token_fail == 0),
        "request_e2e_ms": round(request_e2e_ms, 3),
        "selection_overhead_ms": round(selection_overhead_ms, 3),
        "repair_used": "",  # optional
        "repair_attempted": "",  # optional
        "repair_succeeded": "",  # optional
        "failed_id": failed_id or "",
        "failed_error": "",
        "failed_stage": "",
        "chain_ids": _chain_ids(chain),
        "chain_layers": _chain_layers(chain),
        "client_rss_mb": round(utils.rss_mb(), 2),
        "trust_tau": consts.TRUST_MIN_PER_HOP,
        "trust_min_in_chain": ("" if math.isnan(min_tr) else round(min_tr, 3)),
        "trust_mean_in_chain": ("" if math.isnan(mean_tr) else round(mean_tr, 3)),
    })
    log_trust_snapshot(mode, request_id, chain, failed_id)

def run_client_generation_with_repair(anchor_ip: str, anchor_port: int, mode: str, prompt_text: str, max_new_tokens: int = 50):
    """Autoregressive GPT-2 generation (Robust: With retries/repair)."""

    if mode not in ("naive", "g-trac", "sp", "mr", "larac"):
        print("Mode must be: naive | g-trac | sp | mr | larac")
        return

    request_id = f"req-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    global TOKENIZER, MODEL_LAYERS, GLOBAL_CONFIG
    from transformers import AutoTokenizer, AutoConfig
    import torch

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)

    #load config once
    if GLOBAL_CONFIG is None:
        print("[Client] Loading Config (one-time)...")
        #local_files_only=False allows download first time, but we cache the object in RAM
        GLOBAL_CONFIG = AutoConfig.from_pretrained(consts.MODEL_NAME)

        #update global MODEL_LAYERS only once
        total = int(getattr(GLOBAL_CONFIG, "n_layer", 0))
        if total > 0:
            MODEL_LAYERS = total
            print(f"[Client] Detected {MODEL_LAYERS} layers from config.")

    candidates = list(LOCAL_CACHE.values())
    candidates = [c for c in candidates if utils.is_alive(c)]
    t_sel0 = time.perf_counter()
    chain = routing.select_chain(candidates, mode)
    t_sel1 = time.perf_counter()
    selection_overhead_ms = (t_sel1 - t_sel0) * 1000.0

    if not chain:
        print(f"{mode.upper()}: No feasible chain found.")
        return
    try:
        #notify Anchor to highlight these nodes on the dashboard
        requests.post(
            f"http://{anchor_ip}:{anchor_port}/notify_active",
            json={"chain_ids": [n["id"] for n in chain]},
            timeout=0.5  #fire and forget, don't block generation
        )
    except Exception:
        pass  #don't crash if dashboard is slow


    chain = sorted(chain, key=lambda n: utils.node_layers(n)[0])
    path = [{"id": n["id"], "ip": n["ip"], "port": int(n["port"]),
             "layer_start": int(n["layer_start"]), "layer_end": int(n["layer_end"])} for n in chain]

    print("\n--- Generation ---")
    print(f"Mode: {mode}\nPrompt: {prompt_text}\nOutput: ", end="", flush=True)

    generated = TOKENIZER.encode(prompt_text, return_tensors="pt")
    prompt_len_tokens = int(generated.shape[1])
    t_req0 = time.perf_counter()

    token_lat_ms, token_cpu_ms, token_hop_wall_ms, token_rows = [], [], [], []
    token_success, token_fail = 0, 0

    for step in range(max_new_tokens):
        token_done = False
        retries_per_token = 1

        while not token_done and retries_per_token >= 0:
            entry = path[0]
            url = f"http://{entry['ip']}:{int(entry['port'])}/process"
            trace_id = f"gen-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
            t0 = utils.now_ms()
            success = False
            failed_id = None
            per_hop = []

            try:
                chain_hops = len(path)
                safe_client_timeout = (chain_hops - 1) * consts.FWD_REQ_TIMEOUT_SEC + 5.0
                timeout_to_use = max(consts.CLIENT_REQ_TIMEOUT_SEC, safe_client_timeout)

                r = requests.post(
                    url,
                    json={
                        "trace_id": trace_id,
                        "path": path,
                        "hop_index": 0,
                        "activation_b64": utils.tensor_to_b64(generated),
                    },
                    timeout=timeout_to_use,
                )
                elapsed = utils.now_ms() - t0

                try:
                    j = r.json()
                except Exception:
                    j = {"trace_id": trace_id, "failed_id": entry["id"], "error": "entry_invalid_json", "per_hop": []}

                per_hop = j.get("per_hop", [])

                if r.status_code == 200 and "final_activation_b64" in j:
                    logits = utils.b64_to_tensor(j["final_activation_b64"])
                    next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    next_token = torch.tensor([[next_token_id]], dtype=generated.dtype)

                    print(TOKENIZER.decode([next_token_id]), end="", flush=True)
                    generated = torch.cat([generated, next_token], dim=1)

                    token_lat_ms.append(float(elapsed))
                    token_success += 1
                    success = True
                    token_done = True

                    # Record metrics
                    step_cpu_ms = sum(h.get("cpu_ms", 0.0) for h in per_hop if isinstance(h, dict))
                    token_cpu_ms.append(step_cpu_ms)
                    token_rows.append({
                        "token_idx": step, "trace_id": trace_id, "e2e_latency_ms": float(elapsed),
                        "sum_hop_cpu_ms": float(step_cpu_ms), "num_hops": len(per_hop),
                        "client_rss_mb": float(utils.rss_mb())
                    })
                else:
                    failed_id = str(j.get("failed_id", entry["id"]))
                    print(f"\nFAILURE at {failed_id}. Re-selecting...")
                    retries_per_token -= 1
                    #trigger re-selection for the next attempt of THIS token
                    new_chain = routing.select_chain(candidates, mode)
                    if new_chain:
                        chain = sorted(new_chain, key=lambda n: utils.node_layers(n)[0])
                        path = [{"id": n["id"], "ip": n["ip"], "port": int(n["port"]),
                                 "layer_start": n["layer_start"], "layer_end": n["layer_end"]} for n in chain]

            except Exception as e:
                print(f"\n[Token {step}] TIMEOUT/EXCEPTION: {e}. Reporting entry {entry['id']}...")
                failed_id = entry["id"]
                retries_per_token -= 1
                new_chain = routing.select_chain(candidates, mode)
                if new_chain:
                    chain = sorted(new_chain, key=lambda n: utils.node_layers(n)[0])
                    path = [{"id": n["id"], "ip": n["ip"], "port": int(n["port"]),
                             "layer_start": n["layer_start"], "layer_end": n["layer_end"]} for n in chain]

            finally:
                # Feedback to anchor (Success OR Failure)
                try:
                    requests.post(
                        f"http://{anchor_ip}:{anchor_port}/feedback",
                        json={
                            "trace_id": trace_id, "path_ids": [str(n["id"]) for n in chain],
                            "per_hop": per_hop, "success": bool(success),
                            "failed_id": failed_id, "attribution": "failed_only",
                        },
                        timeout=2.0,
                    )
                except Exception:
                    pass

        if not token_done:
            print(f"\n[Token {step}] Critical failure: Could not complete token after retries.")
            token_fail += 1
            break

    if token_rows:
        out_csv = os.path.join(consts.LOG_DIR, f"distributed_trace_{mode}_{consts.MODEL_NAME}_{int(time.time())}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "token_idx", "trace_id", "e2e_latency_ms", "sum_hop_cpu_ms", "sum_hop_wall_ms", "num_hops",
                "max_hop_rss_mb", "client_rss_mb"
            ])
            w.writeheader()
            w.writerows(token_rows)

        print(f"[Trace] Wrote per-token trace to: {out_csv}")
    else:
        print("[Trace] No token rows to write (no successful tokens).")

    # print statistic...
    print("\n\n------------------------------------------------")
    print(f"RESULTS: Distributed {consts.MODEL_NAME} ({mode})")
    print("------------------------------------------------")
    print(f"Tokens requested:       {max_new_tokens}")
    print(f"Tokens succeeded:       {token_success}")
    print(f"Tokens failed:          {token_fail}")

    if token_lat_ms:
        s = sorted(token_lat_ms)
        avg = sum(s) / len(s)
        p99 = utils.percentile(s, 99.0)
        p95 = utils.percentile(s, 95.0)
        p50 = utils.percentile(s, 50.0)
        print(f"Average latency:        {avg:.2f} ms/token")
        print(f"P50 latency:            {p50:.2f} ms/token")
        print(f"P95 latency:            {p95:.2f} ms/token")
        print(f"P99 latency:            {p99:.2f} ms/token")
        print(f"Min latency:            {s[0]:.2f} ms/token")
        print(f"Max latency:            {s[-1]:.2f} ms/token")
    else:
        print("No successful tokens to report latency stats.")
    print("------------------------------------------------")
    #
    print("DEBUG token_cpu_ms len:", len(token_cpu_ms), flush=True)

    if token_cpu_ms:
        c = sorted(token_cpu_ms)
        avg_c = sum(c) / len(c)
        p99_c = utils.percentile(c, 99.0)
        p95_c = utils.percentile(c, 95.0)
        p50_c = utils.percentile(c, 50.0)
        print(f"Average CPU time:       {avg_c:.2f} ms/token (sum user+sys across hops)")
        print(f"P50 CPU time:           {p50_c:.2f} ms/token")
        print(f"P95 CPU time:           {p95_c:.2f} ms/token")
        print(f"P99 CPU time:           {p99_c:.2f} ms/token")
        print(f"Min CPU time:           {c[0]:.2f} ms/token")
        print(f"Max CPU time:           {c[-1]:.2f} ms/token")
    else:
        print("No successful tokens to report CPU stats.")

    print("\n")

    request_e2e_ms = (time.perf_counter() - t_req0) * 1000.0
    min_tr, mean_tr = _chain_trust_stats(chain)
    utils.log_row(consts.REQUEST_LOG_PATH, consts.REQUEST_FIELDS, {
        "run_id": consts.RUN_ID,
        "ts_unix": utils.unix_ts(),
        "mode": mode,
        "engine": consts.TARP_ENGINE,
        "model": consts.MODEL_NAME,
        "request_id": request_id,
        "prompt_len_tokens": prompt_len_tokens,
        "target_new_tokens": max_new_tokens,
        "generated_new_tokens": token_success,
        "completed": int(token_success == max_new_tokens),
        "request_success": int(token_fail == 0),
        "request_e2e_ms": round(request_e2e_ms, 3),
        "selection_overhead_ms": round(selection_overhead_ms, 3),
        "repair_used": "",  # optional
        "repair_attempted": "",  # optional
        "repair_succeeded": "",  # optional
        "failed_id": failed_id or "",
        "failed_error": "",
        "failed_stage": "",
        "chain_ids": _chain_ids(chain),
        "chain_layers": _chain_layers(chain),
        "client_rss_mb": round(utils.rss_mb(), 2),
        "trust_tau": consts.TRUST_MIN_PER_HOP,
        "trust_min_in_chain": ("" if math.isnan(min_tr) else round(min_tr, 3)),
        "trust_mean_in_chain": ("" if math.isnan(mean_tr) else round(mean_tr, 3)),
    })
    log_trust_snapshot(mode, request_id, chain, failed_id)



def start_client(anchor_ip, anchor_port, mode):
    """
        Interactive client CLI entrypoint.

        - Spawns a background gossip thread (client_gossip_loop) to keep LOCAL_CACHE updated and
          maintain ANCHOR_REACHABLE.
        - Provides a simple REPL for running requests/generation against the selected routing mode:
            [r] run    : execute one request (real generation or synthetic depending on consts.TARP_ENGINE)
            [m] mode   : set routing/selection mode (naive|g-trac|sp|mr|larac)
            [s] reset  : reset anchor-side state (/reset_state), then wait for gossip to refresh
            [w] sweep  : run multi-mode experiment sweep (run_sweep)
            [q] quit   : exit

        Notes:
          - `mode` is updated in-place based on user input.
          - Sleeps provide settling time for anchor reset + cache convergence.
        """

    # Start Gossip Thread
    t = threading.Thread(target=client_gossip_loop, args=(anchor_ip, anchor_port), daemon=True)
    t.start()
    # Give it a second to connect
    time.sleep(1.0)
    while True:
        try:
            cmd = input("\n[r] run  [m] mode  [s] reset  [w] sweep  [q] quit: ").strip().lower()
        except EOFError:
            break  #Ctrl+D

        if cmd == "q":
            break
        if cmd == "m":
            mode2 = input("Mode (naive|g-trac|sp|mr|larac): ").strip().lower()
            if mode2 in ("naive", "g-trac", "sp", "mr",  "larac"):
                mode = mode2
                print("Mode set to:", mode)
            else:
                print("Invalid mode.")
            continue
        if cmd == "r":
            if consts.TARP_ENGINE == "real":
                prompt_text = input("Prompt: ").strip()
                if not prompt_text:
                    prompt_text = "Hello"
                try:
                    n = int(input("Max new tokens [50]: ").strip() or "50")
                except Exception:
                    n = 50
                run_client_generation(anchor_ip, anchor_port, mode, prompt_text, max_new_tokens=n)
            else:
                run_client_request(anchor_ip, anchor_port, mode)
        if cmd == "s":
            print("[Client] Resetting anchor state...")
            try:
                requests.post(
                    f"http://{anchor_ip}:{anchor_port}/reset_state",
                    timeout=2,
                )
                print("[Client] Anchor state reset (trust + latency).")
                time.sleep(2.0)
            except Exception as e:
                print(f"[Client] Reset failed: {e}")
            continue
        if cmd == "w":
            prompt_text = input("Prompt: ").strip() or "Hello there"
            n = int(input("Max new tokens [50]: ").strip() or "50")
            reps = int(input("Reps per mode [3]: ").strip() or "3")
            run_sweep(anchor_ip, anchor_port, prompt_text, n_tokens=n, reps=reps)
            continue