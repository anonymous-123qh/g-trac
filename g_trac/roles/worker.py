from flask import Flask, jsonify, request
import threading
import time
import requests
import random
import traceback
import hashlib
from .. import consts, utils, model_shard
import torch
from transformers import AutoConfig, AutoTokenizer
app = Flask(__name__)
MY_CONFS = {}  #for multi-worker
MY_CONF = {}       #fallback for single-worker mode
MODEL_SHARD = None
TOKENIZER = None #for light mode

#--Helper ---
def transform_activation(inp: bytes, worker_id: str, hop_index: int) -> bytes:
    """
    Simulation mode only: Simple hash transformation to simulate
    processing changing the data.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(inp)
    h.update(worker_id.encode("utf-8"))
    h.update(str(hop_index).encode("utf-8"))
    seed = h.digest()

    out = bytearray()
    counter = 0
    while len(out) < consts.TENSOR_SIZE_BYTES:
        hh = hashlib.blake2b(digest_size=32)
        hh.update(seed)
        hh.update(counter.to_bytes(4, "little"))
        out.extend(hh.digest())
        counter += 1
    return bytes(out[:consts.TENSOR_SIZE_BYTES])


# --- Routes ---
@app.route("/status", methods=["GET"])
def status():
    #backward compatible:
    # - single-id worker returns {"id":..., ...}
    # - multi-id worker returns {"peers":[{...},{...}], "num_peers":N, "layers":[ls,le]}
    if MY_CONFS:
        #all peers share same layer range by design
        any_conf = next(iter(MY_CONFS.values()))
        return jsonify({
            "mode": "multi",
            "num_peers": len(MY_CONFS),
            "layers": [int(any_conf["layer_start"]), int(any_conf["layer_end"])],
            "peers": list(MY_CONFS.values()),
        })
    return jsonify(MY_CONF)

@app.route("/process", methods=["POST"])
def process():
    """
        Request JSON schema:
          {
            "path": [...],
            "hop_index": <int>,
            "activation_b64": "<base64 bytes>",  (Optional if prompt is present)
            "prompt": "<string>",                (Optional, for Light Mode)
            "trace_id": "<string>",
            "per_hop": [ ... ]
          }
    """

    global MODEL_SHARD, TOKENIZER
    req = request.get_json(silent=True) or {}
    # --- DEBUG: inspect incoming payload (protocol + base64 sanity) ---
    print(f"[process] IN from={request.remote_addr} keys={list(req.keys())}")
    prompt = req.get("prompt", None)  #Capture text input
    act_b64 = req.get("activation_b64", "")
    if act_b64:
        print(f"[process] IN activation_b64 missing/not-str type={type(act_b64)}")

    if prompt:
        print(f"[process] IN prompt len={len(prompt)}")
    # ---------------------------------------------

    trace_id = str(req.get("trace_id", "trace-unknown"))
    path = req.get("path", [])
    hop_index = int(req.get("hop_index", 0))


    my_id = str(MY_CONF.get("id", "unknown"))

    # Basic validation
    if not isinstance(path, list) or hop_index < 0 or hop_index >= len(path):
        return jsonify({"trace_id": trace_id, "failed_id": my_id, "error": "bad_request"}), 400
    if not act_b64 and not prompt:
        return jsonify({"trace_id": trace_id, "failed_id": my_id, "error": "missing_activation_or_prompt"}), 400

    # Determine which "peer identity" this hop is supposed to be
    expected_id = str(path[hop_index].get("id", ""))

    # Multi-id mode: pick conf by expected_id
    if MY_CONFS:
        if not expected_id or expected_id not in MY_CONFS:
            return jsonify({
                "trace_id": trace_id,
                "failed_id": expected_id or "unknown",
                "error": "unknown_peer_id_on_this_worker",
            }), 400
        conf = MY_CONFS[expected_id]
        my_id = expected_id
    else:
        # Single-id mode: legacy behavior
        conf = MY_CONF
        my_id = str(MY_CONF.get("id", "unknown"))

        # Protocol consistency: this hop must match this worker id
        if expected_id and expected_id != my_id:
            return jsonify({
                "trace_id": trace_id,
                "failed_id": my_id,
                "error": "protocol_mismatch",
                "expected": expected_id,
                "actual": my_id
            }), 400

    #TRACE START (hop identity)
    try:
        my_ls = int(conf.get("layer_start", -1))
        my_le = int(conf.get("layer_end", -1))
    except Exception:
        my_ls, my_le = -1, -1

    nxt_id = None
    nxt_ip = None
    nxt_port = None
    if isinstance(path, list) and 0 <= hop_index < len(path) - 1:
        nxt = path[hop_index + 1]
        nxt_id = str(nxt.get("id", ""))
        nxt_ip = str(nxt.get("ip", ""))
        try:
            nxt_port = int(nxt.get("port", 0))
        except Exception:
            nxt_port = None

    t_req_in = time.perf_counter()
    print(f"[WKR] START trace={trace_id} hop={hop_index}/{len(path) - 1} "
          f"id={my_id}[{my_ls}-{my_le}] rss={utils.rss_mb():.1f}MB "
          f"next={nxt_id}@{nxt_ip}:{nxt_port}",
          flush=True)
    # -----------------------------------

    # Start wall clock (captures network delay and compute)
    t0_wall = utils.now_ms()
    # Optional network delay emulation (per peer)
    try:
        nd = float(conf.get("net_delay_ms", 0))
    except Exception:
        nd = 0.0
    if nd > 0:
        time.sleep(nd / 1000.0)

    per_hop = req.get("per_hop", [])
    # start cpu clock to capture compute time
    t0_cpu = utils.cpu_time_ms()
    if not isinstance(per_hop, list):
        per_hop = []

    # Simulate failure
    if random.random() < float(conf.get("fail_rate", 0.0)):
        time.sleep(consts.FWD_REQ_TIMEOUT_SEC + 1.0)

        per_hop.append({"id": my_id, "layer_start": int(conf["layer_start"]),
                        "layer_end": int(conf["layer_end"]), "proc_ms": None, "cpu_ms": None,
                        "rss_mb": round(utils.rss_mb(), 2), "ok": False})
        return jsonify({"trace_id": trace_id, "failed_id": my_id, "per_hop": per_hop, "error": "local_fail"}), 500


    #Input handling (either text or tensor)
    input_tensor = None

    #Simulator mode
    if consts.TARP_ENGINE != "real":
        try:
            if act_b64:
                inp = utils.b64d(act_b64)
            elif prompt:
                inp = prompt.encode("utf-8")  # Fake input for sim
            else:
                raise ValueError("No Input")

            #simulated inference mode
            utils.burn_cpu(int(conf.get("cpu_load", 0)))
            out = transform_activation(inp, my_id, hop_index)
            out_b64 = utils.b64e(out)
        except Exception:
            per_hop.append({"id": my_id, "layer_start": int(conf["layer_start"]),
                            "layer_end": int(conf["layer_end"]), "proc_ms": None, "cpu_ms": None,
                            "rss_mb": round(utils.rss_mb(), 2), "ok": False})

            return jsonify({"trace_id": trace_id, "failed_id": my_id, "per_hop": per_hop, "error": "sim_fail"}), 400

    else:
        #real inference mode
        try:
            #Case 1: light mode (text input) only happen at hop 0
            # Priority 1: If we have a prompt AND we are the first hop, TOKENIZE
            if prompt and (not act_b64 or len(act_b64) < 10) and int(conf.get("layer_start", -1)) == 0:
                print(f"[WKR] Light Mode Entry: Tokenizing '{prompt}'")
                #use the MODEL_SHARD instance created in start_worker

                if TOKENIZER is None:
                    TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)


                #tokenizer = model_shard.get_tokenizer()
                input_ids = TOKENIZER.encode(prompt, return_tensors="pt")
                input_tensor = input_ids.to(consts.DEVICE)
                #print(f"[WKR] Tokenized prompt: shape={input_tensor.shape}")
            #Case 2:  standard mode (tensor input)
            elif act_b64 and len(act_b64) > 10:
                input_tensor = utils.b64_to_tensor(act_b64)
            else:
                raise ValueError("No valid input (empty tensor and not an entry prompt)")


        except Exception as e:
            print(f"[process] b64_to_tensor FAILED on {my_id}: {type(e).__name__}: {e}")
            print(f"[process] Input Conversion FAILED on {my_id}: {e}")

            # stop timers for consistency (optional)
            t1_wall = utils.now_ms()
            t1_cpu = utils.cpu_time_ms()
            per_hop.append({
                "id": my_id,
                "layer_start": int(conf["layer_start"]),
                "layer_end": int(conf["layer_end"]),
                "proc_ms": round(t1_wall - t0_wall, 2),
                "cpu_ms": round(t1_cpu - t0_cpu, 2),
                "rss_mb": round(utils.rss_mb(), 2),
                "ok": False
            })
            return jsonify({
                "trace_id": trace_id,
                "failed_id": my_id,
                "per_hop": per_hop,
                "error": f"input_conversion_failed: {type(e).__name__}"
            }), 400

        #Now do inference
        try:
            if input_tensor is None:
                raise ValueError("input_tensor is None before forward pass")

            utils.burn_cpu(int(conf.get("cpu_load", 0)))  # Add CPU Load simulation to Real inference too
            output_tensor = MODEL_SHARD.forward(input_tensor)
        except Exception as e:
            # --- DEBUGGING: Print full error to Worker Console ---
            print(f"!!! INFERENCE ERROR on {my_id} !!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # -----------------------------------------------------
            t1_wall = utils.now_ms()
            t1_cpu = utils.cpu_time_ms()
            per_hop.append({
                "id": my_id,
                "layer_start": int(conf["layer_start"]),
                "layer_end": int(conf["layer_end"]),
                "proc_ms": round(t1_wall - t0_wall, 2),
                "cpu_ms": round(t1_cpu - t0_cpu, 2),
                "rss_mb": round(utils.rss_mb(), 2),
                "ok": False
            })
            return jsonify({"trace_id": trace_id, "failed_id": my_id, "per_hop": per_hop,
                            "error": f"inference_exception:{e}"}), 500

        # proc_ms = now_ms() - t0
        #prepare output for next step, which is tensor
        out_b64 = utils.tensor_to_b64(output_tensor)

    #Recorded metrics
    t1_wall = utils.now_ms()
    t1_cpu = utils.cpu_time_ms()
    proc_ms = t1_wall - t0_wall
    cpu_ms = t1_cpu - t0_cpu
    per_hop.append({
        "id": my_id,
        "layer_start": int(conf["layer_start"]),
        "layer_end": int(conf["layer_end"]),
        "proc_ms": round(proc_ms, 2),
        "cpu_ms": round(cpu_ms, 2),
        "rss_mb": round(utils.rss_mb(), 2),
        "ok": True
    })

    t_req_comp_done = time.perf_counter()
    print(f"[WKR] DONE_COMPUTE trace={trace_id} id={my_id} "
          f"proc_ms={proc_ms:.2f} cpu_ms={cpu_ms:.2f} "
          f"rss={utils.rss_mb():.1f}MB "
          f"since_start_ms={(t_req_comp_done - t_req_in) * 1000.0:.1f}",
          flush=True)

    #Now forward (middle hops) or finish (if last hop)
    if hop_index < len(path) - 1:
        nxt = path[hop_index + 1]
        nxt_url = f"http://{nxt['ip']}:{int(nxt['port'])}/process"
        failed_id = str(nxt.get("id", "unknown"))
        try:

            print(f"[WKR] FWD_START trace={trace_id} from={my_id} -> {failed_id} "
                  f"url={nxt_url} out_b64_len={len(out_b64) if isinstance(out_b64, str) else None}",
                  flush=True)

            payload = {
                "trace_id": trace_id,
                "path": path,
                "hop_index": hop_index + 1,
                "activation_b64": out_b64,  # Always send Tensor to next hop
                "per_hop": per_hop,
            }
            if prompt:
                payload["prompt"] = prompt
            t_fwd0 = time.perf_counter()
            resp = requests.post(
                nxt_url,
                json=payload,
                timeout=consts.FWD_REQ_TIMEOUT_SEC,
            )
            t_fwd1 = time.perf_counter()

            print(f"[WKR] FWD_DONE trace={trace_id} from={my_id} -> {failed_id} "
                  f"status={resp.status_code} fwd_ms={(t_fwd1 - t_fwd0) * 1000.0:.1f}",
                  flush=True)

            try:
                j = resp.json()
            except Exception as e:
                print(f"[WKR] NEXT_INVALID_JSON trace={trace_id} next={failed_id} err={type(e).__name__}:{e}",
                      flush=True)
                j = {
                    "trace_id": trace_id,
                    "failed_id": failed_id,
                    "per_hop": per_hop,
                    "error": "next_invalid_json",
                }
            return jsonify(j), resp.status_code
        except requests.exceptions.Timeout as e:
            t_fwd1 = time.perf_counter()
            print(f"[WKR] FWD_TIMEOUT trace={trace_id} from={my_id} -> {failed_id} "
                  f"after_ms={(t_fwd1 - t_fwd0) * 1000.0:.1f} err={e}",
                  flush=True)
            return jsonify({
                "trace_id": trace_id,
                "failed_id": failed_id,
                "per_hop": per_hop,
                "error": "forward_timeout",
            }), 500

        except Exception as e:
            t_fwd1 = time.perf_counter()
            print(f"[WKR] FWD_FAIL trace={trace_id} from={my_id} -> {failed_id} "
                  f"after_ms={(t_fwd1 - t_fwd0) * 1000.0:.1f} err={type(e).__name__}:{e}",
                  flush=True)
            return jsonify({
                "trace_id": trace_id,
                "failed_id": failed_id,
                "per_hop": per_hop,
                "error": "forward_fail",
            }), 500

    #If this is final hop, need to do text detokenization + penalty
    #if this is last hop and it is a prompt context (text data, not tensor)
    if prompt and consts.TARP_ENGINE == "real":
        try:
            if TOKENIZER is None:
                TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)

            # A. Get Logits for the next token (Last position)
            next_token_logits = output_tensor[:, -1, :]

            # B. Apply Repetition Penalty (Server Side)
            # Re-tokenize the prompt to find what tokens are already used
            input_ids = TOKENIZER.encode(prompt, return_tensors="pt").to(consts.DEVICE)
            used_tokens = set(input_ids[0].tolist())

            for t in used_tokens:
                next_token_logits[0, t] -= 2.0  # Penalty

            # C. Select Best Token
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # D. Decode to Text
            generated_text = TOKENIZER.decode(next_token_id)

            print(f"[WKR] FINAL HOP (Light Mode): Generated '{generated_text}'")

            return jsonify({
                "trace_id": trace_id,
                "generated_text": generated_text,  #Return Text to Light Client
                "per_hop": per_hop,
                "done": True
            })

        except Exception as e:
            print(f"[WKR] Detokenize Error: {e}")
            return jsonify({"error": f"detokenize_error: {e}"}), 500

    #default return (tensor in real mode or simulated mode)
    return jsonify({
        "trace_id": trace_id,
        "final_activation_b64": out_b64,
        "per_hop": per_hop,
    }), 200




#Heartbeat Loop
def worker_heartbeat_loop(anchor_ip: str, anchor_port: int, wid: str):
    url = f"http://{anchor_ip}:{anchor_port}/heartbeat"
    print(f"[Worker] Starting heartbeat loop to {url}")
    while True:
        try:
            print(f"[Worker] Sending heartbeat for {wid}...")
            r= requests.post(url, json={"id": wid}, timeout=1)
            if r.status_code != 200:
                print(f"[Worker] Heartbeat REJECTED: {r.status_code}")
        except Exception as e:
            print(f"[Worker] Heartbeat FAILED: {e}")
            pass
        time.sleep(consts.HEARTBEAT_PERIOD_S)


# --- Startup ---
def start_worker(args):
    """
    args is a namespace/object containing:
    ip, port, cpu_load, fail_rate, trust0, id, anchor_ip, anchor_port, layer_start, layer_end
    """
    global MODEL_SHARD, TOKENIZER, MY_CONF, MY_CONFS

    # 1. Setup Configuration (Matches original dictionary structure)
    MY_CONF = {
        "ip": args.ip,
        "port": args.port,
        "cpu_load": args.cpu_load,
        "fail_rate": args.fail_rate,
        "trust": args.trust0,
        "trust0": args.trust0,
        "lat_ewma_ms": consts.LAT_INIT_MS,
        "id": args.id,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
    }

    # Empty dict for single-worker mode (ensures status() returns single-mode JSON)
    MY_CONFS = {}

    #Initialize Real Model Shard (If enabled)
    if consts.TARP_ENGINE == "real":

        print(f"[Worker] Loading tokenizer for {consts.MODEL_NAME}...")
        # Load it once during startup
        TOKENIZER = AutoTokenizer.from_pretrained(consts.MODEL_NAME)

        # Load config to check total layers and ensure consistency
        cfg = AutoConfig.from_pretrained(consts.MODEL_NAME)
        total = int(getattr(cfg, "n_layer", 0))
        if total <= 0:
            raise RuntimeError(f"Could not read n_layer for model={consts.MODEL_NAME}")

        # Update global constants to match the actual model
        consts.MODEL_LAYERS = total

        print(f"[Worker {args.id}] REAL mode enabled: model={consts.MODEL_NAME} "
              f"layers={consts.MODEL_LAYERS} shard=[{args.layer_start}-{args.layer_end}] "
              f"device={consts.DEVICE}")

        MODEL_SHARD = model_shard.RealModelShard(
            consts.MODEL_NAME,
            args.layer_start,
            args.layer_end,
            device=consts.DEVICE
        )

    #Register with Anchor
    try:
        reg_url = f"http://{args.anchor_ip}:{args.anchor_port}/register"
        resp = requests.post(reg_url, json=MY_CONF, timeout=2)

        if resp.status_code == 200:
            print(f"[Worker {args.id}] Registered with anchor {args.anchor_ip}:{args.anchor_port}")
        else:
            print(f"[Worker {args.id}] Register FAILED: HTTP {resp.status_code}")
            print(f"[Worker {args.id}] Response: {resp.text}")
    except Exception as e:
        print(f"[Worker {args.id}] Register ERROR: {e}")

    #start heartbeat
    threading.Thread(
        target=worker_heartbeat_loop,
        args=(args.anchor_ip, args.anchor_port, args.id),
        daemon=True
    ).start()

    #run Flask App
    print(f"[Worker {args.id}] listening on 0.0.0.0:{args.port} layers=[{args.layer_start},{args.layer_end}]")
    app.run(host="0.0.0.0", port=args.port, threaded=True)