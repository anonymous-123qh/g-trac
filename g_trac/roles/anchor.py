from flask import Flask, jsonify, request, render_template_string
import threading
from .. import consts, utils, dashboard

REGISTRY = {}
app = Flask(__name__)


@app.route('/')
def route_dashboard():
    # Prune dead nodes from the actual registry
    utils.prune_registry(REGISTRY)

    now = utils.unix_ts()

    #Create a "Display Registry" with the fields the HTML needs
    display_registry = {}

    for wid, data in REGISTRY.items():
        # Make a copy so we don't modify the real registry
        node = data.copy()

        # Calculate Age
        last_seen = float(node.get("last_seen_ts", 0))
        age = now - last_seen

        # --- MAP FIELDS FOR THE HTML TEMPLATE ---
        node["alive"] = (age < consts.NODE_TTL_S)  # Required for ONLINE/OFFLINE status
        node["age_str"] = f"{age:.1f}s"  # Required for "Seen X s ago"
        node["latency"] = int(node.get("lat_ewma_ms", 0))  # Required for "Est. Latency"
        node["cpu_load"] = node.get("cpu_load", 0)  # For tooltip
        node["net_delay_ms"] = node.get("net_delay_ms", 0)  # For tooltip

        #CALCULATE TRUST TREND
        # Compare current trust vs previous trust to determine arrow direction
        current_trust = float(node.get("trust", 1.0))
        prev_trust = float(node.get("prev_trust", current_trust))

        # Use a small epsilon for float comparison
        if current_trust > prev_trust + 0.0001:
            node["display_trust_trend"] = "up"
        elif current_trust < prev_trust - 0.0001:
            node["display_trust_trend"] = "down"
        else:
            node["display_trust_trend"] = "eq"

        # Add to display dict
        display_registry[wid] = node

    # 3. Render
    return render_template_string(
        dashboard.DASHBOARD_HTML,
        registry=display_registry,
        now=now
    )

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    wid = str(data.get("id", "")).strip()
    # Normalize layers
    ls, le = utils.node_layers(data)
    if not wid:
        print("[Anchor] Register reject: missing id. Payload:", data)
        return jsonify({"error": "missing id"}), 400

    # Required for Petals-like constraints
    if "layer_start" not in data or "layer_end" not in data:
        print(f"[Anchor] Register reject {wid}: missing layer fields. Payload:", data)
        return jsonify({"error": "missing layer_start/layer_end"}), 400

    data["ip"] = str(data.get("ip", "")).strip()
    data["port"] = int(data.get("port", 0))
    data["trust"] = float(data.get("trust", 0.5))
    data["prev_trust"] = float(data["trust"])

    data["lat_ewma_ms"] = float(data.get("lat_ewma_ms", consts.LAT_INIT_MS))
    data["last_seen_ts"] = utils.unix_ts()

    # Normalize layers
    ls, le = utils.node_layers(data)
    if ls < 0 or le < ls or le >= consts.MODEL_LAYERS:
        print(f"[Anchor] Register reject {wid}: invalid layer range [{ls},{le}] MODEL_LAYERS={consts.MODEL_LAYERS}. Payload:",
              data)
        return jsonify({"error": f"invalid layer range [{ls},{le}] for MODEL_LAYERS={consts.MODEL_LAYERS}"}), 400

    REGISTRY[wid] = data
    print(
        f"[Anchor] Registered: {wid} {data['ip']}:{data['port']} layers=[{ls},{le}] trust={data['trust']:.2f} lat={data['lat_ewma_ms']:.0f}ms")
    return jsonify({"status": "ok"}), 200

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    try:
        data = request.get_json(silent=True) or {}
        wid = str(data.get("id", "")).strip()

        print(f"[Anchor] Heartbeat received from: {wid}")
        if not wid or wid not in REGISTRY:
            print(f"[Anchor] Heartbeat ignored (Unknown ID): {wid}")
            return jsonify({"error": "unknown id"}), 404

        REGISTRY[wid]["last_seen_ts"] = utils.unix_ts()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"[Anchor] Heartbeat ERROR: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/sync", methods=["GET"])
def sync():
    utils.prune_registry(REGISTRY)
    now = utils.unix_ts()
    alive = {wid: meta for wid, meta in REGISTRY.items() if utils.is_alive(meta, now)}

    print("\n[Anchor] SYNC request")
    print(f"  Alive workers: {len(alive)}")
    for wid, m in alive.items():
        ls, le = m.get("layer_start"), m.get("layer_end")
        trust = float(m.get("trust", 0.0))
        lat = float(m.get("lat_ewma_ms", 0.0))
        age = now - float(m.get("last_seen_ts", 0))
        print(
            f"   - {wid:6s} "
            f"{m.get('ip')}:{m.get('port')} "
            f"layers=[{ls}-{le}] "
            f"trust={trust:.2f} "
            f"lat={lat:.0f}ms "
            f"last_seen={age:.1f}s ago"
        )

    return jsonify(alive), 200

@app.route("/feedback", methods=["POST"])
def feedback():
    """
        Client sends:
          {
            "trace_id": "...",
            "success": true/false,
            "failed_id": "wX" (if failure),
            "per_hop": [{"id":"wA","proc_ms":..,"ok":true}, ...],
            "path_ids": ["wA","wB",...]   # optional
            "attribution": "failed_only"|"all"  # optional
          }
        """
    data = request.get_json(silent=True) or {}
    success = bool(data.get("success", False))
    failed_id = data.get("failed_id")
    print(f"[Anchor] FEEDBACK trace={data.get('trace_id')} success={success} failed_id={failed_id}")

    attribution = str(data.get("attribution", "failed_only"))
    per_hop = data.get("per_hop", [])
    if not isinstance(per_hop, list):
        per_hop = []

    #update latency EWMA from per-hop processing times
    for hop in per_hop:
        hid = hop.get("id")
        proc_ms = hop.get("proc_ms")
        ok = hop.get("ok", False)
        if not hid or hid not in REGISTRY:
            continue
        if proc_ms is None or not ok:
            continue
        old = float(REGISTRY[hid].get("lat_ewma_ms", consts.LAT_INIT_MS))
        new = (1.0 - consts.LAT_EWMA_ALPHA) * old + consts.LAT_EWMA_ALPHA * float(proc_ms)
        REGISTRY[hid]["lat_ewma_ms"] = new

    #determine path ids for reward/punish
    path_ids = [h.get("id") for h in per_hop if h.get("id")] or (data.get("path_ids") or [])

    #update trust
    if success:
        for wid in path_ids:
            if wid in REGISTRY:
                oldt = float(REGISTRY[wid].get("trust", 0.5))
                REGISTRY[wid]["prev_trust"] = oldt
                REGISTRY[wid]["trust"] = min(1.0, oldt + consts.TRUST_REWARD)
    else:
        if attribution == "all":
            for wid in path_ids:
                if wid in REGISTRY:
                    oldt = float(REGISTRY[wid].get("trust", 0.5))
                    REGISTRY[wid]["prev_trust"] = oldt
                    REGISTRY[wid]["trust"] = max(0.0, oldt - consts.TRUST_PENALTY)
        else:
            #failed only mode
            if failed_id and failed_id in REGISTRY:
                # punish only the blamed hop strongly
                oldt = float(REGISTRY[failed_id].get("trust", 0.5))
                REGISTRY[failed_id]["prev_trust"] = oldt
                REGISTRY[failed_id]["trust"] = max(0.0, oldt - consts.TRUST_PENALTY)
                print(f"[Anchor] trust[{failed_id}]={REGISTRY[failed_id]['trust']:.2f}")
            else:
                #UNKNOWN culprit: punish all hops lightly
                for wid in path_ids:
                    if wid in REGISTRY:
                        oldt = float(REGISTRY[wid].get("trust", 0.5))
                        REGISTRY[wid]["prev_trust"] = oldt
                        REGISTRY[wid]["trust"] = max(0.0, oldt - consts.TRUST_UNKNOWN_PENALTY)

                print(f"[Anchor] unknown failure: lightly penalized {len(path_ids)} hops")

    return jsonify({"status": "ok"}), 200

@app.route("/notify_active", methods=["POST"])
def notify_active():
    """Client calls this immediately after selecting a chain."""
    data = request.get_json(silent=True) or {}
    chain_ids = data.get("chain_ids", [])
    if not isinstance(chain_ids, list):
        chain_ids = []
    now = utils.unix_ts()
    for wid in chain_ids:
        wid = str(wid).strip()
        if wid in REGISTRY:
            #update the timestamp so the dashboard knows it's active
            REGISTRY[wid]["last_active_ts"] = now

    return jsonify({"status": "ok"}), 200

@app.route("/reset_state", methods=["POST"])
def reset_state():
    for m in REGISTRY.values():
        m["trust"] = m.get("trust0", 1.0)
        m["lat_ewma_ms"] = consts.LAT_INIT_MS
    return jsonify({"status": "reset"})

def start_anchor(port, layers):
    consts.MODEL_LAYERS = layers
    print(f"ANCHOR listening on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)