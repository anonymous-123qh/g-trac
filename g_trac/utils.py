import time
import psutil
import base64
import io
import os
import csv
import math
import hashlib
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

from . import consts

_PS_PROC = psutil.Process(os.getpid())

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def unix_ts() -> float:
    return time.time()

def rss_mb() -> float:
    return _PS_PROC.memory_info().rss / (1024 * 1024)

def cpu_time_ms() -> float:
    t = _PS_PROC.cpu_times()
    return (float(t.user) + float(t.system)) * 1000.0

def burn_cpu(intensity: int) -> None:
    if intensity <= 0: return
    loops = intensity * 25000
    x = 0x12345678
    for _ in range(loops):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        x ^= (x >> 13)
        x = (x * 2246822519) & 0xFFFFFFFF

# --- Serialization ---
def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def tensor_to_b64(tensor) -> str:
    if torch is None: return ""
    t = tensor.cpu()
    buff = io.BytesIO()
    torch.save(t, buff, _use_new_zipfile_serialization=False)
    buff.seek(0)
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def b64_to_tensor(b64_str: str):
    if torch is None: raise ImportError("Torch not installed")
    import binascii
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64: {e}")
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=False)

# --- Node Helpers ---
def is_alive(node: Dict[str, Any], now: Optional[float] = None) -> bool:
    now = unix_ts() if now is None else now
    last = node.get("last_seen_ts")
    if last is None: return False
    return (now - float(last)) <= consts.NODE_TTL_S

def node_layers(node: Dict[str, Any]):
    ls = node.get("layer_start")
    le = node.get("layer_end")
    if ls is None or le is None: return (-1, -1)
    return (int(ls), int(le))

def get_lat_ms(node: Dict[str, Any]) -> float:
    try: return float(node.get("lat_ewma_ms", consts.LAT_INIT_MS))
    except: return consts.LAT_INIT_MS

def get_trust(node: Dict[str, Any]) -> float:
    try: return float(node.get("trust", 0.0))
    except: return 0.0

# --- Logging ---
def _ensure_header(path: str, fields: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

def log_row(path: str, fields: list, row: dict):
    _ensure_header(path, fields)
    out = {k: row.get(k, "") for k in fields}
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(out)

def percentile(sorted_vals, p):
    if not sorted_vals: return float("nan")
    if p <= 0: return sorted_vals[0]
    if p >= 100: return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def prune_registry(registry: Dict[str, Any]):
    """
    Removes nodes from the registry that haven't been seen
    for longer than consts.PRUNE_AFTER_S.
    """
    now = unix_ts()
    to_delete = []

    # Identify dead nodes
    for wid, meta in registry.items():
        last = float(meta.get("last_seen_ts", 0.0))
        if (now - last) > consts.PRUNE_AFTER_S:
            to_delete.append(wid)

    # Remove them
    for wid in to_delete:
        del registry[wid]