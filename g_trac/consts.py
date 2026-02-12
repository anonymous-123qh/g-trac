import os
import time

# --- LOGGING -------------------
LOG_DIR = os.environ.get("GTRAC_LOG_DIR", "logs").strip()
RUN_ID = os.environ.get("GTRAC_RUN_ID", "").strip() or time.strftime("%Y%m%d-%H%M%S")
TRUST_LOG_PATH = os.path.join(LOG_DIR, f"trust_{RUN_ID}.csv")
REQUEST_LOG_PATH = os.path.join(LOG_DIR, f"requests_{RUN_ID}.csv")
#--------------------------------------------

# --- ENGINE CONFIG -----------------------
TARP_ENGINE = os.environ.get("TARP_ENGINE", "sim").strip().lower()  # "sim" | "real"
MODEL_NAME = os.environ.get("TARP_MODEL", "gpt2").strip()
DEVICE = os.environ.get("TARP_DEVICE", "cpu").strip().lower()
TENSOR_SIZE_BYTES = 20 * 1024  # simulated activation size
#----------------------------------------------------------------


# --- AUTO-DETECT LAYERS BASED ON MODEL NAME --------------
#ensures Anchor and Worker agree without Anchor needing to load PyTorch
if "xl" in MODEL_NAME:
    MODEL_LAYERS = 48
elif "large" in MODEL_NAME:
    MODEL_LAYERS = 36
elif "medium" in MODEL_NAME:
    MODEL_LAYERS = 24
else:
    MODEL_LAYERS = 12  # Default gpt2 small
#-----------------------------------------------------------------------

# --- NETWORK & TIMEOUTS -------------------------------------------------
GOSSIP_FREQ_SEC = 2.0
CLIENT_REQ_TIMEOUT_SEC = float(os.environ.get("TARP_CLIENT_TIMEOUT", "25.0"))
FWD_REQ_TIMEOUT_SEC = float(os.environ.get("TARP_FWD_TIMEOUT", "20.0"))
CTRL_TIMEOUT_SEC = float(os.environ.get("TARP_CTRL_TIMEOUT", "2.0"))
#-------------------------------------------------------------------------

# --- TRUST & LATENCY --------------------------------------------
TRUST_REWARD = 0.03
TRUST_PENALTY = 0.20
TRUST_UNKNOWN_PENALTY = 0.05
LAT_EWMA_ALPHA = 0.30
LAT_INIT_MS = 250.0
TRUST_MIN_PER_HOP = 0.96
LARAC_MIN_RELIABILITY = 0.80
#--------------------------------------------------------------------------

# --- LIVENESS ---------------------------------------------------------
HEARTBEAT_PERIOD_S = 8.0
NODE_TTL_S = 30.0
PRUNE_AFTER_S = 60.0
#--------------------------------------------------------------------------

# --- SELECTION ----------------------------------------------
MAX_CHAIN_ENUM = 5000
REPAIR_ON_FAILURE = True
#----------------------------------------------------------------------------

# --- CSV HEADER DEFINITIONS---------------------------------------------
REQUEST_FIELDS = [
    "run_id", "ts_unix", "mode", "engine", "model",
    "request_id", "prompt_len_tokens", "target_new_tokens",
    "generated_new_tokens", "completed", "request_success",
    "request_e2e_ms", "selection_overhead_ms",
    "repair_used", "repair_attempted", "repair_succeeded",
    "failed_id", "failed_error", "failed_stage",
    "chain_ids", "chain_layers", "client_rss_mb",
    "trust_tau", "trust_min_in_chain", "trust_mean_in_chain"
]
#-----------------------------------------------------------------