import os
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from safetensors import safe_open
from typing import List


#Helper function
def _get_safetensors_files(model_dir: str) -> List[str]:
    """
    Return a list of all .safetensors files in the directory.
    Essential for sharded models (like gpt2-large/xl).
    """
    files = [
        os.path.join(model_dir, fn)
        for fn in os.listdir(model_dir)
        if fn.endswith(".safetensors")
    ]
    if not files:
        raise FileNotFoundError(
            f"No .safetensors weights found in {model_dir}. "
            f"Ensure the model is downloaded in safetensors format."
        )
    return sorted(files)


class GPT2TrueShard(nn.Module):
    """
    Minimal GPT-2 shard that holds only the required parameters:
      - embeddings (if start==0)
      - transformer blocks [start..end]
      - ln_f + lm_head (if end==last)
    """

    def __init__(self, cfg, start_layer: int, end_layer: int):
        super().__init__()
        if not hasattr(cfg, "_attn_implementation") or cfg._attn_implementation is None:
            cfg._attn_implementation = "eager"


        self.cfg = cfg
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.total_layers = int(cfg.n_layer)

        #embeddings only needed in first shard
        if self.start_layer == 0:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        else:
            self.wte = None
            self.wpe = None

        #only keep the blocks we serve
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(self.end_layer - self.start_layer + 1)])

        #final layernorm + lm_head only needed in last shard
        if self.end_layer == self.total_layers - 1:
            self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        else:
            self.ln_f = None
            self.lm_head = None

    @torch.no_grad()
    def forward(self, x):
        if self.start_layer == 0:
            if x.dtype != torch.long:
                x = x.long()
            bsz, seqlen = x.shape
            pos = torch.arange(0, seqlen, device=x.device).unsqueeze(0)  # [1, T]
            h = self.wte(x) + self.wpe(pos)
        else:
            h = x

        #run served blocks
        for blk in self.blocks:
            h = blk(h)[0]

        #if it is the last shard, produce logits
        if self.lm_head is not None:
            h = self.ln_f(h)
            h = self.lm_head(h)  # [B, T, vocab]
        return h


class RealModelShard:
    """
    True shard loader: loads only required tensors from safetensors.
    Supports sharded models (multiple .safetensors files).
    """

    def __init__(self, model_name_or_path: str, start_layer: int, end_layer: int, device: str = "cpu",
                 dtype: str = "fp32"):
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)

        cfg = AutoConfig.from_pretrained(model_name_or_path)
        self.total_layers = int(cfg.n_layer)

        if self.start_layer < 0 or self.end_layer >= self.total_layers or self.end_layer < self.start_layer:
            raise ValueError(
                f"Invalid shard range [{self.start_layer},{self.end_layer}] for total_layers={self.total_layers}")

        #choose dtype
        dtype = dtype.lower().strip()
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        #build minimal shard module
        shard = GPT2TrueShard(cfg, self.start_layer, self.end_layer)
        shard.eval()

        #device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        #download/locate model directory
        model_dir = model_name_or_path
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "config.json", "tokenizer.json", "vocab.json",
                                "merges.txt"],
            )

        #get all weight files to handle sharding
        weight_files = _get_safetensors_files(model_dir)

        #helper to find a tensor across multiple files
        def load_tensor_any_file(possible_names: List[str]):
            for wf in weight_files:
                with safe_open(wf, framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    for name in possible_names:
                        if name in keys:
                            return f.get_tensor(name)
            return None

        print(f"[Model] Scanning {len(weight_files)} file(s) for tensors...")
        state = {}

        #embeddings ( if start_layer == 0)

        if self.start_layer == 0:
            #try standard keys (transformer.wte.weight) and raw keys (wte.weight)
            wte = load_tensor_any_file(["transformer.wte.weight", "wte.weight"])
            wpe = load_tensor_any_file(["transformer.wpe.weight", "wpe.weight"])

            if wte is None or wpe is None:
                raise KeyError(f"Could not find embeddings (wte/wpe) in {weight_files}")

            state["wte.weight"] = wte.to(torch_dtype)
            state["wpe.weight"] = wpe.to(torch_dtype)

        #blocks: map transformer.h.{i}.xxx -> blocks.{i-start}.xxx
        #perform a targeted search for the layers we own to avoid scanning every file unnecessarily
        for i in range(self.start_layer, self.end_layer + 1):
            #iterate files to find the specific block components

            #common components in a GPT2 Block
            components = [
                ("ln_1.weight", f"transformer.h.{i}.ln_1.weight"),
                ("ln_1.bias", f"transformer.h.{i}.ln_1.bias"),
                ("attn.c_attn.weight", f"transformer.h.{i}.attn.c_attn.weight"),
                ("attn.c_attn.bias", f"transformer.h.{i}.attn.c_attn.bias"),
                ("attn.c_proj.weight", f"transformer.h.{i}.attn.c_proj.weight"),
                ("attn.c_proj.bias", f"transformer.h.{i}.attn.c_proj.bias"),
                ("ln_2.weight", f"transformer.h.{i}.ln_2.weight"),
                ("ln_2.bias", f"transformer.h.{i}.ln_2.bias"),
                ("mlp.c_fc.weight", f"transformer.h.{i}.mlp.c_fc.weight"),
                ("mlp.c_fc.bias", f"transformer.h.{i}.mlp.c_fc.bias"),
                ("mlp.c_proj.weight", f"transformer.h.{i}.mlp.c_proj.weight"),
                ("mlp.c_proj.bias", f"transformer.h.{i}.mlp.c_proj.bias"),
            ]

            dst_prefix = f"blocks.{i - self.start_layer}."

            for local_name, source_name in components:
                #try finding exact match
                t = load_tensor_any_file([source_name])
                if t is None:
                    #fallback for some HF versions that omit 'transformer.'
                    t = load_tensor_any_file([source_name.replace("transformer.", "")])

                if t is not None:
                    state[dst_prefix + local_name] = t.to(torch_dtype)
                else:
                    print(f"Warning: Tensor {source_name} not found.")

        #final layernorm + lm_head
        if self.end_layer == self.total_layers - 1:
            ln_f_w = load_tensor_any_file(["transformer.ln_f.weight", "ln_f.weight"])
            ln_f_b = load_tensor_any_file(["transformer.ln_f.bias", "ln_f.bias"])

            if ln_f_w is not None: state["ln_f.weight"] = ln_f_w.to(torch_dtype)
            if ln_f_b is not None: state["ln_f.bias"] = ln_f_b.to(torch_dtype)

            #lm Head
            lm_head = load_tensor_any_file(["lm_head.weight"])
            if lm_head is None:
                #tie to wte if not found (standard GPT-2), if we are the last shard but NOT the first shard, we might not have wte loaded in state
                #must fetch wte explicitly again if strictly required
                wte_source = load_tensor_any_file(["transformer.wte.weight", "wte.weight"])
                state["lm_head.weight"] = wte_source.to(torch_dtype)
            else:
                state["lm_head.weight"] = lm_head.to(torch_dtype)

        missing, unexpected = shard.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")

        #move to target device
        shard.to(self.device)

        self.model = shard
        print(f"[Model] True-shard loaded [{self.start_layer},{self.end_layer}] / {self.total_layers} "
              f"on {self.device} dtype={torch_dtype}")

    def forward(self, x):

        x = x.to(self.device)

        #ensure tensor is Long (int64) for the first layer (Embeddings)
        #intermediate layers (start > 0) expect Float (hidden states).
        if self.start_layer == 0 and x.dtype != torch.long:
            x = x.long()

        return self.model(x)