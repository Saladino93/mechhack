"""Core activation extraction logic.

Handles model loading, forward pass, layer selection, and saving.
Designed to be imported by run.py or reused in later experiments.

Usage as a library:
    from experiments.02_extract_activations.extractor import ActivationExtractor

    ext = ActivationExtractor(model_key="gemma4_31b", layers="middle")
    ext.load_model()
    result = ext.extract_single(prompt_text)
    ext.save(result, output_path)
"""
import os
import sys
import time
import json
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STARTER_CODE = REPO_ROOT / "starter_code"

# Add starter_code to path for chunked_sdpa
sys.path.insert(0, str(STARTER_CODE))

# Model registry: key -> HF repo name
MODEL_REGISTRY = {
    "gemma4_31b": "google/gemma-4-31B-it",
    "qwen36": "Qwen/Qwen3.6-27B",
}


def resolve_model_path(model_key, model_path=None):
    """Resolve model to a local directory.

    Lookup order:
      1. Explicit model_path argument
      2. $HACKATHON_MODELS_DIR/<repo-name>
      3. /data/<repo-name>          (cluster RO mount)
      4. <repo-root>/models/<repo-name>  (local download)
      5. HF cache via snapshot_download
    """
    if model_path and Path(model_path).exists():
        return str(Path(model_path).resolve())

    repo_id = MODEL_REGISTRY.get(model_key)
    if not repo_id:
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(MODEL_REGISTRY)}")

    repo_name = repo_id.split("/")[-1]
    candidates = []

    if env_dir := os.environ.get("HACKATHON_MODELS_DIR"):
        candidates.append(Path(env_dir) / repo_name)
    candidates.append(Path("/data") / repo_name)
    candidates.append(REPO_ROOT / "models" / repo_name)

    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: HF cache
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate {model_key} ({repo_id}). Tried:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + f"\nRun `python starter_code/download_models.py --out_dir ./models` "
            f"or set MODEL_PATH / HACKATHON_MODELS_DIR."
        ) from e


def parse_layers(spec, n_layers):
    """Resolve a layer spec string to a sorted list of layer indices.

    Indices are into output_hidden_states: 0 = embeddings, 1..n_layers = block outputs.

    Supported specs:
        "middle"        -> [n_layers // 2]
        "early"         -> [n_layers // 4]
        "late"          -> [3 * n_layers // 4]
        "all"           -> [0, 1, ..., n_layers]
        "32"            -> [32]
        "10,30,50"      -> [10, 30, 50]
        "0:65:8"        -> range(0, 65, 8)
    """
    max_idx = n_layers  # inclusive

    if isinstance(spec, (list, tuple)):
        idxs = [int(x) for x in spec]
    elif isinstance(spec, int):
        idxs = [spec]
    elif isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            idxs = list(range(max_idx + 1))
        elif s == "middle":
            idxs = [n_layers // 2]
        elif s == "early":
            idxs = [n_layers // 4]
        elif s == "late":
            idxs = [3 * n_layers // 4]
        elif ":" in s:
            parts = s.split(":")
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            idxs = list(range(start, stop, step))
        elif "," in s:
            idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
        else:
            idxs = [int(s)]
    else:
        raise ValueError(f"Unsupported layer spec: {spec!r}")

    bad = [i for i in idxs if not (0 <= i <= max_idx)]
    if bad:
        raise ValueError(f"Layer indices {bad} out of range [0, {max_idx}]")

    return sorted(set(idxs))


class ActivationExtractor:
    """Extracts residual stream activations from a transformer model.

    Attributes:
        model_key: str — which model ('gemma4_31b' or 'qwen36')
        layers_spec: str — layer selection spec (see parse_layers)
        dtype: torch.dtype — storage dtype for saved activations
        use_chat_template: bool — whether to wrap prompts in chat template
        device: str — device to run on
    """

    def __init__(self, model_key="gemma4_31b", layers="middle",
                 dtype="fp16", use_chat_template=True, model_path=None,
                 device="cuda:0"):
        self.model_key = model_key
        self.layers_spec = layers
        self.storage_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.dtype_name = dtype
        self.use_chat_template = use_chat_template
        self.model_path = model_path
        self.device = device

        # Set after load_model()
        self.model = None
        self.tokenizer = None
        self.layer_idxs = None
        self.n_layers = None

    def load_model(self):
        """Load the model and tokenizer. Must be called before extract_single()."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        path = resolve_model_path(self.model_key, self.model_path)
        print(f"Loading {self.model_key} from {path}...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Determine number of layers
        if hasattr(self.model.model, "layers"):
            self.n_layers = len(self.model.model.layers)
        else:
            self.n_layers = 64  # fallback

        self.layer_idxs = parse_layers(self.layers_spec, self.n_layers)
        print(f"Model loaded | n_layers={self.n_layers} | "
              f"extracting {len(self.layer_idxs)} layer(s): {self.layer_idxs}", flush=True)

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    def _tokenize(self, prompt_text):
        """Tokenize a prompt, optionally applying chat template."""
        if self.use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt_text

        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        return encoded.input_ids, encoded.attention_mask

    def _needs_chunked_sdpa(self):
        """Gemma 4-31B needs chunked SDPA for head_dim=512."""
        return self.model_key == "gemma4_31b"

    def extract_single(self, prompt_text):
        """Run a forward pass and extract residual activations.

        Args:
            prompt_text: The raw prompt string.

        Returns:
            Dict with keys:
              residuals: Tensor (n_selected_layers, n_tokens, d_model) at storage_dtype
              input_ids: Tensor (n_tokens,) int32
              attention_mask: Tensor (n_tokens,) bool
              n_tokens: int
              layer_idxs: list[int]
              fwd_seconds: float
              peak_vram_gb: float
        """
        assert self.model is not None, "Call load_model() first"

        input_ids, attention_mask = self._tokenize(prompt_text)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()

        # Conditional chunked SDPA scope for Gemma
        if self._needs_chunked_sdpa():
            from chunked_sdpa import chunked_sdpa_scope
            ctx = chunked_sdpa_scope()
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        fwd_time = time.time() - t0
        peak_vram = (torch.cuda.max_memory_allocated() / 1024**3
                     if torch.cuda.is_available() else 0.0)

        # Extract selected layers: hidden_states is tuple of (1, N, d_model)
        hidden_states = out.hidden_states
        stacked = torch.stack(
            [hidden_states[k][0] for k in self.layer_idxs], dim=0
        )  # (n_selected_layers, N, d_model)

        result = {
            "residuals": stacked.to("cpu", dtype=self.storage_dtype).contiguous(),
            "input_ids": input_ids[0].to("cpu", dtype=torch.int32),
            "attention_mask": attention_mask[0].to("cpu", dtype=torch.bool),
            "n_tokens": int(stacked.shape[1]),
            "layer_idxs": self.layer_idxs,
            "fwd_seconds": round(fwd_time, 3),
            "peak_vram_gb": round(peak_vram, 2),
        }

        # Cleanup GPU memory
        del out, hidden_states, stacked
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @staticmethod
    def save(result, path):
        """Save extraction result to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, str(path))

    @staticmethod
    def load(path):
        """Load a previously saved extraction."""
        return torch.load(str(path), weights_only=False)

    def get_metadata(self):
        """Return metadata dict about this extractor's configuration."""
        return {
            "model_key": self.model_key,
            "n_layers_model": self.n_layers,
            "layer_idxs": self.layer_idxs,
            "layers_spec": self.layers_spec,
            "dtype": self.dtype_name,
            "use_chat_template": self.use_chat_template,
        }


def extract_dataset(extractor, samples, out_dir, limit=0):
    """Extract activations for a list of samples, saving each as a .pt file.

    Args:
        extractor: ActivationExtractor (already loaded)
        samples: List of dicts with 'sample_id' and 'prompt'
        out_dir: Directory to save .pt files
        limit: Max samples to process (0 = all)

    Returns:
        metadata dict with extraction stats
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit > 0:
        samples = samples[:limit]

    metadata = extractor.get_metadata()
    metadata["n_samples_requested"] = len(samples)
    metadata["samples"] = []

    t_start = time.time()
    n_ok, n_skip, n_err = 0, 0, 0

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        out_path = out_dir / f"{sid}.pt"

        # Resume: skip if already extracted
        if out_path.exists():
            n_skip += 1
            continue

        try:
            result = extractor.extract_single(sample["prompt"])
            result["label"] = sample.get("label")
            result["sample_id"] = sid
            extractor.save(result, out_path)
            n_ok += 1

            metadata["samples"].append({
                "sample_id": sid,
                "n_tokens": result["n_tokens"],
                "fwd_seconds": result["fwd_seconds"],
            })

            # Progress logging
            if (n_ok + n_skip) % 10 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - t_start
                processed = n_ok + n_skip
                rate = processed / max(elapsed, 1e-3)
                remaining = len(samples) - (i + 1)
                eta = remaining / max(rate, 1e-3) / 60
                sz = out_path.stat().st_size / 1024**2
                print(f"  [{i+1}/{len(samples)}] {sid}: "
                      f"N={result['n_tokens']} fwd={result['fwd_seconds']:.2f}s "
                      f"peak={result['peak_vram_gb']:.1f}GB sz={sz:.0f}MB | "
                      f"{rate:.2f}/s eta={eta:.1f}min", flush=True)

        except Exception as e:
            n_err += 1
            print(f"  [{i+1}/{len(samples)}] {sid}: FAIL "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            metadata["samples"].append({
                "sample_id": sid,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - t_start
    metadata["n_ok"] = n_ok
    metadata["n_skipped"] = n_skip
    metadata["n_errors"] = n_err
    metadata["total_seconds"] = round(total_time, 1)

    # Save metadata
    meta_path = out_dir / "extraction_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== DONE ok={n_ok} skipped={n_skip} errors={n_err} | "
          f"{total_time/60:.1f} min ===", flush=True)
    print(f"Metadata: {meta_path}", flush=True)

    return metadata
