"""Depth-vs-token failure analysis for Constitutional-style linear probes.

This script loads a trained `constitutional_probe.pt` checkpoint plus saved residual
activation `.pt` files and produces:

  * per-class layer x token heatmaps
  * refusal/compliance difference heatmaps
  * per-sample onset statistics
  * optional per-sample arrays for debugging

It works with the extracts produced by `experiments/02_extract_activations`, where
residuals are saved as `(n_layers, n_tokens, d_model)` and labels/sample IDs are
saved inside each `.pt` file.

Interpretation note:
  The trained probe logit is:

      z[token] = sum_layer dot(W_layer, residual[layer, token]) + bias

  We therefore report three maps:

      contribution: dot(W_layer, residual[layer, token])
      cumulative:   cumsum over layers of contribution + bias
      layer_logit:  contribution + bias / n_layers

  `cumulative` is usually the best map for asking "by what depth has the
  representation become refusal-like?"; `contribution` is better for asking
  "which layer contributes most to the final probe score?".

Example:
    python experiments/04_failure_analysis/depth_token_maps.py \
      --probe_ckpt ./probes/constitutional_refusal_gemma/constitutional_probe.pt \
      --extracts_dir ./extracts/gemma4_31b_refusal_layers4_train,./extracts/gemma4_31b_refusal_layers4_test \
      --dataset refusal_gemma4_31b \
      --task refusal_gemma4_31b \
      --token_window last \
      --max_tokens 256 \
      --map_type cumulative \
      --out_dir ./analysis/refusal_depth_token
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (p / "experiments" / "02_extract_activations" / "data.py").exists():
            return p
    return Path.cwd()


def import_data_helpers():
    root = repo_root()
    data_dir = root / "experiments" / "02_extract_activations"
    if not (data_dir / "data.py").exists():
        return None, None
    sys.path.insert(0, str(data_dir))
    from data import get_label_for_task, load_dataset  # type: ignore
    return get_label_for_task, load_dataset


def parse_dirs(arg: str) -> list[Path]:
    return [Path(x).expanduser() for x in arg.split(",") if x.strip()]


def iter_pt_files(dirs: Sequence[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for d in dirs:
        for p in sorted(d.glob("*.pt")):
            if p.name in seen:
                continue
            seen.add(p.name)
            yield p


def load_selection_ids(path_arg: str | None) -> set[str] | None:
    if not path_arg:
        return None
    data = json.loads(Path(path_arg).expanduser().read_text())
    rows = data.get("samples", data) if isinstance(data, dict) else data
    ids = set()
    for row in rows:
        if isinstance(row, str):
            ids.add(row)
        elif isinstance(row, dict) and "sample_id" in row:
            ids.add(str(row["sample_id"]))
    if not ids:
        raise ValueError(f"No sample_id entries found in selection file {path_arg}")
    return ids


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        # Allow a raw state dict if someone saved only model.state_dict().
        ckpt = {"state_dict": ckpt, "config": {}}
    sd = ckpt["state_dict"]
    # Strip possible module prefixes.
    fixed = {}
    for k, v in sd.items():
        kk = k
        for prefix in ("module.", "model.", "probe."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        fixed[kk] = v
    ckpt["state_dict"] = fixed
    if "weight" not in fixed:
        raise KeyError(
            "Checkpoint state_dict does not contain `weight`. "
            "This script expects a StreamingLinearProbe checkpoint."
        )
    if "bias" not in fixed:
        fixed["bias"] = torch.tensor(0.0)
    return ckpt


def residual_tensor(ex: dict) -> torch.Tensor:
    if "residuals" in ex:
        r = ex["residuals"]
    elif "middle_layer_all_tokens" in ex:
        r = ex["middle_layer_all_tokens"].unsqueeze(0)
    else:
        raise KeyError("extract is missing `residuals` or `middle_layer_all_tokens`")
    if r.dim() == 2:
        r = r.unsqueeze(0)
    if r.dim() != 3:
        raise ValueError(f"expected residuals (L,T,D), got {tuple(r.shape)}")
    return r.float()


def resolve_layer_positions(ex: dict, ckpt: dict) -> tuple[list[int], list[int | None]]:
    cfg = ckpt.get("config", {}) or {}
    weight = ckpt["state_dict"]["weight"]
    n_probe_layers = int(weight.shape[0])
    n_extract_layers = int(residual_tensor(ex).shape[0])

    if "layer_positions" in cfg:
        positions = [int(x) for x in cfg["layer_positions"]]
        if len(positions) != n_probe_layers:
            raise ValueError(
                f"checkpoint layer_positions has {len(positions)} layers but weight has {n_probe_layers}"
            )
    elif "source_layer_idxs" in cfg and "layer_idxs" in ex:
        source_to_pos = {int(src): i for i, src in enumerate(ex["layer_idxs"])}
        positions = [source_to_pos[int(src)] for src in cfg["source_layer_idxs"]]
    else:
        positions = list(range(n_probe_layers))

    if max(positions, default=-1) >= n_extract_layers:
        raise ValueError(
            f"checkpoint selects positions {positions}, but extract has only {n_extract_layers} layers"
        )
    source_layers: list[int | None]
    if "layer_idxs" in ex:
        source_layers = [int(ex["layer_idxs"][p]) for p in positions]
    else:
        source_layers = [None for _ in positions]
    return positions, source_layers


def get_binary_label(ex: dict, dataset: str | None, task: str | None) -> int | None:
    raw = ex.get("label")
    if raw is None:
        return None
    if dataset and task:
        get_label_for_task, _ = import_data_helpers()
        if get_label_for_task is not None:
            y = get_label_for_task({"sample_id": ex.get("sample_id"), "label": raw}, task)
            return None if y is None else int(y)
    # Fallbacks for refusal-like bool/int labels.
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw in (0, 1):
            return int(raw)
    if isinstance(raw, str):
        low = raw.lower()
        if low in {"1", "true", "refusal", "refuse", "non_compliance", "non-compliance", "positive"}:
            return 1
        if low in {"0", "false", "compliance", "comply", "negative", "benign"}:
            return 0
    return None


def span_indices(ex: dict, span: str, mask: torch.Tensor) -> list[int]:
    real = mask.bool().nonzero(as_tuple=False).flatten().tolist()
    if span in ("all", "auto"):
        # `auto` means assistant if available, otherwise all.
        if span == "all":
            return real
        spans = ex.get("spans") or ex.get("span") or {}
        if isinstance(spans, dict) and ("assistant" in spans or "assistant_start" in spans):
            return span_indices(ex, "assistant", mask)
        return real

    spans = ex.get("spans") or ex.get("span") or {}
    if not isinstance(spans, dict):
        return real

    start = end = None
    if span in spans:
        val = spans[span]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            start, end = int(val[0]), int(val[1])
        elif isinstance(val, dict):
            start, end = int(val.get("start", 0)), int(val.get("end", len(mask)))
    else:
        sk = f"{span}_start"
        ek = f"{span}_end"
        if sk in spans and ek in spans:
            start, end = int(spans[sk]), int(spans[ek])
    if start is None or end is None:
        return real
    return [i for i in real if start <= i < end]


def window_token_indices(indices: list[int], max_tokens: int, token_window: str) -> list[int]:
    if max_tokens <= 0 or len(indices) <= max_tokens:
        return indices
    if token_window == "last":
        return indices[-max_tokens:]
    # `first` and `all` both become first max_tokens to keep fixed-size outputs.
    return indices[:max_tokens]


def place_in_canvas(arr: np.ndarray, canvas_shape: tuple[int, int], token_window: str) -> tuple[np.ndarray, np.ndarray]:
    """Place an LxT array into a fixed LxTmax canvas with NaNs."""
    L, Tmax = canvas_shape
    out = np.full((L, Tmax), np.nan, dtype=np.float32)
    valid = np.zeros((L, Tmax), dtype=bool)
    if arr.size == 0:
        return out, valid
    T = min(arr.shape[1], Tmax)
    if token_window == "last":
        out[:, -T:] = arr[:, -T:]
        valid[:, -T:] = True
    else:
        out[:, :T] = arr[:, :T]
        valid[:, :T] = True
    return out, valid


def causal_swim_1d(z: np.ndarray, window_size: int) -> np.ndarray:
    M = max(int(window_size), 1)
    out = np.zeros_like(z, dtype=np.float32)
    c = np.concatenate([[0.0], np.cumsum(z, dtype=np.float64)])
    for t in range(len(z)):
        s = max(0, t + 1 - M)
        out[t] = (c[t + 1] - c[s]) / (t + 1 - s)
    return out


def first_stable_crossing(values: np.ndarray, threshold: float, stable: int) -> int | None:
    if values.size == 0:
        return None
    good = values > threshold
    stable = max(int(stable), 1)
    run = 0
    for i, ok in enumerate(good.tolist()):
        run = run + 1 if ok else 0
        if run >= stable:
            return i - stable + 1
    return None


def first_stable_layer(mat: np.ndarray, threshold: float, stable_layers: int) -> int | None:
    """Earliest layer where any token has stable crossing over consecutive layers."""
    if mat.size == 0:
        return None
    good = mat > threshold
    stable_layers = max(int(stable_layers), 1)
    for t in range(good.shape[1]):
        run = 0
        for l in range(good.shape[0]):
            run = run + 1 if bool(good[l, t]) else 0
            if run >= stable_layers:
                return l - stable_layers + 1
    return None


def safe_nanmean(arrs: list[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(arrs, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0).astype(np.float32)


def save_heatmap(path: Path, mat: np.ndarray, title: str, source_layers: list[int | None], token_window: str):
    import matplotlib.pyplot as plt

    if mat.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(3, min(12, mat.shape[0] * 0.35))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title(title)
    ax.set_ylabel("selected layer")
    ax.set_xlabel("token position" + (" (right-aligned)" if token_window == "last" else ""))
    if source_layers and len(source_layers) == mat.shape[0]:
        ax.set_yticks(range(len(source_layers)))
        ax.set_yticklabels([str(x) if x is not None else str(i) for i, x in enumerate(source_layers)])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_hist(path: Path, values: list[int | None], title: str, xlabel: str):
    import matplotlib.pyplot as plt

    xs = [v for v in values if v is not None]
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(xs, bins=min(40, max(5, len(set(xs)))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@dataclass
class SampleRecord:
    sample_id: str
    path: str
    label: int
    n_tokens_used: int
    raw_label: object
    token_onset: int | None
    depth_onset_any: int | None
    depth_onset_final_token: int | None
    final_total_logit: float
    max_swim_logit: float


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_ckpt", required=True, help="Path to constitutional_probe.pt")
    ap.add_argument("--extracts_dir", required=True, help="Directory or comma-separated directories of .pt extracts")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default=None, help="cyber/refusal_gemma4_31b/refusal_qwen36; used for task label mapping")
    ap.add_argument("--task", default=None, help="Task name, e.g. refusal_gemma4_31b or cyber_3")
    ap.add_argument("--selection_json", default=None, help="Optional selection JSON to restrict sample IDs")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all", help="Optional official split filter")
    ap.add_argument("--span", choices=["auto", "all", "prompt", "user", "assistant"], default="auto")
    ap.add_argument("--token_window", choices=["last", "first", "all"], default="last")
    ap.add_argument("--max_tokens", type=int, default=256, help="Fixed heatmap width; <=0 uses longest observed length")
    ap.add_argument("--map_type", choices=["cumulative", "contribution", "layer_logit"], default="cumulative")
    ap.add_argument("--window_size", type=int, default=None, help="SWiM window for onset stats; default comes from checkpoint or 16")
    ap.add_argument("--threshold", type=float, default=0.0, help="Signed threshold for onset stats")
    ap.add_argument("--stable_tokens", type=int, default=4)
    ap.add_argument("--stable_layers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--save_per_sample", action="store_true", help="Save per-sample .npz maps. Can be large.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    arrays_dir = out_dir / "arrays"
    figs_dir = out_dir / "figures"
    per_sample_dir = out_dir / "per_sample"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_sample:
        per_sample_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(Path(args.probe_ckpt).expanduser())
    weight = ckpt["state_dict"]["weight"].float()
    bias = float(ckpt["state_dict"]["bias"].float().item())
    cfg = ckpt.get("config", {}) or {}
    swim_window = int(args.window_size or cfg.get("window_size", 16))

    selection_ids = load_selection_ids(args.selection_json)
    split_ids: set[str] | None = None
    if args.split != "all":
        if not args.dataset:
            raise ValueError("--split requires --dataset so data.py can recover split sample_ids")
        _, load_dataset = import_data_helpers()
        if load_dataset is None:
            raise ImportError("Could not import experiments/02_extract_activations/data.py for --split")
        split_ids = {s["sample_id"] for s in load_dataset(args.dataset, split=args.split)}

    paths = list(iter_pt_files(parse_dirs(args.extracts_dir)))
    pos_maps: list[np.ndarray] = []
    neg_maps: list[np.ndarray] = []
    pos_contrib: list[np.ndarray] = []
    neg_contrib: list[np.ndarray] = []
    pos_cum: list[np.ndarray] = []
    neg_cum: list[np.ndarray] = []
    pos_layer_logit: list[np.ndarray] = []
    neg_layer_logit: list[np.ndarray] = []
    records: list[SampleRecord] = []
    source_layers: list[int | None] = []
    skipped = {"selection": 0, "split": 0, "label": 0, "shape": 0, "empty": 0}

    # If max_tokens <= 0, collect dynamic width by doing one pass to find max; otherwise fixed.
    heatmap_width = int(args.max_tokens)
    if heatmap_width <= 0:
        max_len = 0
        for p in paths:
            ex = torch.load(str(p), map_location="cpu", weights_only=False)
            sid = str(ex.get("sample_id", p.stem))
            if selection_ids is not None and sid not in selection_ids:
                continue
            if split_ids is not None and sid not in split_ids:
                continue
            m = ex.get("attention_mask")
            if m is None:
                continue
            idx = span_indices(ex, args.span, m.bool())
            max_len = max(max_len, len(idx))
        heatmap_width = max_len
        if heatmap_width <= 0:
            raise ValueError("No usable token positions found.")

    for p_i, p in enumerate(paths):
        if args.max_samples and len(records) >= args.max_samples:
            break
        try:
            ex = torch.load(str(p), map_location="cpu", weights_only=False)
            sid = str(ex.get("sample_id", p.stem))
            if selection_ids is not None and sid not in selection_ids:
                skipped["selection"] += 1
                continue
            if split_ids is not None and sid not in split_ids:
                skipped["split"] += 1
                continue
            y = get_binary_label(ex, args.dataset, args.task)
            if y is None:
                skipped["label"] += 1
                continue

            r_all = residual_tensor(ex)
            positions, src_layers = resolve_layer_positions(ex, ckpt)
            if not source_layers:
                source_layers = src_layers
            r = r_all[torch.tensor(positions, dtype=torch.long)]
            if r.shape[0] != weight.shape[0] or r.shape[-1] != weight.shape[-1]:
                skipped["shape"] += 1
                print(
                    f"[warn] shape mismatch for {sid}: residual={tuple(r.shape)} weight={tuple(weight.shape)}",
                    flush=True,
                )
                continue
            mask = ex.get("attention_mask", torch.ones(r.shape[1], dtype=torch.bool)).bool()
            tok_idx = window_token_indices(span_indices(ex, args.span, mask), heatmap_width, args.token_window)
            if not tok_idx:
                skipped["empty"] += 1
                continue
            r = r[:, tok_idx, :]

            contrib = torch.einsum("ltd,ld->lt", r.float(), weight.float()).numpy().astype(np.float32)
            cumulative = np.cumsum(contrib, axis=0).astype(np.float32) + bias
            layer_logit = contrib + (bias / max(1, contrib.shape[0]))
            total = contrib.sum(axis=0).astype(np.float32) + bias
            swim = causal_swim_1d(total, swim_window)

            maps = {
                "contribution": contrib,
                "cumulative": cumulative,
                "layer_logit": layer_logit,
            }
            selected_map = maps[args.map_type]
            canvas, _ = place_in_canvas(selected_map, (weight.shape[0], heatmap_width), args.token_window)
            c_contrib, _ = place_in_canvas(contrib, (weight.shape[0], heatmap_width), args.token_window)
            c_cum, _ = place_in_canvas(cumulative, (weight.shape[0], heatmap_width), args.token_window)
            c_layer, _ = place_in_canvas(layer_logit, (weight.shape[0], heatmap_width), args.token_window)

            if y == 1:
                pos_maps.append(canvas)
                pos_contrib.append(c_contrib)
                pos_cum.append(c_cum)
                pos_layer_logit.append(c_layer)
            else:
                neg_maps.append(canvas)
                neg_contrib.append(c_contrib)
                neg_cum.append(c_cum)
                neg_layer_logit.append(c_layer)

            direction = 1.0 if y == 1 else -1.0
            token_onset = first_stable_crossing(direction * swim, args.threshold, args.stable_tokens)
            depth_onset_any = first_stable_layer(direction * cumulative, args.threshold, args.stable_layers)
            final_col = cumulative[:, -1:] if cumulative.shape[1] else cumulative
            depth_onset_final = first_stable_layer(direction * final_col, args.threshold, args.stable_layers)
            max_swim = float(np.nanmax(swim)) if swim.size else float("nan")
            records.append(
                SampleRecord(
                    sample_id=sid,
                    path=str(p),
                    label=int(y),
                    n_tokens_used=int(len(tok_idx)),
                    raw_label=ex.get("label"),
                    token_onset=token_onset,
                    depth_onset_any=depth_onset_any,
                    depth_onset_final_token=depth_onset_final,
                    final_total_logit=float(total[-1]) if total.size else float("nan"),
                    max_swim_logit=max_swim,
                )
            )
            if args.save_per_sample:
                np.savez_compressed(
                    per_sample_dir / f"{sid}.npz",
                    contribution=contrib,
                    cumulative=cumulative,
                    layer_logit=layer_logit,
                    total_token_logit=total,
                    swim_logit=swim,
                    token_indices=np.asarray(tok_idx, dtype=np.int64),
                    label=np.asarray([y], dtype=np.int64),
                )
        except Exception as e:
            skipped["shape"] += 1
            print(f"[warn] failed on {p.name}: {type(e).__name__}: {str(e)[:180]}", flush=True)

    if not records:
        raise ValueError(f"No usable samples. Skipped counts: {skipped}")

    aggregates = {
        f"pos_{args.map_type}": safe_nanmean(pos_maps),
        f"neg_{args.map_type}": safe_nanmean(neg_maps),
        "pos_contribution": safe_nanmean(pos_contrib),
        "neg_contribution": safe_nanmean(neg_contrib),
        "pos_cumulative": safe_nanmean(pos_cum),
        "neg_cumulative": safe_nanmean(neg_cum),
        "pos_layer_logit": safe_nanmean(pos_layer_logit),
        "neg_layer_logit": safe_nanmean(neg_layer_logit),
    }
    if aggregates[f"pos_{args.map_type}"].size and aggregates[f"neg_{args.map_type}"].size:
        aggregates[f"diff_pos_minus_neg_{args.map_type}"] = (
            aggregates[f"pos_{args.map_type}"] - aggregates[f"neg_{args.map_type}"]
        ).astype(np.float32)

    for name, arr in aggregates.items():
        if arr.size:
            np.save(arrays_dir / f"{name}.npy", arr)
            save_heatmap(
                figs_dir / f"{name}.png",
                arr,
                title=name.replace("_", " "),
                source_layers=source_layers,
                token_window=args.token_window,
            )

    # Histograms/onset summaries.
    pos_token_onsets = [r.token_onset for r in records if r.label == 1]
    neg_token_onsets = [r.token_onset for r in records if r.label == 0]
    pos_depth_onsets = [r.depth_onset_any for r in records if r.label == 1]
    neg_depth_onsets = [r.depth_onset_any for r in records if r.label == 0]
    save_hist(figs_dir / "pos_token_onset_hist.png", pos_token_onsets, "positive token onset", "token index in analysis window")
    save_hist(figs_dir / "neg_token_onset_hist.png", neg_token_onsets, "negative token onset", "token index in analysis window")
    save_hist(figs_dir / "pos_depth_onset_hist.png", pos_depth_onsets, "positive depth onset", "selected layer index")
    save_hist(figs_dir / "neg_depth_onset_hist.png", neg_depth_onsets, "negative depth onset", "selected layer index")

    csv_path = out_dir / "sample_onsets.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "label",
                "raw_label",
                "n_tokens_used",
                "token_onset",
                "depth_onset_any",
                "depth_onset_final_token",
                "final_total_logit",
                "max_swim_logit",
                "path",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sample_id": r.sample_id,
                "label": r.label,
                "raw_label": r.raw_label,
                "n_tokens_used": r.n_tokens_used,
                "token_onset": "" if r.token_onset is None else r.token_onset,
                "depth_onset_any": "" if r.depth_onset_any is None else r.depth_onset_any,
                "depth_onset_final_token": "" if r.depth_onset_final_token is None else r.depth_onset_final_token,
                "final_total_logit": r.final_total_logit,
                "max_swim_logit": r.max_swim_logit,
                "path": r.path,
            })

    summary = {
        "probe_ckpt": str(Path(args.probe_ckpt).expanduser()),
        "extracts_dir": args.extracts_dir,
        "dataset": args.dataset,
        "task": args.task,
        "n_records": len(records),
        "n_positive": sum(r.label == 1 for r in records),
        "n_negative": sum(r.label == 0 for r in records),
        "source_layers": source_layers,
        "map_type": args.map_type,
        "span": args.span,
        "token_window": args.token_window,
        "max_tokens": heatmap_width,
        "window_size": swim_window,
        "threshold": args.threshold,
        "stable_tokens": args.stable_tokens,
        "stable_layers": args.stable_layers,
        "skipped": skipped,
        "outputs": {
            "arrays": str(arrays_dir),
            "figures": str(figs_dir),
            "sample_onsets_csv": str(csv_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
