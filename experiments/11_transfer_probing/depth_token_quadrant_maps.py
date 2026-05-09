"""Depth x relative-token maps grouped by transfer-analysis quadrant.

Use this after running score_cyber_probe_on_refusal.py. It takes a frozen
Constitutional-style linear probe checkpoint, the refusal activation extracts,
and transfer_scores.csv. It then groups samples by e.g.

  high_cyber_refusal
  high_cyber_compliance
  low_cyber_refusal
  low_cyber_compliance

and saves one depth x token heatmap per group plus pairwise difference maps.

This is designed for the question:

  Among prompts that look cyber-bad to a frozen cyber_3 probe, do the ones
  Gemma refuses have different token/depth behavior from the ones Gemma accepts?

Expected checkpoint: constitutional_probe.pt from train_constitutional_probe.py
Expected extracts: .pt files with residuals (L,T,D), attention_mask, layer_idxs,
                   label, sample_id.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


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


def load_transfer_groups(path: Path, group_col: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    groups: dict[str, str] = {}
    rows: dict[str, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column sample_id")
        if group_col not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing group column {group_col!r}; columns={reader.fieldnames}")
        for row in reader:
            sid = str(row["sample_id"])
            g = str(row[group_col])
            if not sid or not g:
                continue
            groups[sid] = g
            rows[sid] = row
    if not groups:
        raise ValueError(f"No sample groups loaded from {path}")
    return groups, rows


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        ckpt = {"state_dict": ckpt, "config": {}}
    sd = ckpt["state_dict"]
    fixed = {}
    for k, v in sd.items():
        kk = k
        for prefix in ("module.", "model.", "probe."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        fixed[kk] = v
    ckpt["state_dict"] = fixed
    if "weight" not in fixed:
        raise KeyError("Checkpoint does not contain state_dict['weight']; expected Constitutional linear probe")
    if "bias" not in fixed:
        fixed["bias"] = torch.tensor(0.0)
    return ckpt


def residual_tensor(ex: dict) -> torch.Tensor:
    if "residuals" in ex:
        r = ex["residuals"]
    elif "middle_layer_all_tokens" in ex:
        r = ex["middle_layer_all_tokens"].unsqueeze(0)
    else:
        raise KeyError("extract is missing residuals or middle_layer_all_tokens")
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
    elif "source_layer_idxs" in cfg and "layer_idxs" in ex:
        source_to_pos = {int(src): i for i, src in enumerate(ex["layer_idxs"])}
        missing = [int(src) for src in cfg["source_layer_idxs"] if int(src) not in source_to_pos]
        if missing:
            raise ValueError(
                f"extract missing trained source layers {missing}; extract has layer_idxs={ex.get('layer_idxs')}"
            )
        positions = [source_to_pos[int(src)] for src in cfg["source_layer_idxs"]]
    else:
        positions = list(range(n_probe_layers))

    if len(positions) != n_probe_layers:
        raise ValueError(f"checkpoint uses {n_probe_layers} probe layers but resolved {len(positions)} positions")
    if max(positions, default=-1) >= n_extract_layers:
        raise ValueError(f"positions {positions} out of range for extract with {n_extract_layers} layers")

    if "layer_idxs" in ex:
        source_layers: list[int | None] = [int(ex["layer_idxs"][p]) for p in positions]
    else:
        source_layers = [None for _ in positions]
    return positions, source_layers


def span_indices(ex: dict, span: str, mask: torch.Tensor) -> list[int]:
    real = mask.bool().nonzero(as_tuple=False).flatten().tolist()
    if span in ("all", "auto"):
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
        sk, ek = f"{span}_start", f"{span}_end"
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
    return indices[:max_tokens]


def place_in_canvas(arr: np.ndarray, canvas_shape: tuple[int, int], token_window: str) -> np.ndarray:
    L, Tmax = canvas_shape
    out = np.full((L, Tmax), np.nan, dtype=np.float32)
    if arr.size == 0:
        return out
    T = min(arr.shape[1], Tmax)
    if token_window == "last":
        out[:, -T:] = arr[:, -T:]
    else:
        out[:, :T] = arr[:, :T]
    return out


def resample_relative_bins(arr: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = max(int(n_bins), 1)
    if arr.size == 0:
        L = arr.shape[0] if arr.ndim else 0
        return np.full((L, n_bins), np.nan, dtype=np.float32)
    L, T = arr.shape
    out = np.full((L, n_bins), np.nan, dtype=np.float32)
    if T == 1:
        out[:, 0] = arr[:, 0]
        return out
    bin_ids = np.floor(np.arange(T, dtype=np.float64) * n_bins / T).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    for b in range(n_bins):
        cols = np.where(bin_ids == b)[0]
        if cols.size:
            with np.errstate(invalid="ignore"):
                out[:, b] = np.nanmean(arr[:, cols], axis=1)
    return out


def safe_nanmean(arrs: list[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(arrs, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0).astype(np.float32)


def robust_limits(mat: np.ndarray, percentile: float, symmetric: bool) -> tuple[float | None, float | None]:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return None, None
    if percentile <= 0 or percentile >= 100:
        if symmetric:
            m = float(np.nanmax(np.abs(vals)))
            return -m, m
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    if symmetric:
        m = float(np.nanpercentile(np.abs(vals), percentile))
        return -m, m
    lo = float(np.nanpercentile(vals, 100 - percentile))
    hi = float(np.nanpercentile(vals, percentile))
    return lo, hi


def save_heatmap(
    path: Path,
    mat: np.ndarray,
    title: str,
    source_layers: list[int | None],
    position_mode: str,
    token_window: str,
    clip_percentile: float,
    symmetric: bool,
):
    import matplotlib.pyplot as plt

    if mat.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(3, min(12, mat.shape[0] * 0.35))))
    vmin, vmax = robust_limits(mat, clip_percentile, symmetric)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel("selected layer")
    if source_layers and len(source_layers) == mat.shape[0]:
        ax.set_yticks(range(len(source_layers)))
        ax.set_yticklabels([str(x) if x is not None else str(i) for i, x in enumerate(source_layers)])
    if position_mode == "relative":
        ax.set_xlabel("relative position in selected span (%)")
        if mat.shape[1] > 1:
            ticks = np.linspace(0, mat.shape[1] - 1, 6)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 100, 6)])
    else:
        ax.set_xlabel("token position" + (" (right-aligned)" if token_window == "last" else ""))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_ckpt", required=True)
    ap.add_argument("--extracts_dir", required=True, help="Directory or comma-separated directories of .pt extracts")
    ap.add_argument("--transfer_scores_csv", required=True, help="CSV produced by score_cyber_probe_on_refusal.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_col", default="quadrant", help="Column in transfer_scores.csv to group by")
    ap.add_argument("--groups", default="", help="Comma-separated groups to plot. Default: all groups in CSV")
    ap.add_argument(
        "--compare_groups",
        action="append",
        default=[],
        help="Pair to difference as A,B. Can be passed multiple times.",
    )
    ap.add_argument("--span", choices=["auto", "all", "prompt", "user", "assistant"], default="auto")
    ap.add_argument("--position_mode", choices=["absolute", "relative"], default="relative")
    ap.add_argument("--relative_bins", type=int, default=100)
    ap.add_argument("--token_window", choices=["last", "first", "all"], default="last")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument(
        "--map_types",
        default="contribution,cumulative,layer_logit",
        help="Comma-separated from contribution,cumulative,layer_logit",
    )
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--clip_percentile", type=float, default=99.0, help="Robust color scaling percentile; set 100 for full range")
    ap.add_argument("--save_arrays", action="store_true", help="Save .npy aggregate arrays")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figures"
    arrays_dir = out_dir / "arrays"
    figs_dir.mkdir(parents=True, exist_ok=True)
    if args.save_arrays:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    map_types = [x.strip() for x in args.map_types.split(",") if x.strip()]
    valid = {"contribution", "cumulative", "layer_logit"}
    bad = [m for m in map_types if m not in valid]
    if bad:
        raise ValueError(f"Invalid map_types {bad}; valid={sorted(valid)}")

    group_for_sid, transfer_rows = load_transfer_groups(Path(args.transfer_scores_csv).expanduser(), args.group_col)
    if args.groups:
        target_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    else:
        target_groups = sorted(set(group_for_sid.values()))
    target_set = set(target_groups)

    ckpt = load_checkpoint(Path(args.probe_ckpt).expanduser())
    weight = ckpt["state_dict"]["weight"].float()
    bias = float(ckpt["state_dict"]["bias"].float().item())

    paths = list(iter_pt_files(parse_dirs(args.extracts_dir)))
    if args.position_mode == "relative":
        heatmap_width = int(args.relative_bins)
        if heatmap_width <= 0:
            raise ValueError("--relative_bins must be positive")
    else:
        heatmap_width = int(args.max_tokens)
        if heatmap_width <= 0:
            # Dynamic absolute width: longest selected span among target groups.
            heatmap_width = 0
            for p in paths:
                ex = torch.load(str(p), map_location="cpu", weights_only=False)
                sid = str(ex.get("sample_id", p.stem))
                if group_for_sid.get(sid) not in target_set:
                    continue
                mask = ex.get("attention_mask")
                if mask is None:
                    continue
                heatmap_width = max(heatmap_width, len(span_indices(ex, args.span, mask.bool())))
            if heatmap_width <= 0:
                raise ValueError("No usable token positions found for selected groups")

    accum: dict[str, dict[str, list[np.ndarray]]] = {
        g: {m: [] for m in map_types} for g in target_groups
    }
    counts = {g: 0 for g in target_groups}
    source_layers: list[int | None] = []
    skipped = defaultdict(int)
    errors: list[dict[str, str]] = []

    for p in paths:
        if args.max_samples and sum(counts.values()) >= args.max_samples:
            break
        try:
            ex = torch.load(str(p), map_location="cpu", weights_only=False)
            sid = str(ex.get("sample_id", p.stem))
            group = group_for_sid.get(sid)
            if group is None:
                skipped["not_in_transfer_csv"] += 1
                continue
            if group not in target_set:
                skipped["not_in_requested_groups"] += 1
                continue

            r_all = residual_tensor(ex)
            positions, src_layers = resolve_layer_positions(ex, ckpt)
            if not source_layers:
                source_layers = src_layers
            r = r_all[torch.tensor(positions, dtype=torch.long)]
            if r.shape[0] != weight.shape[0] or r.shape[-1] != weight.shape[-1]:
                raise ValueError(f"shape mismatch: residual={tuple(r.shape)} weight={tuple(weight.shape)}")

            mask = ex.get("attention_mask", torch.ones(r.shape[1], dtype=torch.bool)).bool()
            span_tok_idx = span_indices(ex, args.span, mask)
            if args.position_mode == "relative":
                tok_idx = span_tok_idx
            else:
                tok_idx = window_token_indices(span_tok_idx, heatmap_width, args.token_window)
            if not tok_idx:
                skipped["empty_span"] += 1
                continue
            r = r[:, tok_idx, :]

            contrib = torch.einsum("ltd,ld->lt", r.float(), weight.float()).numpy().astype(np.float32)
            cumulative = np.cumsum(contrib, axis=0).astype(np.float32) + bias
            layer_logit = contrib + (bias / max(1, contrib.shape[0]))
            maps = {
                "contribution": contrib,
                "cumulative": cumulative,
                "layer_logit": layer_logit,
            }
            for m in map_types:
                if args.position_mode == "relative":
                    arr = resample_relative_bins(maps[m], heatmap_width)
                else:
                    arr = place_in_canvas(maps[m], (weight.shape[0], heatmap_width), args.token_window)
                accum[group][m].append(arr)
            counts[group] += 1
        except Exception as e:
            skipped["errors"] += 1
            if len(errors) < 20:
                errors.append({"path": str(p), "error": f"{type(e).__name__}: {str(e)[:220]}"})

    aggregates: dict[str, dict[str, np.ndarray]] = {g: {} for g in target_groups}
    for g in target_groups:
        for m in map_types:
            aggregates[g][m] = safe_nanmean(accum[g][m])
            if aggregates[g][m].size:
                name = f"{sanitize(g)}_{m}"
                if args.save_arrays:
                    np.save(arrays_dir / f"{name}.npy", aggregates[g][m])
                save_heatmap(
                    figs_dir / f"{name}.png",
                    aggregates[g][m],
                    title=name.replace("_", " "),
                    source_layers=source_layers,
                    position_mode=args.position_mode,
                    token_window=args.token_window,
                    clip_percentile=args.clip_percentile,
                    symmetric=False,
                )

    comparisons = []
    for pair in args.compare_groups:
        parts = [x.strip() for x in pair.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"--compare_groups expects A,B; got {pair!r}")
        a, b = parts
        comparisons.append((a, b))
    if not comparisons and {"high_cyber_refusal", "high_cyber_compliance"}.issubset(target_set):
        comparisons.append(("high_cyber_refusal", "high_cyber_compliance"))
    if not comparisons and len(target_groups) >= 2:
        comparisons.append((target_groups[0], target_groups[1]))

    for a, b in comparisons:
        if a not in aggregates or b not in aggregates:
            print(f"[warn] comparison {a},{b} skipped: missing group", file=sys.stderr)
            continue
        for m in map_types:
            A = aggregates[a].get(m, np.empty((0, 0)))
            B = aggregates[b].get(m, np.empty((0, 0)))
            if not A.size or not B.size:
                continue
            diff = (A - B).astype(np.float32)
            name = f"diff_{sanitize(a)}_minus_{sanitize(b)}_{m}"
            if args.save_arrays:
                np.save(arrays_dir / f"{name}.npy", diff)
            save_heatmap(
                figs_dir / f"{name}.png",
                diff,
                title=name.replace("_", " "),
                source_layers=source_layers,
                position_mode=args.position_mode,
                token_window=args.token_window,
                clip_percentile=args.clip_percentile,
                symmetric=True,
            )

    summary = {
        "probe_ckpt": str(Path(args.probe_ckpt).expanduser()),
        "extracts_dir": args.extracts_dir,
        "transfer_scores_csv": str(Path(args.transfer_scores_csv).expanduser()),
        "group_col": args.group_col,
        "groups": target_groups,
        "counts": counts,
        "source_layers": source_layers,
        "map_types": map_types,
        "position_mode": args.position_mode,
        "relative_bins": args.relative_bins,
        "span": args.span,
        "token_window": args.token_window,
        "max_tokens": heatmap_width,
        "comparisons": comparisons,
        "skipped": dict(skipped),
        "first_errors": errors,
        "figures_dir": str(figs_dir),
        "arrays_dir": str(arrays_dir) if args.save_arrays else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
