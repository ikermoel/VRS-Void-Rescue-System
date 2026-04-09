#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_DIM = 512
NUM_LAYERS = 9
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = MODEL_DIM // NUM_HEADS
MLP_MULT = 2
ROPE_BASE = 10000.0


class StorageType:
    def __init__(self, name: str, dtype: np.dtype):
        self.name = name
        self.dtype = np.dtype(dtype)


class StorageRef:
    def __init__(self, storage_type: StorageType, key: str, location: str, numel: int):
        self.storage_type = storage_type
        self.key = str(key)
        self.location = location
        self.numel = int(numel)


class TensorRef:
    def __init__(
        self,
        storage: StorageRef,
        offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
    ):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(x) for x in size)
        self.stride = tuple(int(x) for x in stride)


def rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks):
    return TensorRef(storage, offset, tuple(size), tuple(stride))


class TorchPickleReader(pickle.Unpickler):
    STORAGE_DTYPES = {
        "FloatStorage": np.float32,
        "HalfStorage": np.float16,
        "CharStorage": np.int8,
        "BFloat16Storage": np.uint16,
    }

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name in self.STORAGE_DTYPES:
            return StorageType(name, self.STORAGE_DTYPES[name])
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        tag, storage_type, key, location, numel = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
        return StorageRef(storage_type, key, location, numel)


def bfloat16_to_float32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


def tensor_from_ref(zf: zipfile.ZipFile, prefix: str, ref: TensorRef) -> np.ndarray:
    raw = zf.read(f"{prefix}/data/{ref.storage.key}")
    stype = ref.storage.storage_type.name
    if stype == "BFloat16Storage":
        base = bfloat16_to_float32(np.frombuffer(raw, dtype="<u2"))
    else:
        base = np.frombuffer(raw, dtype=ref.storage.storage_type.dtype)
        if base.dtype.byteorder == ">":
            base = base.byteswap().newbyteorder()

    shape = ref.size
    if ref.offset:
        base = base[ref.offset :]

    needed = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if ref.stride != contiguous_stride(shape):
        byte_strides = tuple(s * base.dtype.itemsize for s in ref.stride)
        arr = np.lib.stride_tricks.as_strided(base, shape=shape, strides=byte_strides)
        return np.array(arr)
    return np.array(base[:needed].reshape(shape))


def resolve_refs(obj, zf: zipfile.ZipFile, prefix: str):
    if isinstance(obj, TensorRef):
        return tensor_from_ref(zf, prefix, obj)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, resolve_refs(v, zf, prefix)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: resolve_refs(v, zf, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(v, zf, prefix) for v in obj]
    if isinstance(obj, tuple):
        return tuple(resolve_refs(v, zf, prefix) for v in obj)
    return obj


def read_torch_zip(raw: bytes):
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        data_pkl = next(name for name in zf.namelist() if name.endswith("/data.pkl"))
        prefix = data_pkl.rsplit("/", 1)[0]
        meta = TorchPickleReader(io.BytesIO(zf.read(data_pkl))).load()
        return resolve_refs(meta, zf, prefix)


def load_state_dict(path: Path) -> OrderedDict[str, np.ndarray]:
    raw = path.read_bytes()
    if path.suffix == ".ptz":
        raw = zlib.decompress(raw)
    obj = read_torch_zip(raw)

    if isinstance(obj, OrderedDict):
        return OrderedDict((k, v.astype(np.float32, copy=False)) for k, v in obj.items())

    if obj.get("__quant_format__") != "int8_clean_per_row_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")

    out: OrderedDict[str, np.ndarray] = OrderedDict()
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        scale = obj["scales"][name].astype(np.float32, copy=False)
        q32 = q.astype(np.float32, copy=False)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            scale = scale.reshape((scale.shape[0],) + (1,) * (q32.ndim - 1))
        out[name] = q32 * scale

    for name, value in obj["passthrough"].items():
        out[name] = value.astype(np.float32, copy=False)
    return out


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x @ weight.T


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def apply_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), axis=-1)


def rotary_tables(seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_BASE ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = np.outer(np.arange(seq_len, dtype=np.float32), inv_freq)
    return np.cos(freqs)[None, None, :, :].astype(np.float32), np.sin(freqs)[None, None, :, :].astype(np.float32)


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    repeat = NUM_HEADS // NUM_KV_HEADS
    if repeat != 1:
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(q.shape[-1])
    seq_len = q.shape[-2]
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores[..., mask] = -1.0e30
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.matmul(probs, v)


def attn_forward(w: dict[str, np.ndarray], layer: int, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    base = f"blocks.{layer}.attn"
    bsz, seq_len, _ = x.shape
    q = linear(x, w[f"{base}.c_q.weight"]).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = linear(x, w[f"{base}.c_k.weight"]).reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = linear(x, w[f"{base}.c_v.weight"]).reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    q = apply_rotary(rms_norm(q), cos, sin)
    k = apply_rotary(rms_norm(k), cos, sin)
    q = q * w[f"{base}.q_gain"][None, :, None, None]
    y = causal_attention(q, k, v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, MODEL_DIM)
    return linear(y, w[f"{base}.proj.weight"])


def mlp_forward(w: dict[str, np.ndarray], layer: int, x: np.ndarray) -> np.ndarray:
    h = np.maximum(linear(x, w[f"blocks.{layer}.mlp.fc.weight"]), 0.0) ** 2
    return linear(h, w[f"blocks.{layer}.mlp.proj.weight"])


def block_forward(w: dict[str, np.ndarray], layer: int, x: np.ndarray, x0: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    mix = w[f"blocks.{layer}.resid_mix"]
    x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    x = x + w[f"blocks.{layer}.attn_scale"][None, None, :] * attn_forward(w, layer, rms_norm(x), cos, sin)
    x = x + w[f"blocks.{layer}.mlp_scale"][None, None, :] * mlp_forward(w, layer, rms_norm(x))
    return x.astype(np.float32, copy=False)


def navigator_forward(w: dict[str, np.ndarray], input_ids: np.ndarray) -> np.ndarray:
    _, seq_len = input_ids.shape
    cos, sin = rotary_tables(seq_len)
    x = rms_norm(w["tok_emb.weight"][input_ids])
    x0 = x.copy()
    skips = []
    encoder_layers = NUM_LAYERS // 2
    decoder_layers = NUM_LAYERS - encoder_layers
    for layer in range(encoder_layers):
        x = block_forward(w, layer, x, x0, cos, sin)
        skips.append(x)
    for i in range(decoder_layers):
        if skips:
            x = x + w["skip_weights"][i][None, None, :] * skips.pop()
        x = block_forward(w, encoder_layers + i, x, x0, cos, sin)
    return rms_norm(x)


def rescuer_forward(w: dict[str, np.ndarray], v_void: np.ndarray) -> np.ndarray:
    return linear(silu(linear(v_void, w["rescuer.fc.weight"])), w["rescuer.proj.weight"])


def pca_project(points: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0, keepdims=True)
    centered = points - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:dims].T
    explained = (singular_values[:dims] ** 2) / np.sum(singular_values ** 2)
    return coords, explained, mean


def select_probe_examples(w: dict[str, np.ndarray], batches: int, seq_len: int, seed: int):
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(3, w["tok_emb.weight"].shape[0], size=(batches, seq_len + 1), dtype=np.int64)
    x = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]

    v_void = navigator_forward(w, x)
    v_rescued = rescuer_forward(w, v_void)
    target = w["tok_emb.weight"][target_ids]

    d_void = np.linalg.norm(v_void - target, axis=-1)
    d_rescued = np.linalg.norm(v_rescued - target, axis=-1)
    reduction = d_void - d_rescued
    improvement_pct = 100.0 * reduction / np.maximum(d_void, 1e-12)

    flat_order = np.argsort(reduction.reshape(-1))[::-1]
    chosen = []
    for flat_idx in flat_order:
        b, pos = np.unravel_index(flat_idx, reduction.shape)
        if reduction[b, pos] <= 0:
            continue
        chosen.append((int(b), int(pos)))
        if len(chosen) == 5:
            break
    if not chosen:
        chosen = [tuple(map(int, np.unravel_index(int(flat_order[0]), reduction.shape)))]

    stats = {
        "probe_batches": int(batches),
        "probe_seq_len": int(seq_len),
        "probe_positions": int(d_void.size),
        "mean_void_distance": float(d_void.mean()),
        "mean_rescued_distance": float(d_rescued.mean()),
        "mean_improvement_pct": float(improvement_pct.mean()),
        "median_improvement_pct": float(np.median(improvement_pct)),
        "improved_fraction": float(np.mean(reduction > 0.0)),
    }

    rows = []
    for rank, (b, pos) in enumerate(chosen, start=1):
        rows.append(
            {
                "rank": rank,
                "batch": b,
                "position": pos,
                "input_token_id": int(x[b, pos]),
                "target_token_id": int(target_ids[b, pos]),
                "d_navigator_to_target": float(d_void[b, pos]),
                "d_rescuer_to_target": float(d_rescued[b, pos]),
                "absolute_distance_reduction": float(reduction[b, pos]),
                "relative_improvement_pct": float(improvement_pct[b, pos]),
                "v_void": v_void[b, pos].copy(),
                "v_rescued": v_rescued[b, pos].copy(),
                "target": target[b, pos].copy(),
            }
        )
    return rows, stats


def make_projection_payload(w: dict[str, np.ndarray], rows: list[dict], seed: int):
    rng = np.random.default_rng(seed + 99)
    background_idx = rng.choice(w["tok_emb.weight"].shape[0], size=min(512, w["tok_emb.weight"].shape[0]), replace=False)
    background = w["tok_emb.weight"][background_idx]
    special = []
    labels = []
    for row in rows:
        special.extend([row["v_void"], row["v_rescued"], row["target"]])
        labels.extend([f"A #{row['rank']}", f"Rescuer #{row['rank']}", f"Target #{row['rank']}"])
    all_points = np.vstack([background, np.asarray(special, dtype=np.float32)])
    coords3, explained3, _ = pca_project(all_points, 3)
    coords2 = coords3[:, :2]
    return {
        "background_idx": background_idx,
        "background2": coords2[: len(background)],
        "background3": coords3[: len(background)],
        "special2": coords2[len(background) :],
        "special3": coords3[len(background) :],
        "labels": labels,
        "explained3": explained3,
    }


def plot_2d(payload: dict, rows: list[dict], stats: dict, out_base: Path):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    bg = payload["background2"]
    sp = payload["special2"]
    ax.scatter(bg[:, 0], bg[:, 1], s=8, c="#c8c8c8", alpha=0.45, linewidths=0, label="Token embeddings")
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, row in enumerate(rows):
        a, r, t = sp[3 * i : 3 * i + 3]
        color = colors[i % len(colors)]
        ax.scatter([a[0]], [a[1]], s=60, marker="x", c="#d62728", linewidths=2)
        ax.scatter([r[0]], [r[1]], s=55, marker="o", facecolors="white", edgecolors=color, linewidths=1.8)
        ax.scatter([t[0]], [t[1]], s=70, marker="*", c="#2ca02c", edgecolors="black", linewidths=0.4)
        ax.annotate("", xy=r, xytext=a, arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        ax.annotate("", xy=t, xytext=r, arrowprops=dict(arrowstyle="->", linestyle="--", color="#555555", lw=1.0, alpha=0.8))
        ax.text(t[0], t[1], f"  #{row['rank']}", fontsize=8, va="center")
    exp = payload["explained3"]
    ax.set_title("JEPA-VRS x2 embedding-space correction")
    ax.set_xlabel(f"PC1 ({100 * exp[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({100 * exp[1]:.1f}% var.)")
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.text(
        0.02,
        0.02,
        f"Probe positions: {stats['probe_positions']} | improved: {100 * stats['improved_fraction']:.1f}% | "
        f"median improvement: {stats['median_improvement_pct']:.1f}%",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.95),
    )
    handles = [
        plt.Line2D([0], [0], marker="x", color="none", markeredgecolor="#d62728", markersize=8, label=r"Navigator $v_{void}$"),
        plt.Line2D([0], [0], marker="o", color="none", markeredgecolor="#1f77b4", markerfacecolor="white", markersize=7, label=r"Rescuer $v_{rescued}$"),
        plt.Line2D([0], [0], marker="*", color="none", markeredgecolor="black", markerfacecolor="#2ca02c", markersize=10, label=r"Target $e_{target}$"),
    ]
    ax.legend(handles=handles, frameon=True, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_3d(payload: dict, rows: list[dict], out_base: Path):
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    bg = payload["background3"]
    sp = payload["special3"]
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=6, c="#c8c8c8", alpha=0.25, depthshade=False)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, row in enumerate(rows):
        a, r, t = sp[3 * i : 3 * i + 3]
        color = colors[i % len(colors)]
        ax.scatter([a[0]], [a[1]], [a[2]], s=55, marker="x", c="#d62728", depthshade=False)
        ax.scatter([r[0]], [r[1]], [r[2]], s=50, marker="o", facecolors="white", edgecolors=color, depthshade=False)
        ax.scatter([t[0]], [t[1]], [t[2]], s=70, marker="*", c="#2ca02c", edgecolors="black", depthshade=False)
        ax.plot([a[0], r[0]], [a[1], r[1]], [a[2], r[2]], color=color, lw=1.8)
        ax.plot([r[0], t[0]], [r[1], t[1]], [r[2], t[2]], color="#555555", lw=1.0, ls="--")
    exp = payload["explained3"]
    ax.set_title("JEPA-VRS x2 correction vectors in 3D PCA")
    ax.set_xlabel(f"PC1 ({100 * exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * exp[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({100 * exp[2]:.1f}%)")
    ax.view_init(elev=23, azim=-58)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_distances(rows: list[dict], out_base: Path):
    ranks = np.arange(len(rows))
    d_a = np.array([r["d_navigator_to_target"] for r in rows])
    d_b = np.array([r["d_rescuer_to_target"] for r in rows])
    labels = [f"#{r['rank']}\ntok {r['target_token_id']}" for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.36
    ax.bar(ranks - width / 2, d_a, width, label="Navigator to target", color="#d62728", alpha=0.82)
    ax.bar(ranks + width / 2, d_b, width, label="Rescuer to target", color="#1f77b4", alpha=0.82)
    for i, row in enumerate(rows):
        ax.text(i, max(d_a[i], d_b[i]) + 0.02, f"{row['relative_improvement_pct']:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(ranks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 distance in 512D embedding space")
    ax.set_title("Rescuer reduces L2 distance to target embeddings")
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.8)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_base)


def save_figure(fig, out_base: Path):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_metrics(rows: list[dict], stats: dict, output_dir: Path):
    serializable_rows = []
    for row in rows:
        serializable_rows.append({k: v for k, v in row.items() if not isinstance(v, np.ndarray)})

    (output_dir / "jepa_vrs_geometry_metrics.json").write_text(
        json.dumps({"summary": stats, "examples": serializable_rows}, indent=2),
        encoding="utf-8",
    )

    csv_lines = [
        "rank,batch,position,input_token_id,target_token_id,d_navigator_to_target,d_rescuer_to_target,absolute_distance_reduction,relative_improvement_pct"
    ]
    for row in serializable_rows:
        csv_lines.append(
            ",".join(
                str(row[key])
                for key in [
                    "rank",
                    "batch",
                    "position",
                    "input_token_id",
                    "target_token_id",
                    "d_navigator_to_target",
                    "d_rescuer_to_target",
                    "absolute_distance_reduction",
                    "relative_improvement_pct",
                ]
            )
        )
    (output_dir / "jepa_vrs_geometry_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def write_caption(stats: dict, output_dir: Path):
    text = f"""Figure caption draft:
JEPA-VRS x2 geometric correction in embedding space. The Navigator (Model A) produces a context-conditioned vector v_void; the Rescuer maps it to v_rescued, closer to the correct token embedding target_emb. Points are projected from the original 512-dimensional embedding space using PCA for visualization only; all reported distances are computed in the original 512D space. The visualization uses deterministic local token-ID probes because no tokenizer or validation shard is present in this x2 folder. Across {stats['probe_positions']} probed positions, {100 * stats['improved_fraction']:.1f}% moved closer after the Rescuer, with mean distance {stats['mean_void_distance']:.4f} -> {stats['mean_rescued_distance']:.4f} and median relative improvement {stats['median_improvement_pct']:.2f}%.
"""
    (output_dir / "jepa_vrs_geometry_caption.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate JEPA-VRS x2 Navigator/Rescuer embedding-space figures.")
    parser.add_argument("--weights", default="final_jepa_vrs_seed42_x2.int8.ptz", help="Path to the x2 .ptz or .pt weights file.")
    parser.add_argument("--output-dir", default="paper_figures", help="Directory for generated figures and metrics.")
    parser.add_argument("--batches", type=int, default=24, help="Number of deterministic token-ID probe sequences.")
    parser.add_argument("--seq-len", type=int, default=24, help="Probe sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic probe seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights: {args.weights}")
    weights = load_state_dict(Path(args.weights))
    print(f"Loaded tensors: {len(weights)}")
    print(f"Running NumPy forward pass on {args.batches} x {args.seq_len} probe positions...")
    rows, stats = select_probe_examples(weights, args.batches, args.seq_len, args.seed)
    stats["weights"] = str(Path(args.weights))
    payload = make_projection_payload(weights, rows, args.seed)

    print("Writing figures...")
    plot_2d(payload, rows, stats, output_dir / "jepa_vrs_rescuer_geometry_2d")
    plot_3d(payload, rows, output_dir / "jepa_vrs_rescuer_geometry_3d")
    plot_distances(rows, output_dir / "jepa_vrs_rescuer_distance_bars")
    write_metrics(rows, stats, output_dir)
    write_caption(stats, output_dir)
    print(f"Done: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
