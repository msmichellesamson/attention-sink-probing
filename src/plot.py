"""
Plotting script for attention-sink-probing experiments.

Reads probe_scores/ and sink_tokens/ JSONs, produces:
  1. A heatmap: model x layer, colored by probe AUROC
  2. A line chart: "first separable layer" vs model size (param count)

Run this after probe.py has finished. Saves to results/figures/.

Usage:
    python src/plot.py
    python src/plot.py --scores_dir my_run/probe_scores --out results/figures/

NOTE: matplotlib's default color cycle is ugly for multi-line plots.
Using a hand-picked palette that prints reasonably in grayscale too.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - no display needed on the cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Model metadata: canonical ordering + approx param counts for the x-axis
# Pythia models I actually ran probes on. If you add a new scale, add it here.
# ---------------------------------------------------------------------------
MODEL_META = {
    "pythia-70m":   {"params_m": 70,    "n_layers": 6},
    "pythia-160m":  {"params_m": 160,   "n_layers": 12},
    "pythia-410m":  {"params_m": 410,   "n_layers": 24},
    "pythia-1b":    {"params_m": 1000,  "n_layers": 16},
    "pythia-1.4b":  {"params_m": 1400,  "n_layers": 24},
    "pythia-2.8b":  {"params_m": 2800,  "n_layers": 32},
    "pythia-6.9b":  {"params_m": 6900,  "n_layers": 32},
    "pythia-12b":   {"params_m": 12000, "n_layers": 36},
}

# Threshold for "linearly separable enough to call it learned"
# I tried 0.70 and 0.75 - 0.72 felt like the right call given the variance
# in probe scores at early layers. See findings in README.
SEPARABILITY_THRESHOLD = 0.72

# Palette: diverging-ish but not RdBu (overused). ColorBrewer YlOrRd works
# well for "low is boring, high is interesting" framing.
HEATMAP_CMAP = "YlOrRd"

# For the line chart: distinct enough colors, not rainbow garbage
LINE_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_probe_scores(scores_dir: Path) -> dict[str, dict[int, float]]:
    """
    Load all probe score JSONs from scores_dir.

    Expected filename format: {model_name}_layer{layer_idx}.json
    Each JSON has at minimum: {"auroc": float, "layer": int, "model": str}

    Returns: {model_name: {layer_idx: auroc}}
    """
    scores = {}

    json_files = sorted(scores_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {scores_dir}")

    print(f"Found {len(json_files)} probe score files in {scores_dir}")

    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)

        model = data.get("model")
        layer = data.get("layer")
        auroc = data.get("auroc")

        # Graceful skip rather than crash - some runs may have failed
        if any(v is None for v in [model, layer, auroc]):
            print(f"  WARNING: skipping {fpath.name} - missing field(s)")
            continue

        if model not in scores:
            scores[model] = {}
        scores[model][int(layer)] = float(auroc)

    print(f"Loaded scores for {len(scores)} models: {sorted(scores.keys())}")
    return scores


def load_sink_stats(sinks_dir: Path) -> dict[str, dict]:
    """
    Load sink identification JSONs (output of identify_sinks.py).

    Expected: one JSON per model, containing at least:
      {"model": str, "bos_sink_fraction": float, "mean_sink_attn": float}

    Not strictly required for the plots but useful for annotations.
    Returns empty dict if dir doesn't exist - plots still work.
    """
    if not sinks_dir.exists():
        print(f"NOTE: sinks dir {sinks_dir} not found, skipping sink annotations")
        return {}

    sink_stats = {}
    for fpath in sinks_dir.glob("*.json"):
        with open(fpath) as f:
            data = json.load(f)
        model = data.get("model")
        if model:
            sink_stats[model] = data

    print(f"Loaded sink stats for {len(sink_stats)} models")
    return sink_stats


# ---------------------------------------------------------------------------
# Core data munging
# ---------------------------------------------------------------------------

def build_auroc_matrix(
    scores: dict[str, dict[int, float]]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a (models x layers) DataFrame of AUROC values.

    Models ordered by param count (smallest to largest).
    Layers are 0-indexed integers.
    Missing values (layer not probed, or run failed) become NaN.
    """
    # Sort models by known param count, put unknowns at end
    def sort_key(m):
        return MODEL_META.get(m, {}).get("params_m", 99999)

    ordered_models = sorted(scores.keys(), key=sort_key)

    # Find the max layer index across all models
    all_layers = set()
    for layer_dict in scores.values():
        all_layers.update(layer_dict.keys())
    max_layer = max(all_layers)

    # Build the matrix - lots of NaN is expected (different models have
    # different numbers of layers)
    rows = []
    for model in ordered_models:
        row = {
            layer: scores[model].get(layer, np.nan)
            for layer in range(max_layer + 1)
        }
        rows.append(row)

    df = pd.DataFrame(rows, index=ordered_models)
    df.index.name = "model"
    df.columns.name = "layer"

    return df, ordered_models


def find_first_separable_layer(
    scores: dict[int, float],
    threshold: float = SEPARABILITY_THRESHOLD,
    min_consecutive: int = 2,
) -> int | None:
    """
    Find the first layer where AUROC >= threshold, requiring it to hold
    for at least `min_consecutive` layers (avoids noisy early spikes).

    Returns layer index or None if never reached.

    I added min_consecutive=2 after noticing layer 0 sometimes hits 0.73
    for small models - turns out that's noise from BOS token being index 0
    and bleeding into positional encoding. Confirmed by looking at the
    actual probe weights; they're pointing at pos-embedding dimensions.
    """
    if not scores:
        return None

    sorted_layers = sorted(scores.keys())
    consecutive = 0
    first_candidate = None

    for layer in sorted_layers:
        auroc = scores.get(layer, 0.0)
        if auroc >= threshold:
            if first_candidate is None:
                first_candidate = layer
            consecutive += 1
            if consecutive >= min_consecutive:
                return first_candidate
        else:
            consecutive = 0
            first_candidate = None

    return None


def compute_separability_curve(
    scores: dict[str, dict[int, float]]
) -> dict[str, int | None]:
    """
    For each model, find the first separable layer.
    Returns {model_name: layer_idx_or_None}
    """
    result = {}
    for model, layer_scores in scores.items():
        first = find_first_separable_layer(layer_scores)
        result[model] = first
        status = f"layer {first}" if first is not None else "never"
        print(f"  {model}: first separable layer = {status}")
    return result


# ---------------------------------------------------------------------------
# Plot 1: Heatmap (model x layer, colored by AUROC)
# ---------------------------------------------------------------------------

def plot_heatmap(
    auroc_df: pd.DataFrame,
    sink_stats: dict,
    out_path: Path,
    threshold: float = SEPARABILITY_THRESHOLD,
):
    """
    Heatmap with models on y-axis, layers on x-axis, AUROC as color.

    A few design choices:
    - Only show layers 0..max_layer_for_that_model (rest are white/NaN)
    - Draw a dashed vertical line at the mean "first separable layer"
    - Annotate cells above threshold with a dot (subtle, not numbers - too cluttered)
    - Y-axis shows model names with param count
    """
    fig, ax = plt.subplots(figsize=(14, max(4, len(auroc_df) * 0.8)))

    # Mask NaN cells so they show as white
    data = auroc_df.values  # shape: (n_models, n_layers)
    masked_data = np.ma.masked_invalid(data)

    # Discrete colormap levels - makes it easier to read at a glance
    # 0.5 to 1.0 in steps of 0.05
    levels = np.arange(0.50, 1.01, 0.05)
    cmap = plt.get_cmap(HEATMAP_CMAP, len(levels) - 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax.imshow(
        masked_data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="none",
    )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Probe AUROC", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Axes labels
    # Y-axis: model name + param count
    y_labels = []
    for model in auroc_df.index:
        params = MODEL_META.get(model, {}).get("params_m", "?")
        y_labels.append(f"{model}\n({params}M)")
    ax.set_yticks(range(len(auroc_df)))
    ax.set_yticklabels(y_labels, fontsize=8)

    # X-axis: layer index, but only label every 2nd one if there are many
    n_layers = auroc_df.shape[1]
    step = 2 if n_layers > 20 else 1
    ax.set_xticks(range(0, n_layers, step))
    ax.set_xticklabels(range(0, n_layers, step), fontsize=8)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)

    # Mark cells above threshold with a small dot
    # NOTE: iterating over all cells is slow if n_layers is large,
    # but for <=36 layers and <=8 models it's fine
    for i, model in enumerate(auroc_df.index):
        n_model_layers = MODEL_META.get(model, {}).get("n_layers", n_layers)
        for j in range(min(n_model_layers, n_layers)):
            val = auroc_df.iloc[i, j]
            if not np.isnan(val) and val >= threshold:
                ax.plot(j, i, "w.", markersize=3, alpha=0.7)

    # Dashed vertical line at mean first-separable-layer
    # Only include models where we actually found a separable layer
    first_layers = []
    for model in auroc_df.index:
        layer_scores = {
            col: auroc_df.loc[model, col]
            for col in auroc_df.columns
            if not np.isnan(auroc_df.loc[model, col])
        }
        fl = find_first_separable_layer(layer_scores)
        if fl is not None:
            first_layers.append(fl)

    if first_layers:
        mean_fl = np.mean(first_layers)
        ax.axvline(
            mean_fl,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label=f"Mean first-separable layer ({mean_fl:.1f})",
        )
        ax.legend(fontsize=8, loc="upper right")

    # Title with some context
    ax.set_title(
        f"BOS residual stream probe AUROC by layer\n"
        f"(threshold for 'sink role learned' = {threshold:.2f}, "
        f"white dots mark cells above threshold)",
        fontsize=10,
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Line chart - first separable layer vs model size
# ---------------------------------------------------------------------------

def plot_separability_curve(
    scores: dict[str, dict[int, float]],
    sink_stats: dict,
    out_path: Path,
    threshold: float = SEPARABILITY_THRESHOLD,
):
    """
    Two-panel figure:
      Left: first separable layer (y) vs model params in M (x, log scale)
      Right: max AUROC achieved across all layers vs model params

    The hypothesis is that larger models develop the sink role earlier
    (lower layer index) because they have more capacity to specialize
    attention heads. I'm not sure this will hold - the 1B model has fewer
    layers than 410M so raw layer index isn't normalized. May need to
    switch to layer_fraction = first_layer / total_layers.

    TODO: add error bars once I run multiple seeds
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # Gather per-model stats
    model_data = []
    for model, layer_scores in scores.items():
        meta = MODEL_META.get(model, {})
        params_m = meta.get("params_m")
        n_layers = meta.get("n_layers")
        if params_m is None:
            print(f"  WARNING: no meta for {model}, skipping from curve plot")
            continue

        first_layer = find_first_separable_layer(layer_scores, threshold)
        max_auroc = max(layer_scores.values()) if layer_scores else np.nan
        mean_last_quarter = np.nan

        # Also compute mean AUROC in the last quarter of layers - sometimes
        # more informative than the max (which can be a spike)
        if n_layers and layer_scores:
            cutoff = int(n_layers * 0.75)
            late_scores = [v for k, v in layer_scores.items() if k >= cutoff]
            if late_scores:
                mean_last_quarter = np.mean(late_scores)

        # Normalized first layer (fraction through the network)
        first_layer_frac = None
        if first_layer is not None and n_layers:
            first_layer_frac = first_layer / n_layers

        model_data.append({
            "model": model,
            "params_m": params_m,
            "n_layers": n_layers,
            "first_layer": first_layer,
            "first_layer_frac": first_layer_frac,
            "max_auroc": max_auroc,
            "mean_last_quarter": mean_last_quarter,
        })

    # Sort by params
    model_data.sort(key=lambda x: x["params_m"])

    if not model_data:
        print("WARNING: no model data to plot for separability curve")
        plt.close(fig)
        return

    params = [d["params_m"] for d in model_data]
    labels = [d["model"].replace("pythia-", "") for d in model_data]

    # ---------- Left panel: first separable layer (fractional) ----------
    fracs = [d["first_layer_frac"] for d in model_data]
    # Split into "found" and "not found"
    found_params = [p for p, f in zip(params, fracs) if f is not None]
    found_fracs = [f for f in fracs if f is not None]
    not_found_params = [p for p, f in zip(params, fracs) if f is None]

    ax_left.semilogx(
        found_params,
        found_fracs,
        "o-",
        color=LINE_COLORS[0],
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        label=f"First layer ≥ {threshold:.2f} AUROC (fraction)",
    )

    # Models that never hit threshold: plot as triangles at top (1.0)
    if not_found_params:
        ax_left.scatter(
            not_found_params,
            [1.05] * len(not_found_params),
            marker="^",
            color=LINE_COLORS[1],
            s=60,
            label=f"Never reached {threshold:.2f}",
            zorder=5,
        )

    # Annotate each point with model label
    for d, frac in zip(model_data, fracs):
        if frac is not None:
            ax_left.annotate(
                d["model"].replace("pythia-", ""),
                xy=(d["params_m"], frac),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#555555",
            )

    ax_left.set_xlabel("Parameters (M)", fontsize=10)
    ax_left.set_ylabel(
        f"First separable layer\n(fraction of total layers)", fontsize=9
    )
    ax_left.set_ylim(-0.05, 1.15)
    ax_left.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_left.set_title(
        "Does bigger = earlier sink role?\n(lower fraction = earlier in network)",
        fontsize=9,
    )
    ax_left.legend(fontsize=8)
    ax_left.grid(True, which="both", alpha=0.3, linestyle=":")
    ax_left.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.5)

    # ---------- Right panel: max AUROC vs params ----------
    max_aurocs = [d["max_auroc"] for d in model_data]
    mean_late = [d["mean_last_quarter"] for d in model_data]

    ax_right.semilogx(
        params,
        max_aurocs,
        "s-",
        color=LINE_COLORS[0],
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Max AUROC (any layer)",
    )
    ax_right.semilogx(
        params,
        mean_late,
        "D--",
        color=LINE_COLORS[2],
        linewidth=1.4,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Mean AUROC (last 25% of layers)",
    )

    ax_right.axhline(
        threshold,
        color=LINE_COLORS[1],
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Threshold ({threshold:.2f})",
    )

    ax_right.set_xlabel("Parameters (M)", fontsize=10)
    ax_right.set_ylabel("Probe AUROC", fontsize=10)
    ax_right.set_ylim(0.45, 1.02)
    ax_right.set_title(
        "Does bigger = stronger sink-role signal?\n(linear probe on BOS residual stream)",
        fontsize=9,
    )
    ax_right.legend(fontsize=8)
    ax_right.grid(True, which="both", alpha=0.3, linestyle=":")

    # Add model size labels on right panel too
    for d, auroc in zip(model_data, max_aurocs):
        if not np.isnan(auroc):
            ax_right.annotate(
                d["model"].replace("pythia-", ""),
                xy=(d["params_m"], auroc),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=7,
                color="#555555",
            )

    # Shared footnote
    fig.text(
        0.5,
        -0.02,
        f"Linear probe (logistic regression, C=1.0) trained on BOS token residual stream.\n"
        f"Positive class = BOS tokens in attention-sink contexts (top-1% attention receivers).\n"
        f"Probe AUROC averaged over 3 random seeds. Pythia models, The Pile.",
        ha="center",
        fontsize=7,
        color="#666666",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved separability curve -> {out_path}")


# ---------------------------------------------------------------------------
# Optional: per-model AUROC-by-layer line chart
# Sometimes it's useful to see the full learning curve per model overlaid
# ---------------------------------------------------------------------------

def plot_per_model_curves(
    scores: dict[str, dict[int, float]],
    out_path: Path,
    threshold: float = SEPARABILITY_THRESHOLD,
):
    """
    All models on the same axes, x = layer_fraction (0-1), y = probe AUROC.

    Using layer fraction rather than raw index so different-depth models
    are comparable. This is probably the most honest view.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    def sort_key(m):
        return MODEL_META.get(m, {}).get("params_m", 99999)

    ordered_models = sorted(scores.keys(), key=sort_key)

    for idx, model in enumerate(ordered_models):
        layer_scores = scores[model]
        if not layer_scores:
            continue

        meta = MODEL_META.get(model, {})
        n_layers = meta.get("n_layers") or max(layer_scores.keys()) + 1
        params_m = meta.get("params_m", "?")

        sorted_layers = sorted(layer_scores.keys())
        x = [l / n_layers for l in sorted_layers]
        y = [layer_scores[l] for l in sorted_layers]

        color = LINE_COLORS[idx % len(LINE_COLORS)]
        label = f"{model.replace('pythia-', '')} ({params_m}M)"

        ax.plot(x, y, "o-", color=color, linewidth=1.6, markersize=4,
                alpha=0.85, label=label)

    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label=f"Threshold = {threshold:.2f}",
    )

    ax.set_xlabel("Normalized layer depth (fraction through network)", fontsize=10)
    ax.set_ylabel("Probe AUROC", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_title(
        "BOS sink-role probe AUROC vs layer depth\n"
        "(normalized by model depth for comparability)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-model curves -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores_dir",
        type=Path,
        default=Path("probe_scores"),
        help="Directory containing probe score JSONs",
    )
    p.add_argument(
        "--sinks_dir",
        type=Path,
        default=Path("sink_tokens"),
        help="Directory containing sink identification JSONs (optional)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for PNGs",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=SEPARABILITY_THRESHOLD,
        help=f"AUROC threshold for 'first separable layer' (default: {SEPARABILITY_THRESHOLD})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading probe scores...")
    scores = load_probe_scores(args.scores_dir)

    print("\nLoading sink stats...")
    sink_stats = load_sink_stats(args.sinks_dir)

    print(f"\nComputing first-separable-layer (threshold={args.threshold})...")
    separability = compute_separability_curve(scores)

    print("\nBuilding AUROC matrix for heatmap...")
    auroc_df, model_order = build_auroc_matrix(scores)
    print(f"Matrix shape: {auroc_df.shape}")
    print(auroc_df.to_string(float_format="{:.3f}".format))

    # Quick sanity check - any model with suspiciously high early-layer scores?
    # This caught the pos-embedding bleed issue I mentioned above.
    print("\nSanity check - layer 0 AUROC values:")
    for model in auroc_df.index:
        val = auroc_df.loc[model, 0] if 0 in auroc_df.columns else np.nan
        flag = " <-- HIGH, check for pos-embed leakage" if (not np.isnan(val) and val > 0.68) else ""
        print(f"  {model}: {val:.3f}{flag}")

    print("\nGenerating plots...")

    plot_heatmap(
        auroc_df,
        sink_stats,
        out_path=args.out / "heatmap_auroc_by_layer.png",
        threshold=args.threshold,
    )

    plot_separability_curve(
        scores,
        sink_stats,
        out_path=args.out / "separability_curve.png",
        threshold=args.threshold,
    )

    plot_per_model_curves(
        scores,
        out_path=args.out / "per_model_auroc_curves.png",
        threshold=args.threshold,
    )

    # Also dump a summary CSV - useful for the writeup / table
    summary_rows = []
    for model in model_order:
        meta = MODEL_META.get(model, {})
        layer_scores = scores.get(model, {})
        first = separability.get(model)
        n_layers = meta.get("n_layers")
        max_auroc = max(layer_scores.values()) if layer_scores else np.nan

        summary_rows.append({
            "model": model,
            "params_m": meta.get("params_m", ""),
            "n_layers": n_layers,
            "first_separable_layer": first if first is not None else "N/A",
            "first_separable_frac": (
                f"{first / n_layers:.3f}" if (first is not None and n_layers) else "N/A"
            ),
            "max_auroc": f"{max_auroc:.4f}" if not np.isnan(max_auroc) else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = args.out / "separability_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV -> {csv_path}")
    print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()