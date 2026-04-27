src/probe.py

import os
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# suppress the wall of sklearn convergence warnings when C is tiny
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS_DIR = Path("results/probe_scores")
ACTIVATIONS_DIR = Path("data/activations")

# --------------------------------------------------------------------------
# NOTE: the design choice here is to probe at the *residual stream* after
# each layer (i.e. post-LN, pre-attn input to layer n+1), which is what
# extract_residuals.py saves. An alternative would be probing mid-layer
# (e.g. after attn, before MLP), but that requires twice the disk space
# and I'm not sure it tells us more for the sink-detection question.
#
# The "sink role" hypothesis (from Xiao et al. 2023, "Efficient Streaming LLMs
# with Attention Sinks") is that BOS tokens accumulate attention mass not
# because they're semantically important, but because attention is effectively
# using them as a soft no-op. If the model *knows* a token plays this role,
# that should be linearly readable from the residual stream even at early
# layers. If not, we'd expect chance accuracy until late layers.
# --------------------------------------------------------------------------


def load_activations(model_name: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load saved activation tensors for a given model + layer.
    
    Expected file layout (from extract_residuals.py):
        data/activations/{model_name}/layer_{layer:02d}.npz
        
    Each .npz has:
        'X'      - shape (n_tokens, d_model), residual stream at layer exit
        'labels' - shape (n_tokens,), 1 = BOS token, 0 = non-BOS
        'positions' - shape (n_tokens,) absolute sequence positions (for analysis)
    
    Returns (X, y) ready for sklearn.
    """
    path = ACTIVATIONS_DIR / model_name / f"layer_{layer:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No activations found at {path}. Did extract_residuals.py run?"
        )
    
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(np.int32)
    
    # sanity check - class imbalance will be severe (1 BOS per sequence)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos < 20:
        print(f"  WARNING: only {n_pos} positive (BOS) examples at layer {layer}. "
              f"Probe results will be noisy.")
    
    return X, y


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    max_iter: int = 2000,
) -> dict:
    """
    Fit a logistic regression probe with cross-validation.
    
    Using LogisticRegressionCV because:
    - auto-selects regularization strength (important since d_model can be large)
    - gives us the per-fold scores without extra boilerplate
    - L2 penalty is fine here; we're not doing feature selection
    
    Returns a dict with accuracy, per-fold scores, best C, and the fitted probe
    (so we can later extract the probe direction for ablations).
    
    NOTE: class_weight='balanced' matters a lot here due to 1:many BOS ratio.
    Typical sequences are ~512 tokens so ~1:511 imbalance without it. Balanced
    weighting doesn't change *which* direction is separable, just whether the
    model bothers to separate at all during training.
    """
    # standardize - this is load-bearing; without it LR with L2 penalizes large
    # dimensions (and d_model can have highly varied scales per layer)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    probe = LogisticRegressionCV(
        Cs=10,          # searches 10 C values on a log scale
        cv=cv,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        scoring="balanced_accuracy",  # more meaningful than accuracy for imbalanced
    )
    
    probe.fit(X_scaled, y)
    
    # CV scores shape is (n_classes, n_Cs, n_folds) for multi-class,
    # for binary it's (1, n_Cs, n_folds) - need to index [0]
    best_c_idx = np.argmax(probe.scores_[1].mean(axis=1))
    fold_scores = probe.scores_[1][best_c_idx]  # (n_folds,)
    
    return {
        "mean_balanced_acc": float(fold_scores.mean()),
        "std_balanced_acc": float(fold_scores.std()),
        "fold_scores": fold_scores.tolist(),
        "best_C": float(probe.C_[0]),
        "probe": probe,
        "scaler": scaler,
    }


def run_probes_for_model(
    model_name: str,
    n_layers: int,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run probes across all layers for a given model. Returns a dict mapping
    layer index -> probe result dict (without the sklearn objects, just scores).
    """
    print(f"\n{'='*60}")
    print(f"Probing: {model_name}  ({n_layers} layers)")
    print(f"{'='*60}")
    
    results = {}
    probes_by_layer = {}  # keep the actual probe objects for the ablation later
    
    for layer in range(n_layers):
        try:
            X, y = load_activations(model_name, layer)
        except FileNotFoundError as e:
            print(f"  Layer {layer:2d}: SKIP ({e})")
            continue
        
        print(f"  Layer {layer:2d}: X={X.shape}, pos={y.sum()}, neg={(~y.astype(bool)).sum()}", end="")
        
        result = fit_probe(X, y, n_folds=n_folds, random_state=random_state)
        
        print(f"  -> bal_acc={result['mean_balanced_acc']:.3f} ± {result['std_balanced_acc']:.3f}  "
              f"(best C={result['best_C']:.2e})")
        
        probes_by_layer[layer] = {
            "probe": result.pop("probe"),
            "scaler": result.pop("scaler"),
        }
        results[layer] = result
    
    return results, probes_by_layer


def find_separation_layer(scores: dict, threshold: float = 0.75) -> Optional[int]:
    """
    Find the first layer where balanced accuracy exceeds threshold.
    
    "First linearly separable" is a bit subjective - using 0.75 as a threshold
    because:
    - chance for balanced accuracy is 0.5
    - 0.75 = halfway between chance and perfect
    - empirically (from Geva et al. 2022 probing work) this seems to correspond
      to when human-interpretable features become readable
    
    TODO: might want to smooth the curve first (single-layer spikes are noise)
    """
    layers = sorted(scores.keys())
    for l in layers:
        if scores[l]["mean_balanced_acc"] >= threshold:
            return l
    return None  # never crosses threshold


# --------------------------------------------------------------------------
# MLP-out ablation
#
# Motivated by: if the sink signal is *written* by MLP layers (as Geva et al.
# "Transformer Feed-Forward Layers Are Key-Value Memories" suggests), then
# zeroing out the MLP contribution at the separation layer should kill the
# probe. If the signal persists, it was written by attention.
#
# We do this by loading the raw MLP-out activations (saved separately by
# extract_residuals.py as layer_{l:02d}_mlp_out.npz) and re-running the
# probe on X_residual - X_mlp_out for just the identified layer.
# --------------------------------------------------------------------------

def mlp_ablation_at_layer(
    model_name: str,
    layer: int,
    probe: LogisticRegressionCV,
    scaler: StandardScaler,
) -> dict:
    """
    Rerun the probe on the residual *minus* the MLP contribution at a given layer.
    
    This tells us: is the linearly-readable sink signal written by MLP or attention?
    """
    residual_path = ACTIVATIONS_DIR / model_name / f"layer_{layer:02d}.npz"
    mlp_out_path = ACTIVATIONS_DIR / model_name / f"layer_{layer:02d}_mlp_out.npz"
    
    if not mlp_out_path.exists():
        print(f"  Ablation SKIP: no MLP-out activations at {mlp_out_path}")
        return {}
    
    residual_data = np.load(residual_path)
    mlp_data = np.load(mlp_out_path)
    
    X_residual = residual_data["X"].astype(np.float32)
    X_mlp_out = mlp_data["X"].astype(np.float32)
    y = residual_data["labels"].astype(np.int32)
    
    # ablated representation: subtract the MLP contribution
    # NOTE: this is a linear approximation. The true counterfactual would
    # require re-running the model with the MLP zeroed, which would change
    # downstream layer inputs. But for a probing-only analysis this is fine.
    X_ablated = X_residual - X_mlp_out
    
    X_ablated_scaled = scaler.transform(X_ablated)  # use the same scaler
    
    # just use the fitted probe - no retraining needed, we want to see if
    # the *same direction* still carries signal after ablation
    preds = probe.predict(X_ablated_scaled)
    from sklearn.metrics import balanced_accuracy_score
    bal_acc_ablated = balanced_accuracy_score(y, preds)
    
    # also fit a *new* probe on ablated X to ask: is there *any* direction
    # in ablated space that encodes sinkness?
    new_result = fit_probe(X_ablated, y)
    
    return {
        "original_probe_on_ablated": float(bal_acc_ablated),
        "new_probe_on_ablated_mean": new_result["mean_balanced_acc"],
        "new_probe_on_ablated_std": new_result["std_balanced_acc"],
    }


def plot_probe_curves(
    all_results: dict,
    output_path: Path,
    threshold: float = 0.75,
):
    """
    Plot balanced accuracy vs layer for each model. One line per model,
    with error bands from CV folds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(all_results.keys())
    colors = cm.viridis(np.linspace(0.1, 0.9, len(models)))
    
    for color, model_name in zip(colors, models):
        scores = all_results[model_name]["layer_scores"]
        layers = sorted(int(k) for k in scores.keys())
        
        means = [scores[l]["mean_balanced_acc"] for l in layers]
        stds = [scores[l]["std_balanced_acc"] for l in layers]
        
        means = np.array(means)
        stds = np.array(stds)
        
        # normalize layer index to 0-1 for comparability across model sizes
        # (Pythia-70M has 6 layers, Pythia-6.9B has 32 - we want to compare
        #  where *proportionally* the signal emerges)
        norm_layers = np.array(layers) / max(layers) if max(layers) > 0 else np.array(layers)
        
        ax.plot(norm_layers, means, "-o", color=color, label=model_name, markersize=4)
        ax.fill_between(norm_layers, means - stds, means + stds, alpha=0.15, color=color)
        
        sep_layer = find_separation_layer(scores, threshold)
        if sep_layer is not None:
            sep_norm = sep_layer / max(layers)
            ax.axvline(sep_norm, color=color, linestyle="--", alpha=0.4)
    
    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.5, label=f"threshold ({threshold})")
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=1, label="chance")
    
    ax.set_xlabel("Normalized layer depth (0 = embedding, 1 = final layer)")
    ax.set_ylabel("Balanced accuracy (5-fold CV)")
    ax.set_title("Logistic probe: BOS 'sink role' signal by layer depth\n"
                 "(higher = sink role more linearly readable from residual stream)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved probe curve plot -> {output_path}")


def save_results(all_results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save serializable summary (no sklearn objects)
    summary = {}
    for model_name, model_data in all_results.items():
        summary[model_name] = {
            "layer_scores": {
                str(l): {k: v for k, v in scores.items()}
                for l, scores in model_data["layer_scores"].items()
            },
            "separation_layer": model_data.get("separation_layer"),
            "ablation_results": model_data.get("ablation_results", {}),
        }
    
    out_path = output_dir / "probe_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary -> {out_path}")
    
    # also save a quick-read CSV: model, layer, mean_bal_acc, std
    import csv
    csv_path = output_dir / "probe_scores_flat.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "layer", "mean_balanced_acc", "std_balanced_acc",
                         "best_C", "fold_scores"])
        for model_name, model_data in all_results.items():
            for layer, scores in model_data["layer_scores"].items():
                writer.writerow([
                    model_name,
                    layer,
                    f"{scores['mean_balanced_acc']:.4f}",
                    f"{scores['std_balanced_acc']:.4f}",
                    f"{scores['best_C']:.2e}",
                    ";".join(f"{s:.4f}" for s in scores["fold_scores"]),
                ])
    print(f"Saved flat CSV -> {csv_path}")


# --------------------------------------------------------------------------
# Model configs
# We ran extract_residuals.py on these checkpoints. Layer counts are the
# actual transformer block counts (not counting the embedding layer).
# --------------------------------------------------------------------------

MODEL_CONFIGS = {
    "pythia-70m":   {"n_layers": 6,  "d_model": 512},
    "pythia-160m":  {"n_layers": 12, "d_model": 768},
    "pythia-410m":  {"n_layers": 24, "d_model": 1024},
    "pythia-1.4b":  {"n_layers": 24, "d_model": 2048},
    "pythia-2.8b":  {"n_layers": 32, "d_model": 2560},
    # NOTE: pythia-6.9b activations are ~40GB, skipping unless we have more disk
    # "pythia-6.9b":  {"n_layers": 32, "d_model": 4096},
}


def main():
    parser = argparse.ArgumentParser(
        description="Train logistic probes to detect BOS 'sink role' signal in residual streams"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="Which model checkpoints to probe (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Balanced accuracy threshold for 'linearly separable' (default: 0.75)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds for LogisticRegressionCV (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="Skip the MLP-out ablation (faster, useful for quick iteration)",
    )
    parser.add_argument(
        "--activations_dir",
        type=Path,
        default=ACTIVATIONS_DIR,
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
    )
    args = parser.parse_args()
    
    global ACTIVATIONS_DIR, RESULTS_DIR
    ACTIVATIONS_DIR = args.activations_dir
    RESULTS_DIR = args.results_dir
    
    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Probing sink role signal in BOS residual streams")
    print(f"Models: {args.models}")
    print(f"Separation threshold: {args.threshold}")
    print(f"CV folds: {args.n_folds}")
    print(f"Seed: {args.seed}")
    
    all_results = {}
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: unknown model '{model_name}', skipping")
            continue
        
        cfg = MODEL_CONFIGS[model_name]
        
        layer_scores, probes_by_layer = run_probes_for_model(
            model_name=model_name,
            n_layers=cfg["n_layers"],
            n_folds=args.n_folds,
            random_state=args.seed,
        )
        
        sep_layer = find_separation_layer(layer_scores, threshold=args.threshold)
        print(f"\n  -> First separation layer (>={args.threshold:.2f}): {sep_layer}")
        
        ablation_results = {}
        if not args.skip_ablation and sep_layer is not None:
            print(f"\n  Running MLP-out ablation at layer {sep_layer}...")
            ablation_results = mlp_ablation_at_layer(
                model_name=model_name,
                layer=sep_layer,
                probe=probes_by_layer[sep_layer]["probe"],
                scaler=probes_by_layer[sep_layer]["scaler"],
            )
            if ablation_results:
                print(f"  Ablation results:")
                print(f"    Original probe on ablated X: "
                      f"{ablation_results['original_probe_on_ablated']:.3f}")
                print(f"    New probe on ablated X: "
                      f"{ablation_results['new_probe_on_ablated_mean']:.3f} "
                      f"± {ablation_results['new_probe_on_ablated_std']:.3f}")
                
                # interpret the result
                orig_acc = layer_scores[sep_layer]["mean_balanced_acc"]
                abl_new = ablation_results["new_probe_on_ablated_mean"]
                drop = orig_acc - abl_new
                if drop > 0.15:
                    print(f"    Interpretation: MLP contributes significantly "
                          f"(drop={drop:.3f}) - sink signal partially MLP-written")
                elif drop < 0.05:
                    print(f"    Interpretation: MLP contribution small "
                          f"(drop={drop:.3f}) - signal likely attention-written")
                else:
                    print(f"    Interpretation: ambiguous (drop={drop:.3f})")
        
        # save the probe objects in case we want to do more analysis later
        # (e.g., extract probe directions for activation patching)
        probe_cache_dir = args.results_dir / "probe_objects" / model_name
        probe_cache_dir.mkdir(parents=True, exist_ok=True)
        for layer, probe_data in probes_by_layer.items():
            if layer == sep_layer or True:  # save all for now, might prune later
                with open(probe_cache_dir / f"layer_{layer:02d}.pkl", "wb") as f:
                    pickle.dump(probe_data, f)
        
        all_results[model_name] = {
            "layer_scores": layer_scores,
            "separation_layer": sep_layer,
            "ablation_results": ablation_results,
        }
    
    # save numeric results
    save_results(all_results, args.results_dir)
    
    # plot
    plot_probe_curves(
        all_results,
        output_path=args.results_dir / "probe_curves.png",
        threshold=args.threshold,
    )
    
    # print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: First layer where sink signal is linearly separable")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Sep Layer':>10} {'Sep Layer (norm)':>18} {'Acc at sep':>12}")
    print(f"{'-'*62}")
    for model_name, model_data in all_results.items():
        cfg = MODEL_CONFIGS.get(model_name, {})
        n_layers = cfg.get("n_layers", "?")
        sep = model_data["separation_layer"]
        if sep is not None:
            norm = f"{sep / n_layers:.2f}" if isinstance(n_layers, int) else "?"
            acc = model_data["layer_scores"][sep]["mean_balanced_acc"]
            print(f"  {model_name:<18} {sep:>10} {norm:>18} {acc:>12.3f}")
        else:
            print(f"  {model_name:<18} {'none':>10} {'—':>18} {'<threshold':>12}")


if __name__ == "__main__":
    main()