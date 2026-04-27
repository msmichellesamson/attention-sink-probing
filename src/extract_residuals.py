"""
extract_residuals.py

Pull residual stream activations at the BOS token for every layer of a Pythia model,
across a sample of text. The goal is to see whether BOS encodes a stable "sink role"
signal that's linearly separable from non-BOS token representations.

Motivation: Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks"
showed BOS consistently attracts disproportionate attention mass. But *why* does the
model route attention there? One hypothesis: the residual stream at BOS develops a
distinguishable internal signature by some layer -- a "this is the sink" feature --
that attention heads learn to query against.

This script:
  1. Loads a Pythia checkpoint via TransformerLens
  2. Hooks every residual stream + optionally MLP-out at BOS position
  3. Runs a batch of text through the model
  4. Saves per-layer BOS activations and matched non-BOS activations to disk
     (we'll need the non-BOS activations as a "negative class" for the probes)

Usage:
  python src/extract_residuals.py --model pythia-1.4b --n_samples 500
  python src/extract_residuals.py --model pythia-160m --n_samples 500 --also_mlp_out

Hardware note: ran development on a single A10G (24GB). 1.4B model uses ~8GB at fp32,
fine to run fp16 if you want to go bigger. The 6.9B model needs --dtype float16 or
it'll OOM on extraction.

TODO: add --layer_subset flag so we can skip every-other-layer on big models
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/activations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# These are the Pythia model names as TransformerLens knows them.
# Double-checked: TL uses "pythia-XYZ" (no "EleutherAI/" prefix needed for from_pretrained).
SUPPORTED_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
]

# How many non-BOS token positions to sample per sequence as negatives.
# We want a reasonably balanced probe dataset. BOS is 1 position per seq,
# so taking 5 non-BOS positions per seq gives us ~5:1 neg:pos ratio --
# intentionally imbalanced to reflect reality, but maybe tune this.
N_NEG_PER_SEQ = 5

# Tokenizer ID for BOS in Pythia (neox tokenizer). Confirm at runtime.
PYTHIA_BOS_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_text_samples(n_samples: int, seq_len: int = 128, seed: int = 42) -> list[str]:
    """
    Pull text from The Pile (via HF datasets). We want naturally-occurring text
    where BOS is genuinely the first token -- not constructed examples.

    Using 'monology/pile-uncopyrighted' which is a subset of the Pile that
    Pythia was actually trained on (or close enough). This matters because
    we're trying to understand *this model's* representations, not probing
    in-distribution for some other model.

    seq_len: target token count -- we just filter by rough character count,
             actual tokenization happens later.
    """
    print(f"Loading {n_samples} text samples from Pile subset...")
    
    # streaming=True so we don't download the whole thing
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    rng = random.Random(seed)
    samples = []
    
    # rough char budget: ~4 chars/token is a decent heuristic for English
    char_budget = seq_len * 4
    
    # NOTE: streaming dataset iteration is slow -- we skip randomly to
    # avoid always getting the same documents. This is a bit hacky.
    skip_every = 7  # take roughly every 7th doc to get diversity
    
    for i, example in enumerate(dataset):
        if i % skip_every != 0:
            continue
        text = example.get("text", "")
        if len(text) < char_budget // 2:
            continue
        # trim to roughly seq_len tokens worth
        samples.append(text[:char_budget])
        if len(samples) >= n_samples:
            break
    
    print(f"  Loaded {len(samples)} samples (requested {n_samples})")
    if len(samples) < n_samples:
        print(f"  WARNING: only got {len(samples)} samples, continuing anyway")
    
    return samples


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_hooks(
    model: HookedTransformer,
    n_layers: int,
    also_mlp_out: bool,
) -> tuple[dict, list]:
    """
    Register forward hooks on TransformerLens hook points to capture BOS
    residual stream at each layer.

    TransformerLens hook naming:
      - 'hook_embed': embedding output (before any layers)
      - 'blocks.{i}.hook_resid_post': residual stream after block i
      - 'blocks.{i}.hook_mlp_out': MLP output at block i (before adding to residual)

    We're capturing 'hook_resid_post' because that's the running sum --
    it's what gets used as K/V in the *next* layer's attention, so if
    BOS is going to "absorb" attention, its residual_post is the right
    thing to characterize.

    Returns:
      activations: dict mapping hook_name -> list of tensors (one per forward pass)
      hook_handles: list of hook handles (for cleanup)
    """
    activations = {}
    hook_handles = []
    
    # Always grab the embedding output as "layer -1" / "layer 0 pre"
    hook_names = ["hook_embed"]
    
    for i in range(n_layers):
        hook_names.append(f"blocks.{i}.hook_resid_post")
        if also_mlp_out:
            hook_names.append(f"blocks.{i}.hook_mlp_out")
    
    for name in hook_names:
        activations[name] = []
    
    def make_hook(hook_name):
        def hook_fn(value, hook):
            # value shape: (batch, seq_len, d_model)
            # We'll store the whole tensor here and extract BOS later,
            # so we can also grab arbitrary non-BOS positions.
            # Clone to detach from computation graph and avoid memory issues.
            activations[hook_name].append(value.detach().cpu())
        return hook_fn
    
    for name in hook_names:
        # TL's add_hook returns a handle
        handle = model.add_hook(name, make_hook(name))
        hook_handles.append(handle)
    
    return activations, hook_handles


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    tokenizer,
    texts: list[str],
    seq_len: int,
    batch_size: int,
    also_mlp_out: bool,
    device: str,
) -> dict:
    """
    Run all texts through the model, collect per-layer BOS activations
    and a matched set of non-BOS activations.

    Returns a dict:
      {
        hook_name: {
          "bos": np.ndarray of shape (n_seqs, d_model),
          "non_bos": np.ndarray of shape (n_seqs * N_NEG_PER_SEQ, d_model),
          "non_bos_positions": np.ndarray of shape (n_seqs * N_NEG_PER_SEQ,)
        }
      }
    """
    n_layers = model.cfg.n_layers
    
    # Tokenize all texts up front
    print("Tokenizing...")
    all_tokens = []
    for text in tqdm(texts, desc="tokenize"):
        # add special tokens=False because TL models add BOS themselves
        # actually, for Pythia/neox this is a bit ambiguous -- let's be explicit
        enc = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]  # (seq,)
        
        # Verify BOS is first token. If not, something is off.
        # Pythia uses neox tokenizer where BOS is prepended automatically.
        # TransformerLens should handle this, but worth checking.
        if ids[0].item() != PYTHIA_BOS_TOKEN_ID:
            # prepend BOS manually -- this happens occasionally with some
            # HF tokenizer configs
            bos = torch.tensor([PYTHIA_BOS_TOKEN_ID])
            ids = torch.cat([bos, ids[: seq_len - 1]])
        
        all_tokens.append(ids)
    
    # Verify BOS rate
    bos_first = sum(1 for t in all_tokens if t[0].item() == PYTHIA_BOS_TOKEN_ID)
    print(f"  BOS is first token in {bos_first}/{len(all_tokens)} sequences ({bos_first/len(all_tokens):.1%})")
    
    # Pad to uniform length for batching
    # Using right-padding; BOS is always at position 0 so this is fine.
    padded = torch.nn.utils.rnn.pad_sequence(
        all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id or 1
    )
    
    print(f"Padded token tensor shape: {padded.shape}")
    print(f"Running forward passes (batch_size={batch_size})...")
    
    # Register hooks
    activations, hook_handles = register_hooks(model, n_layers, also_mlp_out)
    
    model.eval()
    with torch.no_grad():
        n_seqs = padded.shape[0]
        for batch_start in tqdm(range(0, n_seqs, batch_size), desc="forward passes"):
            batch = padded[batch_start : batch_start + batch_size].to(device)
            # TL's forward pass -- we don't need the logits here
            _ = model(batch, return_type=None)
    
    # Remove hooks to clean up
    for handle in hook_handles:
        handle.remove()
    
    print("Extracting BOS and non-BOS positions from cached activations...")
    
    # For each hook, concatenate all batches into one big tensor,
    # then split into BOS vs non-BOS
    result = {}
    
    # Precompute which positions are non-BOS for each sequence
    # (we'll use the same positions across all hooks for consistency)
    seq_lens = [t.shape[0] for t in all_tokens]
    
    rng = np.random.default_rng(42)
    non_bos_positions = []
    for slen in seq_lens:
        # position 0 is BOS, positions 1..slen-1 are candidates
        available = np.arange(1, slen)
        if len(available) <= N_NEG_PER_SEQ:
            chosen = available
        else:
            chosen = rng.choice(available, size=N_NEG_PER_SEQ, replace=False)
        non_bos_positions.append(chosen)
    
    for hook_name, batch_list in tqdm(activations.items(), desc="processing hooks"):
        # Concatenate all batches: list of (batch, seq, d_model) -> (n_seqs, seq, d_model)
        all_acts = torch.cat(batch_list, dim=0)  # (n_seqs, seq, d_model)
        
        assert all_acts.shape[0] == len(all_tokens), (
            f"Activation count mismatch: got {all_acts.shape[0]}, "
            f"expected {len(all_tokens)}"
        )
        
        # BOS activations: position 0 for every sequence
        bos_acts = all_acts[:, 0, :].numpy()  # (n_seqs, d_model)
        
        # Non-BOS activations: sample N_NEG_PER_SEQ positions per sequence
        non_bos_acts = []
        non_bos_pos_flat = []
        for seq_idx, positions in enumerate(non_bos_positions):
            for pos in positions:
                non_bos_acts.append(all_acts[seq_idx, pos, :].numpy())
                non_bos_pos_flat.append(pos)
        
        non_bos_acts = np.stack(non_bos_acts)  # (n_seqs * N_NEG_PER_SEQ, d_model)
        
        result[hook_name] = {
            "bos": bos_acts,
            "non_bos": non_bos_acts,
            "non_bos_positions": np.array(non_bos_pos_flat),
        }
    
    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_activations(
    result: dict,
    model_name: str,
    n_samples: int,
    seq_len: int,
    also_mlp_out: bool,
) -> Path:
    """
    Save to results/activations/{model_name}/
    
    Layout:
      {model_name}/
        metadata.json
        {hook_name}_bos.npy
        {hook_name}_non_bos.npy
        {hook_name}_non_bos_positions.npy
    
    Using .npy instead of .pt because the probing script uses scikit-learn
    which expects numpy anyway -- avoids the torch -> numpy conversion at
    load time and removes the torch version dependency on saved files.
    """
    # sanitize model name for filesystem (e.g. "pythia-1.4b" -> "pythia-1.4b")
    save_dir = RESULTS_DIR / model_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model_name": model_name,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "also_mlp_out": also_mlp_out,
        "n_neg_per_seq": N_NEG_PER_SEQ,
        "hook_names": list(result.keys()),
        "bos_token_id": PYTHIA_BOS_TOKEN_ID,
        "shapes": {},
    }
    
    for hook_name, data in result.items():
        # Replace dots with dashes for filenames
        safe_name = hook_name.replace(".", "-")
        
        bos_path = save_dir / f"{safe_name}_bos.npy"
        non_bos_path = save_dir / f"{safe_name}_non_bos.npy"
        pos_path = save_dir / f"{safe_name}_non_bos_positions.npy"
        
        np.save(bos_path, data["bos"])
        np.save(non_bos_path, data["non_bos"])
        np.save(pos_path, data["non_bos_positions"])
        
        metadata["shapes"][hook_name] = {
            "bos": list(data["bos"].shape),
            "non_bos": list(data["non_bos"].shape),
        }
    
    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved activations to: {save_dir}")
    print(f"  Hooks saved: {len(result)}")
    for hook_name, data in result.items():
        print(f"    {hook_name}: BOS={data['bos'].shape}, non-BOS={data['non_bos'].shape}")
    
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract BOS residual stream activations from Pythia models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-160m",
        choices=SUPPORTED_MODELS,
        help="Pythia model variant to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of text sequences to process",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Token sequence length (shorter = faster, longer = more non-BOS diversity)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Forward pass batch size. Reduce if OOM.",
    )
    parser.add_argument(
        "--also_mlp_out",
        action="store_true",
        help="Also save MLP output activations at each layer (larger files)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Model dtype. Use float16 for 2.8B+ to avoid OOM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override. Defaults to cuda if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    
    if device == "cpu" and args.model in ["pythia-2.8b", "pythia-6.9b"]:
        print("WARNING: running a 2.8B+ model on CPU is going to be very slow.")
        print("Consider using --device cuda or a smaller model.")
    
    # Load model
    print(f"\nLoading {args.model} via TransformerLens...")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    model = HookedTransformer.from_pretrained(
        args.model,
        center_writing_weights=True,   # standard TL practice
        center_unembed=True,
        fold_ln=True,                  # folds layernorm params into weights
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    
    # Grab the tokenizer from the model
    tokenizer = model.tokenizer
    
    # Sanity check BOS token ID
    actual_bos = tokenizer.bos_token_id
    if actual_bos is not None and actual_bos != PYTHIA_BOS_TOKEN_ID:
        print(f"WARNING: Expected BOS token ID {PYTHIA_BOS_TOKEN_ID}, "
              f"got {actual_bos}. Updating.")
        global PYTHIA_BOS_TOKEN_ID
        PYTHIA_BOS_TOKEN_ID = actual_bos
    
    print(f"  n_layers: {model.cfg.n_layers}")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  n_heads: {model.cfg.n_heads}")
    
    # Load data
    texts = load_text_samples(args.n_samples, args.seq_len, seed=args.seed)
    
    # Run extraction
    result = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        also_mlp_out=args.also_mlp_out,
        device=device,
    )
    
    # Save
    save_dir = save_activations(
        result=result,
        model_name=args.model,
        n_samples=len(texts),
        seq_len=args.seq_len,
        also_mlp_out=args.also_mlp_out,
    )
    
    # Quick sanity check on what we got
    print("\n--- Sanity check ---")
    first_hook = list(result.keys())[0]
    bos_sample = result[first_hook]["bos"][:5]  # 5 BOS activations
    non_bos_sample = result[first_hook]["non_bos"][:5]
    
    print(f"BOS activation norm (first 5 seqs, {first_hook}): "
          f"{np.linalg.norm(bos_sample, axis=-1).round(3)}")
    print(f"Non-BOS activation norm (first 5, {first_hook}): "
          f"{np.linalg.norm(non_bos_sample, axis=-1).round(3)}")
    
    # If BOS has a consistent and distinctive norm, that's already interesting
    bos_norms = np.linalg.norm(result[first_hook]["bos"], axis=-1)
    non_bos_norms = np.linalg.norm(result[first_hook]["non_bos"], axis=-1)
    print(f"\nNorm stats at {first_hook}:")
    print(f"  BOS:     mean={bos_norms.mean():.3f}, std={bos_norms.std():.3f}")
    print(f"  non-BOS: mean={non_bos_norms.mean():.3f}, std={non_bos_norms.std():.3f}")
    
    # Check if BOS norm is distinctive at last layer too
    last_hook = [k for k in result.keys() if "resid_post" in k][-1]
    bos_norms_last = np.linalg.norm(result[last_hook]["bos"], axis=-1)
    non_bos_norms_last = np.linalg.norm(result[last_hook]["non_bos"], axis=-1)
    print(f"\nNorm stats at {last_hook}:")
    print(f"  BOS:     mean={bos_norms_last.mean():.3f}, std={bos_norms_last.std():.3f}")
    print(f"  non-BOS: mean={non_bos_norms_last.mean():.3f}, std={non_bos_norms_last.std():.3f}")
    
    print(f"\nDone. Run src/identify_sinks.py to probe for linear separability.")
    print(f"Activations at: {save_dir}")


if __name__ == "__main__":
    main()