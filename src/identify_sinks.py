src/identify_sinks.py
"""
identify_sinks.py

Sweep attention entropy at BOS position across Pythia model family,
identify (layer, head) pairs that consistently have low entropy (= sink heads),
and emit a JSON of candidates.

The "attention sink" phenomenon: Xiao et al. 2023 (StreamingLLM) noticed that
attention mass piles up on initial tokens even when they're semantically
irrelevant. The BOS token (or first token) acts as a "sink" that absorbs
probability that doesn't go anywhere useful. My question here: is this a
*role* that gets encoded into the residual stream at BOS, or is it purely
a softmax artifact? This script is step 1: just find the sink heads reliably
across models so I have a consistent set to probe in the next script.

Entropy threshold: low-entropy heads attend almost exclusively to BOS,
which is the sink signature. Threshold is a CLI arg so I can sweep
and see if the candidate set is stable.

Hardware note: ran this on an A100 40GB, Pythia-6.9B needs ~14GB in bfloat16.
For smaller machines, just use the 70M/160M/410M models.

Usage:
    python src/identify_sinks.py \
        --models pythia-70m pythia-160m pythia-410m pythia-1.4b \
        --n_samples 200 \
        --threshold 0.5 \
        --output results/sink_candidates.json

TODO: also check token position 1 - some papers report sinks at pos 1 not 0
NOTE: transformer_lens auto-moves things to GPU if available, but be careful
      with the larger models - OOM is silent and wrong on some setups
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm


# reproducibility - set before anything else
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Pythia models in the TransformerLens naming convention
# The full family goes up to 12B but I'm focusing on the range where
# I can actually fit multiple runs in a day
PYTHIA_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

# TL name format for pythia
# EleutherAI/pythia-70m-deduped is the cleaner version but standard
# pythia is fine for this - deduped matters more for memorization studies
TL_MODEL_NAMES = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
}


def entropy_bits(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Shannon entropy in bits. Expects a probability vector (sums to ~1).
    Using log2 so the scale is interpretable: max entropy for seq_len=512
    is log2(512) = 9 bits. A sink head attending almost entirely to one
    token should be < 0.5 bits.

    eps: avoid log(0). In practice attn weights are already softmax'd so
    they shouldn't be exactly 0, but numerics happen.
    """
    p = probs.clamp(min=eps)
    return -(p * torch.log2(p)).sum(dim=-1)


def load_text_sample(n_samples: int, min_length: int = 64) -> list[str]:
    """
    Pull a small corpus sample from The Pile (via HF datasets).
    Using the 'all' config, streaming to avoid downloading 800GB.

    min_length: skip very short sequences - we want BOS to be in a real
    context, not a 3-token sequence where sink behavior is trivial.

    NOTE: streaming means results depend on HF dataset ordering,
    which should be deterministic but I've seen edge cases. If results
    look weird, check that the dataset isn't shuffled upstream.
    """
    print(f"Loading {n_samples} samples from The Pile (streaming)...")

    # Using pile-uncopyrighted as it's more reliably accessible
    # and the copyright issues with the full pile don't matter for this use
    try:
        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Pile unavailable ({e}), falling back to wikitext-103")
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=True,
        )

    texts = []
    for sample in dataset:
        text = sample.get("text", "")
        # filter: needs to be long enough to have real context
        if len(text.split()) >= min_length:
            texts.append(text)
        if len(texts) >= n_samples:
            break

    print(f"  Got {len(texts)} usable samples (min_length={min_length} words)")
    return texts


def tokenize_and_truncate(
    model: HookedTransformer,
    texts: list[str],
    max_seq_len: int = 256,
) -> list[torch.Tensor]:
    """
    Tokenize texts and truncate to max_seq_len.

    Why 256? Long enough to give the model real context, short enough
    that the attention matrix isn't dominated by padding effects.
    The sink phenomenon shows up in the first few tokens anyway,
    so very long sequences don't add much signal here.

    Returns a list of 1D tensors (token ids).
    """
    tokenized = []
    for text in texts:
        # HookedTransformer.to_tokens adds BOS automatically based on
        # the tokenizer config - need to check this is actually happening
        tokens = model.to_tokens(text, prepend_bos=True)  # shape: [1, seq_len]
        tokens = tokens[0]  # unbatch: [seq_len]

        if tokens.shape[0] > max_seq_len:
            tokens = tokens[:max_seq_len]

        # sanity check: first token should be BOS
        bos_id = model.tokenizer.bos_token_id
        if tokens[0].item() != bos_id:
            # This happened to me with some tokenizers - log and skip
            print(f"  WARNING: first token {tokens[0].item()} != BOS {bos_id}, skipping")
            continue

        tokenized.append(tokens)

    print(f"  Tokenized {len(tokenized)} sequences")
    return tokenized


def compute_attention_entropy_at_bos(
    model: HookedTransformer,
    token_sequences: list[torch.Tensor],
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """
    For each (layer, head), compute mean attention entropy at BOS position
    across all sequences.

    The BOS token is at position 0. We want the entropy of the row in
    the attention matrix corresponding to BOS as the QUERY. That is:
    how spread out is BOS's attention over the rest of the sequence?

    Wait - actually there are two sinks to think about:
    1. BOS as KEY: other tokens attending to BOS (the classic sink)
    2. BOS as QUERY: where does BOS itself attend?

    For sink identification (step 1), I want BOS-as-KEY: I should be
    looking at the column for BOS in the attention pattern matrix,
    i.e., how much does each other token attend to BOS.

    But for the "does BOS encode a sink role" question (step 2 in the
    probing script), I need BOS-as-QUERY: what does BOS attend to?

    This function does BOS-as-QUERY (entropy of row 0), which tells me
    about heads where BOS has degenerate attention patterns. I'll look at
    BOS-as-KEY separately in another function.

    TODO: refactor this into two functions so it's unambiguous.

    Returns: np.ndarray of shape [n_layers, n_heads] with mean entropy values
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # accumulate entropy sums for averaging later
    entropy_sums = np.zeros((n_layers, n_heads))
    count = 0

    model.eval()

    # Process in batches to avoid OOM
    # NOTE: sequences have different lengths, so we can't easily batch
    # without padding. For now just do one at a time. Slow but safe.
    # Could speed up with right-padding + masking, but not worth it
    # for n_samples=200.
    with torch.no_grad():
        for i, tokens in enumerate(tqdm(token_sequences, desc="  Computing entropy")):
            tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]
            seq_len = tokens.shape[1]

            if seq_len < 2:
                continue  # can't have meaningful attention with 1 token

            # Run with attention pattern caching
            # HookedTransformer caches "pattern" = softmax attention weights
            # shape per head: [batch, n_heads, seq_len, seq_len]
            try:
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("pattern"),
                    return_type=None,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  OOM at sample {i}, skipping. Consider reducing max_seq_len.")
                    torch.cuda.empty_cache()
                    continue
                raise

            for layer in range(n_layers):
                pattern_key = f"blocks.{layer}.attn.hook_pattern"
                if pattern_key not in cache:
                    # shouldn't happen but let's be defensive
                    continue

                attn_pattern = cache[pattern_key]  # [1, n_heads, seq_len, seq_len]
                attn_pattern = attn_pattern[0]      # [n_heads, seq_len, seq_len]

                # Row 0 = BOS as query, attending over all positions
                # shape: [n_heads, seq_len]
                bos_query_attn = attn_pattern[:, 0, :]  # [n_heads, seq_len]

                # Entropy of BOS's attention distribution per head
                h = entropy_bits(bos_query_attn)  # [n_heads]

                entropy_sums[layer] += h.cpu().numpy()

            count += 1

            # free the cache explicitly - TL caches can be large
            del cache
            if device != "cpu":
                torch.cuda.empty_cache()

    if count == 0:
        raise ValueError("No samples processed successfully. Check model and data.")

    mean_entropy = entropy_sums / count
    print(f"  Processed {count} sequences")
    print(f"  Entropy range: [{mean_entropy.min():.3f}, {mean_entropy.max():.3f}] bits")

    return mean_entropy


def compute_bos_column_entropy(
    model: HookedTransformer,
    token_sequences: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    """
    BOS-as-KEY: for each other token position, how much attention goes to BOS?
    
    Specifically: mean over all (non-BOS) query positions of the attention weight
    on BOS (position 0). High values = lots of attention mass flowing to BOS
    from other tokens = classic sink behavior.

    This is a different angle than entropy. A head could have HIGH attention-to-BOS
    from other tokens (sink) while BOS itself attends somewhere specific (not random).

    Returns: [n_layers, n_heads] mean attention weight on BOS (0 to 1 scale)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_to_bos_sums = np.zeros((n_layers, n_heads))
    count = 0

    model.eval()

    with torch.no_grad():
        for tokens in tqdm(token_sequences, desc="  BOS-as-KEY attention"):
            tokens = tokens.unsqueeze(0).to(device)
            seq_len = tokens.shape[1]

            if seq_len < 3:
                continue

            try:
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("pattern"),
                    return_type=None,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

            for layer in range(n_layers):
                pattern_key = f"blocks.{layer}.attn.hook_pattern"
                if pattern_key not in cache:
                    continue

                attn_pattern = cache[pattern_key][0]  # [n_heads, seq_len, seq_len]

                # Column 0 = BOS as key. For each query position q > 0,
                # attn_pattern[:, q, 0] is the weight on BOS.
                # Average over query positions 1..seq_len-1
                # shape: [n_heads, seq_len-1]
                attn_to_bos = attn_pattern[:, 1:, 0]  # skip q=0 (BOS attending to itself)
                mean_attn_to_bos = attn_to_bos.mean(dim=-1)  # [n_heads]

                attn_to_bos_sums[layer] += mean_attn_to_bos.cpu().float().numpy()

            count += 1
            del cache
            if device != "cpu":
                torch.cuda.empty_cache()

    if count == 0:
        raise ValueError("No samples processed.")

    return attn_to_bos_sums / count


def identify_sink_heads(
    bos_query_entropy: np.ndarray,
    bos_key_attn: np.ndarray,
    entropy_threshold: float,
    attn_threshold: float,
    model_name: str,
) -> list[dict]:
    """
    A head is a "sink head" if:
    1. BOS-as-query entropy is low (BOS attends in a focused, non-diffuse way)
    2. BOS-as-key attention weight is high (other tokens dump probability here)

    Using both criteria because either alone gives false positives:
    - Entropy alone: could just be a head that attends to the previous token,
      not specifically a sink
    - Attn-to-BOS alone: any early-layer induction head might hit this

    The intersection should be cleaner.

    entropy_threshold: heads with entropy BELOW this are candidates (low = focused)
    attn_threshold: heads with mean attn-to-BOS ABOVE this are candidates
    """
    n_layers, n_heads = bos_query_entropy.shape
    candidates = []

    for layer in range(n_layers):
        for head in range(n_heads):
            e = float(bos_query_entropy[layer, head])
            a = float(bos_key_attn[layer, head])

            is_low_entropy = e < entropy_threshold
            is_high_attn = a > attn_threshold

            if is_low_entropy and is_high_attn:
                candidates.append({
                    "model": model_name,
                    "layer": layer,
                    "head": head,
                    "bos_query_entropy_bits": round(e, 4),
                    "mean_attn_to_bos": round(a, 4),
                })

    # Sort by attention-to-BOS descending - more useful ordering for next steps
    candidates.sort(key=lambda x: x["mean_attn_to_bos"], reverse=True)

    print(f"  Found {len(candidates)} sink head candidates "
          f"(entropy < {entropy_threshold:.2f} bits, attn_to_bos > {attn_threshold:.2f})")

    if candidates:
        top = candidates[0]
        print(f"  Top candidate: L{top['layer']}H{top['head']} "
              f"(entropy={top['bos_query_entropy_bits']:.3f}, "
              f"attn_to_bos={top['mean_attn_to_bos']:.3f})")

    return candidates


def run_model(
    model_name: str,
    texts: list[str],
    entropy_threshold: float,
    attn_threshold: float,
    max_seq_len: int,
    device: str,
) -> tuple[list[dict], dict]:
    """
    Full pipeline for a single model: load, tokenize, compute metrics, identify sinks.

    Returns (sink_candidates, raw_stats_dict) so I can save both the candidates
    and the raw entropy/attn arrays for later visualization.
    """
    tl_name = TL_MODEL_NAMES[model_name]
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({tl_name})")
    print(f"{'='*60}")

    # Load model - dtype=torch.bfloat16 halves memory, negligible quality loss
    # for this kind of analysis (we're not generating, just extracting attention)
    print(f"  Loading model...")
    try:
        model = HookedTransformer.from_pretrained(
            tl_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False,  # keep original QK structure
            dtype=torch.bfloat16,
            device=device,
        )
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return [], {}

    model.eval()

    cfg = model.cfg
    print(f"  n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, d_model={cfg.d_model}")
    print(f"  d_head={cfg.d_head}, context_length={cfg.n_ctx}")

    # Tokenize - do this once per model since vocab might differ
    # (Pythia all use the same tokenizer but good to not assume)
    print(f"  Tokenizing {len(texts)} samples...")
    token_sequences = tokenize_and_truncate(model, texts, max_seq_len=max_seq_len)

    if len(token_sequences) < 10:
        print(f"  WARNING: only {len(token_sequences)} valid sequences. Results may be noisy.")

    # Compute both metrics
    print(f"\n  [1/2] BOS-as-query entropy...")
    bos_query_entropy = compute_attention_entropy_at_bos(
        model, token_sequences, device
    )

    print(f"\n  [2/2] BOS-as-key attention weight...")
    bos_key_attn = compute_bos_column_entropy(
        model, token_sequences, device
    )

    # Identify sink heads with both criteria
    print(f"\n  Identifying sink heads...")
    candidates = identify_sink_heads(
        bos_query_entropy,
        bos_key_attn,
        entropy_threshold=entropy_threshold,
        attn_threshold=attn_threshold,
        model_name=model_name,
    )

    # Package raw stats for later use (heatmap plotting etc.)
    raw_stats = {
        "model": model_name,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "bos_query_entropy": bos_query_entropy.tolist(),
        "mean_attn_to_bos": bos_key_attn.tolist(),
        "n_sequences": len(token_sequences),
    }

    # Explicitly delete model to free VRAM before next model
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return candidates, raw_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify attention sink heads in Pythia models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pythia-70m", "pythia-160m", "pythia-410m"],
        choices=PYTHIA_MODELS,
        help="Which Pythia models to analyze",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of text samples from corpus",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Max BOS-as-query entropy (bits) for a head to be a sink candidate. "
             "log2(seq_len) is the max; 0.5 bits is very focused attention.",
    )
    parser.add_argument(
        "--attn_threshold",
        type=float,
        default=0.3,
        help="Min mean attention-to-BOS for a head to be a sink candidate. "
             "0.3 means 30% of attention mass going to BOS on average.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Truncate sequences to this many tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sink_candidates.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0' etc.",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=64,
        help="Minimum word count for corpus samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Output setup
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Also save raw stats alongside candidates
    raw_output_path = output_path.parent / (output_path.stem + "_raw_stats.json")

    print(f"\nConfig:")
    print(f"  Models: {args.models}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  entropy_threshold: {args.entropy_threshold} bits")
    print(f"  attn_threshold: {args.attn_threshold}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  output: {output_path}")

    # Load corpus once, reuse across models
    texts = load_text_sample(args.n_samples, min_length=args.min_text_length)

    all_candidates = []
    all_raw_stats = []

    for model_name in args.models:
        candidates, raw_stats = run_model(
            model_name=model_name,
            texts=texts,
            entropy_threshold=args.entropy_threshold,
            attn_threshold=args.attn_threshold,
            max_seq_len=args.max_seq_len,
            device=device,
        )
        all_candidates.extend(candidates)
        if raw_stats:
            all_raw_stats.append(raw_stats)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total sink head candidates found: {len(all_candidates)}")
    print(f"\nBy model:")
    for model_name in args.models:
        model_candidates = [c for c in all_candidates if c["model"] == model_name]
        if model_candidates:
            layers = [c["layer"] for c in model_candidates]
            print(f"  {model_name}: {len(model_candidates)} candidates, "
                  f"layers {min(layers)}-{max(layers)}")
        else:
            print(f"  {model_name}: 0 candidates "
                  f"(maybe threshold is too strict?)")

    # Save results
    output_data = {
        "config": {
            "models": args.models,
            "n_samples": args.n_samples,
            "entropy_threshold": args.entropy_threshold,
            "attn_threshold": args.attn_threshold,
            "max_seq_len": args.max_seq_len,
            "seed": SEED,
        },
        "candidates": all_candidates,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved candidates to {output_path}")

    with open(raw_output_path, "w") as f:
        json.dump(all_raw_stats, f, indent=2)
    print(f"Saved raw stats to {raw_output_path}")

    # Quick sanity check: are the same layers showing up across models?
    # If sink heads are a systematic phenomenon, I'd expect L0 and maybe
    # one or two other consistent layers to appear regardless of scale.
    if len(all_candidates) > 0:
        print(f"\nLayer distribution across all candidates:")
        from collections import Counter
        layer_counts = Counter(c["layer"] for c in all_candidates)
        for layer, cnt in sorted(layer_counts.items()):
            bar = "█" * cnt
            print(f"  L{layer:2d}: {bar} ({cnt})")
        print("\n(If L0 dominates, that matches the StreamingLLM finding.)")
        print("(If it's spread out, the sink role might be distributed differently at scale.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())