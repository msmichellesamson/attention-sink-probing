data/sample_wikitext.py

# Pull 500 sequences from WikiText-103-raw-v1, tokenize to fixed length,
# and cache to disk. This runs once and everything else loads the cache.
#
# Why WikiText-103? It's long-form prose with real sentence structure,
# so BOS tokens are actually meaningful anchors rather than artifacts of
# short-snippet padding. We want sequences where the model has real context
# to build up.
#
# 512 tokens is a compromise: long enough for attention sinks to manifest
# (they tend to show up within the first ~100 tokens and stabilize),
# short enough that we can cache activations for all Pythia scales without
# filling a disk.
#
# NOTE: sampling is deterministic via a fixed seed so every experiment
# file loads the exact same 500 sequences. Change the seed and you'll
# break reproducibility for cached activation files - don't do that
# mid-experiment.

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
N_SEQUENCES = 500
SEQ_LEN = 512
CACHE_DIR = Path(__file__).parent / "cache"
OUTPUT_FILE = CACHE_DIR / "wikitext_sequences.pt"
META_FILE = CACHE_DIR / "wikitext_meta.json"

# Using Pythia tokenizer (NeoX) as the canonical tokenizer for this project.
# The vocab is shared across all Pythia scales which is the whole point -
# we want the same token sequences fed to each model size.
TOKENIZER_NAME = "EleutherAI/pythia-70m"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_cache_sequences(
    n_sequences: int = N_SEQUENCES,
    seq_len: int = SEQ_LEN,
    seed: int = SEED,
    force_rebuild: bool = False,
) -> torch.Tensor:
    """
    Returns tensor of shape (n_sequences, seq_len) with token ids.
    First token of every row is BOS (token id 0 for NeoX tokenizer).

    If cache exists and force_rebuild=False, just loads from disk.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists() and not force_rebuild:
        print(f"Loading cached sequences from {OUTPUT_FILE}")
        sequences = torch.load(OUTPUT_FILE)
        with open(META_FILE) as f:
            meta = json.load(f)
        print(f"  Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]}")
        print(f"  Original seed: {meta['seed']}, tokenizer: {meta['tokenizer']}")
        return sequences

    print("Building sequence cache from scratch...")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # WikiText-103-raw-v1: the 'raw' variant gives us the full text without
    # the stripped/tokenized version. We want the train split - it's big
    # enough that 500 sequences are genuinely independent chunks.
    print("Downloading WikiText-103-raw-v1 (this might take a moment)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Concatenate all text into one big string, then chunk it.
    # Alternative would be to tokenize each article separately and filter
    # by length, but that biases toward longer articles. Chunking is cleaner
    # for the "model sees real prose" requirement.
    print(f"Dataset has {len(dataset)} rows, concatenating text...")
    full_text = "\n\n".join(
        doc for doc in dataset["text"] if doc.strip()
    )

    print(f"Total characters: {len(full_text):,}")

    # Tokenize the full corpus once, then slice. This is faster than
    # tokenizing each chunk individually and avoids boundary artifacts.
    # The 'add_special_tokens=False' here is intentional - we'll prepend
    # BOS manually so it's always at position 0, guaranteed.
    print("Tokenizing full corpus (this takes a while for 103M words)...")
    # use batched encoding for speed; return_tensors doesn't work well on
    # 100M+ token strings so we just get a list back
    encoded = tokenizer.encode(
        full_text,
        add_special_tokens=False,
    )
    print(f"Total tokens: {len(encoded):,}")

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        # NeoX tokenizer: BOS is token 0
        bos_id = 0
        print(f"  tokenizer.bos_token_id was None, using 0 (NeoX default)")
    else:
        print(f"  BOS token id: {bos_id}")

    # Each sequence is: [BOS] + tokens[start : start + seq_len - 1]
    # We need seq_len - 1 content tokens per sequence.
    content_len = seq_len - 1

    # How many non-overlapping chunks can we draw?
    n_available = len(encoded) // content_len
    print(f"  Available non-overlapping chunks: {n_available}")
    assert n_available >= n_sequences, (
        f"Not enough tokens for {n_sequences} sequences of length {seq_len}. "
        f"Have {n_available} available chunks."
    )

    # Sample without replacement from available chunk positions.
    # Using chunk indices rather than token positions to guarantee
    # non-overlapping sequences.
    chunk_indices = random.sample(range(n_available), n_sequences)
    chunk_indices.sort()  # sorted so disk access is sequential (minor)

    sequences = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    for i, chunk_idx in enumerate(chunk_indices):
        start = chunk_idx * content_len
        end = start + content_len
        content = encoded[start:end]
        sequences[i, 0] = bos_id
        sequences[i, 1:] = torch.tensor(content, dtype=torch.long)

    # Sanity check: every row should start with BOS
    assert (sequences[:, 0] == bos_id).all(), "BOS prepend failed"

    # Quick stats
    unique_tokens = sequences.unique().numel()
    print(f"\nSanity checks:")
    print(f"  Shape: {sequences.shape}")
    print(f"  All rows start with BOS ({bos_id}): True")
    print(f"  Unique tokens across all sequences: {unique_tokens}")
    print(f"  Token id range: [{sequences.min().item()}, {sequences.max().item()}]")

    # Check that sequences don't look degenerate (no row that's all BOS etc.)
    # The min of each row should not equal the max (would mean all same token)
    row_min = sequences.min(dim=1).values
    row_max = sequences.max(dim=1).values
    degenerate = (row_min == row_max).sum().item()
    if degenerate > 0:
        print(f"  WARNING: {degenerate} degenerate (constant) sequences found")
    else:
        print(f"  No degenerate sequences")

    print(f"\nSaving to {OUTPUT_FILE}...")
    torch.save(sequences, OUTPUT_FILE)

    meta = {
        "seed": seed,
        "n_sequences": n_sequences,
        "seq_len": seq_len,
        "bos_token_id": bos_id,
        "tokenizer": TOKENIZER_NAME,
        "source": "wikitext-103-raw-v1 train split",
        "chunk_indices": chunk_indices,  # save so we can audit which chunks
        "total_corpus_tokens": len(encoded),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_FILE}")

    return sequences


def load_sequences() -> torch.Tensor:
    """
    Thin wrapper for other scripts to use. Fails loudly if cache doesn't exist
    rather than silently rebuilding - callers should know if they're getting
    stale data.
    """
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Sequence cache not found at {OUTPUT_FILE}. "
            f"Run `python data/sample_wikitext.py` first to build it."
        )
    sequences = torch.load(OUTPUT_FILE)
    return sequences


def load_meta() -> dict:
    if not META_FILE.exists():
        raise FileNotFoundError(f"Meta file not found at {META_FILE}")
    with open(META_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample and cache WikiText sequences for attention sink probing"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if cache exists (will invalidate any cached activations!)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=N_SEQUENCES,
        help=f"Number of sequences to sample (default: {N_SEQUENCES})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length including BOS (default: {SEQ_LEN})",
    )
    args = parser.parse_args()

    if args.force_rebuild:
        print("WARNING: --force-rebuild will invalidate cached activations.")
        print("Make sure you really want this before re-running extract_residuals.py.")
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            exit(0)

    sequences = load_and_cache_sequences(
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        force_rebuild=args.force_rebuild,
    )

    print(f"\nDone. Sequences tensor shape: {sequences.shape}")
    print(f"Cache location: {OUTPUT_FILE}")

    # Print a few token sequences decoded back to text so we can eyeball
    # that they look like real prose and not garbage
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("\n--- First 3 sequences (first 50 tokens each) ---")
    for i in range(3):
        tokens = sequences[i, :50].tolist()
        text = tokenizer.decode(tokens)
        print(f"\nSeq {i}: {repr(text[:200])}")