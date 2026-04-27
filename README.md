# Where Do Attention Sinks Form?

> Probing whether the BOS token's residual stream encodes a learnable "sink role" signal across Pythia scales, and mapping the layer at which that signal first becomes linearly separable.

---

## The question I'm exploring

When a transformer generates text, certain attention heads consistently route a disproportionate fraction of their attention to the first token — even when that token is semantically irrelevant. [Xiao et al. (2023), "Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) documented this phenomenon and showed it was stable enough to exploit for KV-cache compression. What they didn't fully characterize is *when* the sink role gets established in the residual stream — meaning, at which layer does the BOS token's internal representation start looking meaningfully different from a "normal" token, in a way that predicts it will attract sink attention in later layers?

The related question from the [Anthropic monosemanticity paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html) is lurking here too: if superposition allows tokens to encode many features simultaneously, does the sink role compete with or co-exist with other features encoded at BOS? I don't have an answer to the second question yet, but it's shaping how I'm designing the probes.

Going in, I knew: sinks form reliably at BOS, they're more pronounced in larger models, and they can be partially transferred to a second "sentinel" token if you add one. What I didn't know: whether the *residual stream* at BOS develops a linearly separable "sink signal" early in the network, or whether sink behavior is an emergent property of late-layer attention composition with no early footprint.

---

## Why I care

The infrastructure angle got me here first. I was reading about KV-cache eviction strategies and noticed that every sink-aware eviction policy I could find treats sink detection as a heuristic — count low-entropy heads, flag the token, protect it forever. That works, but it's a static rule bolted onto a dynamic system. If the sink role is encoded in the residual stream early enough, you could in principle *predict* which tokens will become sinks before the attention computation that would confirm it, which has real implications for speculative decoding, cache pre-allocation, and streaming inference.

But the deeper reason I'm spending time here is interpretability. I've been thinking a lot about what it means for a model to have "roles" — functional identities that individual positions or neurons settle into during training. The sink phenomenon is one of the cleaner examples where a role is clearly present and measurable. Understanding how it forms layer-by-layer feels like a tractable entry point into the broader question of how representational structure builds up through a forward pass, which matters for any serious attempt at mechanistic understanding of model behavior.

---

## What's in here

**Data sampling**
`data/sample_wikitext.py` — Pulls 500 sequences from [WikiText-103-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext), tokenizes to 512 tokens, and caches to disk. All four experiments use the same sample so comparisons across model sizes aren't confounded by input variation.

**Sink identification**
`src/identify_sinks.py` — Runs a forward pass with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) hooks on each Pythia checkpoint, computes per-head attention entropy at the BOS position, and writes a JSON of `(layer, head)` sink candidates. The entropy threshold is a CLI argument so I can sweep it without editing code. This is the "ground truth" label source for the probes.

**Activation extraction**
`src/extract_residuals.py` — Uses TransformerLens cache hooks to grab the residual stream at BOS position after each layer, and optionally the MLP output at each layer, for a given Pythia checkpoint. Saves tensors to `results/activations/`. Keeping extraction separate from probing turned out to matter — the activation tensors for Pythia-1.4B are large enough that re-running extraction on every probe iteration would be slow.

**Probing**
`src/probe.py` — Trains a logistic regression probe (scikit-learn `LogisticRegressionCV`) at each layer on the saved activation tensors. The label for each sequence is whether BOS becomes a sink (using the `identify_sinks.py` output). Outputs a per-layer accuracy dict to `results/probe_scores/`. Also runs an MLP-out ablation: fits the same probe on *only* the MLP contribution at each layer, to separate attention-mediated and MLP-mediated parts of the signal.

**Plotting**
`src/plot.py` — Reads `probe_scores/` and the sink JSON files to produce two figures: a heatmap of probe accuracy by (model, layer) and a line chart showing where accuracy crosses a threshold per scale. Kept deliberately separate from the experiment code so I can tweak presentation without re-running anything.

**Scratch notebook**
`notebooks/exploration.ipynb` — Where I figured out which TransformerLens cache keys I actually needed, spot-checked attention patterns on single sequences, and verified that hook shapes were what I expected before building the pipeline. It's messy and I'm keeping it that way.

**Findings**
`results/findings.md` — The written-up research output, generated after the experiments. The README summarizes it; the findings doc has the full tables and the longer discussion of where the interpretation gets shaky.

---

## What I'm finding (so far)

- **The signal appears earlier than I expected.** Even at layer 0 — before any attention computation has run — the probe achieves above-chance accuracy on Pythia-1.4B. I don't fully trust this yet: it might be an artifact of positional embedding alone encoding something about BOS that's trivially learnable, rather than a genuine "sink role" signal. I need to run the same probe on a non-BOS position that never becomes a sink as a proper control.

- **Probe accuracy rises sharply between layers 3 and 7 on Pythia-1.4B**, plateaus through the middle layers, and doesn't improve much in the final third of the network. This is consistent with a picture where the sink role is "decided" early and then preserved — but I'm not confident the plateau isn't just the probe saturating.

- **The MLP-out ablation is confusing.** On the smaller models (Pythia-160m, Pythia-410m), the MLP contribution at mid-layers actually carries more of the probe signal than the full residual stream does at the same layer. That's backwards from what I'd naively expect if attention heads are the primary mechanism. I don't have a clean explanation for this yet.

- **Scaling matters, but not monotonically in the way I expected.** Pythia-160m and Pythia-410m show a later "emergence layer" for the probe signal than Pythia-1.4B, which fits the story that larger models develop more stable internal roles. But Pythia-70m is an outlier — the signal is noisier across layers and the sink designation itself is less consistent, which might just mean sinks are weaker in that model rather than that the probing setup is wrong.

- **The entropy threshold for sink designation matters a lot.** I've been running at a default threshold of 0.5 bits below mean entropy, but when I tighten it, the label set changes enough that probe accuracy drops substantially — which suggests I'm not working with a clean binary phenomenon. The "sink" label is more of a spectrum than the Xiao et al. framing implies, at least on this dataset.

- **I haven't validated on a second dataset yet.** Everything above is on WikiText-103. It's a reasonable first corpus but it's clean, English, long-form prose. I'd expect sink behavior to look different on code or short conversational sequences, and I shouldn't overgeneralize until I check.

---

## What I'd do next

1. **Run the positional-embedding control.** Take a non-BOS token that's always at a fixed position (say, the second token) and probe its residual stream with the same setup. If that achieves similar early-layer accuracy, then the "layer 0 signal" is position, not sink role, and I need to be more careful about what the probe is actually learning.

2. **Try probing intermediate checkpoints.** Pythia provides [training checkpoints](https://github.com/EleutherAI/pythia) at regular steps. The natural follow-up question is whether the sink signal emerges gradually during training or switches on suddenly at a particular checkpoint — which would say something about whether sink formation is a phase transition.

3. **Ablate the sink heads and re-probe.** If I use TransformerLens to zero out the sink heads identified in `identify_sinks.py` and then re-run the probe, does early-layer probe accuracy drop? If yes, that would suggest the residual stream signal is being written by the sink heads in earlier layers (i.e. attention-to-attention communication across layers). If no, something else is writing it.

4. **Test whether the early-layer signal is predictive of later sinks in a causal sense.** Right now the probe is correlational. The stronger claim would be: if I can read off the "sink signal" from layer 4, can I accurately predict which heads will sink at layer 16, before I actually run layer 16? That would require a sequential prediction setup rather than the current layer-by-layer classification.

5. **Extend to Pythia-2.8B and Pythia-6.9B.** I've been staying in the range where a T4 can handle the forward passes. Getting to the larger checkpoints would either require a different compute setup or more aggressive batching, but the scaling question is genuinely open and the current range might be too narrow to draw strong conclusions.

6. **Compare against a model without a BOS token.** Some architectures don't use a BOS token and sinks migrate to the first real content token or to punctuation. Testing whether the probing setup generalizes to that case would help distinguish "sink role as a feature of BOS tokens" from "sink role as an emergent feature of whatever token gets volunteered for the role."

---

## Status

The extraction, probing, and plotting pipeline runs end-to-end on Pythia-160m through Pythia-1.4B. The sink identification and activation extraction are working and producing sensible-looking outputs. The probe results in `results/findings.md` are real numbers from real experiments — I haven't fabricated any of them, but I also haven't validated them carefully enough to claim they're robust findings rather than preliminary observations. The positional control I flagged above is the most important gap: until I run it, the layer-0 result in particular should be treated skeptically. The MLP ablation finding is the part I find most interesting and least understood, and it's the place I'd focus first if I had more time this week.

---

## References

- [Xiao et al. (2023) — "Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453)
- [Bricken et al. (2023) — "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html)
- [Biderman et al. (2023) — "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"](https://arxiv.org/abs/2304.01373)
- [Nanda et al. (2022) — "TransformerLens"](https://github.com/TransformerLensOrg/TransformerLens)
- [WikiText-103-raw-v1 dataset](https://huggingface.co/datasets/Salesforce/wikitext)