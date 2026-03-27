# Parameter Golf - OpenAI Challenge

**Author:** Ciprian-Florin Ifrim | **Date:** March 2026

Submission repository for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best language model fitting in a **16MB artifact** within **10 minutes on 8xH100 SXM GPUs**, evaluated by bits-per-byte (BPB) on the FineWeb validation set.

Three architectures explored: **Ternary Transformers** (BitNet b1.58), **Binary Transformers** (1-bit), and **ByteJEPA** (Mamba-2 SSM with LeWorldModel-style latent prediction). Full experiment logs covering 250+ runs: [RESULTS.md](results/RESULTS.md), [RESULTS_CONTINUED.md](results/RESULTS_CONTINUED.md) and [RESULTS_JEPA](results/RESULTS_JEPA.md). Training logs: [logs/cuda/](logs/cuda/).

---

## Summary of Results

### 10-Minute Submissions (competition-valid, 8xH100 SXM, <=600s)

| Architecture | Config | Sliding BPB | val_bpb | Artifact | Params | Seeds |
|-------------|--------|-------------|---------|----------|--------|-------|
| **Ternary v2** | 10L 768d relu2 4x MLP, EMBED=312, BF16 scales | **1.1539** | 1.1803 | 15.88MB | 74.3M | 3 (std 0.0004) |
| Ternary v1 | 10L 768d relu2 4x MLP, EMBED=254 | 1.1570 | 1.1821 | 15.92MB | 73.7M | 3 (std 0.0007) |
| ByteJEPA BPE | 10L 640d Mamba-2, mlp=4 every=2 | 1.2566 | 1.2721 | 15.50MB | 32.8M | 1 |
| ByteJEPA Byte | 10L 768d Mamba-2, mlp=3 every=2 | 1.3263 | 1.3348 | 15.86MB | 37.0M | 1 |

### Extended Training (unconstrained compute, notable submissions)

| Architecture | Config | Sliding BPB | val_bpb | Artifact | Steps | Time |
|-------------|--------|-------------|---------|----------|-------|------|
| **Ternary** | 10L 768d + SmearGate, 100k steps | **1.1090** | 1.1344 | 15.95MB | 100k | ~3h |
| Binary | 15L 768d + SmearGate, 50k steps | 1.1239 | 1.1497 | 15.67MB | 50k | ~2h |
| ByteJEPA BPE | 10L 640d Mamba-2, 100k steps | 1.2064 | 1.2235 | 15.75MB | 100k | ~2.7h |

---

## 1. Ternary Transformer (BitNet b1.58)

> PRs: [#640](https://github.com/openai/parameter-golf/pull/640) (original), [#641](https://github.com/openai/parameter-golf/pull/641) (v2 with BF16 scales + EMBED=312)

BitNet b1.58 ternary quantisation constrains weights to {-1, 0, +1} at ~1.6 bits/param with per-group (128) absmean scaling. This allows 74.3M parameters to fit in 15.88MB, compared to ~4M at full precision. The architecture is a U-Net transformer with learned skip connections, GQA (8 heads, 4 KV heads), relu2 4x MLP, polynomial softcap, YaRN positional encoding, and FlashAttention-3.

### v2 Results (3 seeds, 8xH100 SXM, 600s)

| Seed | Steps | Sliding BPB | val_bpb | RT bpb | RT gap | Artifact |
|------|-------|-------------|---------|--------|--------|----------|
| 7 | 6,530 | **1.1535** | 1.1802 | 1.1808 | 0.0006 | 15,951,196 bytes |
| 42 | 6,540 | 1.1542 | 1.1805 | 1.1824 | 0.0019 | 15,952,348 bytes |
| 1337 | 6,530 | 1.1540 | 1.1803 | 1.1811 | 0.0008 | 15,953,260 bytes |
| **Mean** | **6,533** | **1.1539** | **1.1803** | **1.1814** | **0.0011** | **15,952,268 bytes** |

### v1 -> v2 Improvements

| Metric | v1 (#640) | v2 (#641) | Delta |
|--------|-----------|-----------|-------|
| Sliding BPB | 1.1570 | 1.1539 | -0.0031 |
| RT gap | 0.0021 | 0.0011 | -0.0010 |
| Seed std | 0.0007 | 0.0004 | more stable |
| EMBED_DIM | 254 | 312 | +58 dims |

Two changes: BF16 scale storage (eliminates shrinkage correction amplification at high zero_frac, zero byte cost) and EMBED_DIM 254->312 (richer token representations within freed budget).

### Extended Ternary (100k steps, ~3h)

SmearGate enabled, 100k iterations unconstrained. Reaches **1.1090 BPB sliding** -- 0.0445 below the 10-minute submission. The BF16 scale fix was critical: without it, 150k step runs with FP16 scales showed catastrophic 0.039 BPB roundtrip gaps. With BF16, the gap stays at 0.0022 at 100k steps.

### Key Techniques

- **Width over depth:** 768d/10L outperforms 512d/25L -- faster steps (91ms vs 127ms) yield more training in 600s
- **relu2 activation:** -0.024 BPB over relu at zero compute cost
- **NeoMuon** (3 Newton-Schulz steps): compensates for ternary STE gradient attenuation
- **Base-3 + LZMA compression:** 5 trits/byte packing, 39% reduction over int8+zlib
- **FP8 QAT:** halves fp_params storage, only 0.002 BPB roundtrip penalty
- **Temperature scaling (T=0.90):** auto-calibrated on training data

---

## 2. Binary Transformer (1-bit)

> PR: [#640 non-record track](records/track_non_record_16mb/)

Binary quantisation ({-1, +1}, exactly 1 bit/param) packs 60% more parameters per MB than ternary, enabling 15 layers and 106.2M parameters in 15.67MB. The tradeoff: no zero state means less representational flexibility per weight, requiring more training steps to converge.

### Results (seed=42, 50k steps, ~2h)

| Metric | Value |
|--------|-------|
| Sliding BPB | **1.1239** |
| val_bpb | 1.1497 |
| RT bpb | 1.1516 |
| Artifact | 15.67MB |
| Parameters | 106.2M |

Within 10 minutes (4,820 steps), binary's best achieves 1.1824 sliding -- 0.025 BPB worse than ternary. The zero state is worth more at convergence than the density advantage. Binary only surpasses ternary after ~15k steps of training.

### Architecture Differences from Ternary

- 15 layers (vs 10) enabled by 1 bit/param density
- **SmearGate:** causal cumulative mean blending with learned tanh gate -- adds 22ms/step but -0.007 BPB at scale
- No shrinkage correction needed (no zero state, correction factor always 1.0)
- Clean roundtrip: RT gap 0.0019 at 50k steps with FP16 scales

---

## 3. ByteJEPA (Mamba-2 SSM + JEPA)

> PR: [#903](https://github.com/openai/parameter-golf/pull/903)

First application of LeWorldModel-style JEPA (Maes et al. 2026) to text language modelling, combined with Mamba-2 state-space models. The encoder learns to predict its own next latent state via MSE while simultaneously training a cross-entropy decode head -- no attention, no EMA, no stop-gradient.

### BPE Results (10min and extended)

| Config | Sliding BPB | val_bpb | Artifact | Steps | Time |
|--------|-------------|---------|----------|-------|------|
| 10min (8xH100, 600s) | 1.2566 | 1.2721 | 15.50MB | 6,090 | 600s |
| **100k steps** | **1.2064** | **1.2235** | **15.75MB** | **100k** | **~2.7h** |

### Architecture

```
Tokens -> Embedding -> [Mamba-2 SSM + ReLU2 MLP] x 10 (U-Net skips) -> RMSNorm -> h
    JEPA branch:   h -> Projector -> z -> Predictor (3-step rollout) -> MSE loss
    Decode branch: h -> Tied lm_head -> Logits -> CE + Z-loss
    SIGReg:        z -> Per-timestep Gaussian regularization
```

- 10L 640d Mamba-2 SSM (expand=1, d_state=64) with relu2 4x MLP on alternate blocks
- LeWorldModel temporal latent prediction (not cross-view like LLM-JEPA)
- Per-timestep SIGReg (Epps-Pulley characteristic function test, integration range [0.2, 4.0])
- JEPA components (projector, predictor, pred_proj) discarded from artifact -- zero byte cost
- INT4 + FP8 QAT with snap/restore for Mamba-2's fused CUDA kernels
- Dual tokenizer: byte (256 vocab) or BPE (8192 vocab) via single flag

### Key Findings

- **BPE essential for competitive BPB:** byte-level (1.3263) cannot match subword (1.2566) due to the tokens/bytes ratio in the BPB formula
- **Roundtrip improvement from QAT:** quantized BPB consistently better than pre-quantization BPB -- the model optimizes for the INT4 grid
- **MLP every block vs alternate:** mlp=3 on all 8 blocks (1.2715) outperforms mlp=3 on 4/8 blocks (1.3051), but mlp=4 on 5/10 blocks (1.2721) matches mlp=3 on 8/8 blocks while enabling deeper architecture
- **JEPA stays alive with BPE:** prediction task remains meaningful through training (~0.004 loss), unlike byte-level where it saturates quickly (~0.003)

### LeWorldModel Adaptation

The implementation adapts LeWorldModel from robotics video to text. Key divergences: addition of CE loss (mandatory for BPB evaluation), simplified 2-layer MLP predictor (vs 6-layer transformer), no action conditioning, per-timestep SIGReg matching the paper's Algorithm 1 specification. Full discussion of faithful vs deliberate divergences in the [ByteJEPA README](records/ByteJEPA-Mamba2/README.md).

---

## Repository Structure

```
.
├── train_gpt_cuda_ternary.py       # Ternary transformer (8xH100)
├── train_gpt_cuda_binary.py        # Binary transformer (8xH100)
├── train_jepa_ssm.py               # ByteJEPA Mamba-2 (8xH100)
├── train_gpt_mlx_ternary.py        # MLX ternary (Apple Silicon dev)
│
├── run_cuda_ternary.sh             # Launch: ternary
├── run_cuda_binary.sh              # Launch: binary
├── run_jepa_bpe.sh                 # Launch: ByteJEPA BPE mode
├── run_jepa_ssm.sh                 # Launch: ByteJEPA byte mode
├── run_combined.sh                 # Launch: JEPA then ternary sequential
│
├── setup.sh                        # Environment setup (ternary/binary)
├── setup_jepa.sh                   # Environment setup (ByteJEPA + mamba_ssm)
│
├── RESULTS.md                      # Experiment log: 250+ runs (ternary/binary)
├── RESULTS_CONTINUED.md            # Post-submission research (BF16, EMBED, RMS)
│
├── records/
│   ├── track_10min_16mb/           # Ternary v1 submission (#640)
│   ├── track_10min_16mb_v2/        # Ternary v2 submission (#641)
│   ├── track_non_record_16mb/      # Binary notable (#640)
│   ├── track_non_record_ternary/   # Ternary 100k notable
│   └── ByteJEPA-Mamba2/           # ByteJEPA submission (#903)
│
├── research/
│   ├── tversky_investigation.ipynb
│   ├── microbenchmark.ipynb
│   ├── charcnn.ipynb
│   └── asymmetric_tokenizer_test.ipynb
│
├── models/                         # Final artifacts
├── logs/cuda/                      # Training logs
└── data/                           # Tokenizers and dataset scripts
```

---

## Quick Start

### Ternary (competition submission)

```bash
bash setup.sh
conda activate golf
SEED=42 bash run_cuda_ternary.sh
```

Trains for 600s on 8xH100. Expected: ~1.154 BPB sliding, ~15.95MB artifact.

### Binary (extended, ~2h)

```bash
conda activate golf
bash run_cuda_binary.sh
```

50k steps unconstrained. Expected: ~1.124 BPB sliding, ~15.67MB artifact.

### ByteJEPA BPE (competition or extended)

```bash
bash setup_jepa.sh
conda activate golf
bash run_jepa_bpe.sh                                          # 10min
MAX_WALLCLOCK_SECONDS=0 ITERATIONS=100000 bash run_jepa_bpe.sh  # extended
```

10min expected: ~1.257 BPB sliding. Extended expected: ~1.206 BPB sliding.

---

## Configuration Reference

Key parameters for the ternary v2 submission (full list in shell scripts):

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_LAYERS | 10 | Width beats depth at this budget |
| MODEL_DIM | 768 | Optimal for 10L ternary |
| MLP_MULT | 4 | relu2 4x, fused gate+up |
| EMBED_DIM | 312 | Largest multiple of 8 fitting budget |
| VOCAB_SIZE | 8192 | 8k BPE, largest single improvement |
| FP_STORAGE | FP8 | Halves fp_params, enables wider MLP |
| ACTIVATION | relu2 | -0.024 BPB over relu, zero cost |
| TRAIN_BATCH_TOKENS | 524288 | Optimal for ternary STE |
| SLIDING_EVAL_STRIDE | 16 | Maximum context per scored token |
| TEMP_SCALING | 1 | Auto-calibrates to T=0.90 |

---

## PRs

### #640 - Ternary v1 (1.1570 BPB, 3-seed)
<img width="907" height="431" alt="PR640" src="https://github.com/user-attachments/assets/0f89b8eb-c365-4a54-b18e-5cc37d4f8bfc" />

### #641 - Ternary v2 (1.1539 BPB, 3-seed)
<img width="983" height="606" alt="PR641" src="https://github.com/user-attachments/assets/e95d459e-2743-4289-b8b2-b40735e190de" />

### #903 - ByteJEPA Mamba-2 (1.2064 BPB, extended)

---

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).