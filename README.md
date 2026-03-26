# Parameter Golf - OpenAI Challenge
**Author:** Ciprian-Florin Ifrim | **Date:** March 2026
<br>
<br>
Submission for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100 SXM GPUs**, evaluated by bits-per-byte (BPB) on the FineWeb validation set.

This repo uses work previously done (July 2024 - September 2025) on the **SPARROW/SPARROW-Next & AURI MLM (Microcontroller Language Model/Micro Language Model - Private Research)** together with the [Ternary Transformer Lab Personal Repository](https://github.com/CiprianFlorin-Ifrim/ternary-transformer-lab) and [Transformer Tversky Prototypes Lab](https://github.com/CiprianFlorin-Ifrim/transformer-tversky-prototypes-lab).

The purpose was to not only have a very large model (parameter wise) that can be trained well in 10 minutes, but with _infinite compute_ is SOTA, while consuming extremely little space, RAM and with custom kernel, processing power, **inspired by the BitNet 1-bit and 1.58-bit models from Microsoft**.

---

## Results

### Records Leaderboard Results

| Track | Config | Sliding BPB | Artifact | Params | Training Time |
|-------|--------|-------------|----------|--------|---------------|
| **Ternary** | 10L 768d relu² 4×MLP fp8 | **1.1570** (mean, 3 seeds) | 15.92MB | 73.7M | 599s |
| Baseline | 9L 512d int8+zlib | 1.2244 | 15.86MB | 17.1M | 600s |

Ternary improves over baseline by **0.067 bpb** within the competition budget. Three-seed standard deviation: 0.0007 bpb.

### Notable Leaderboard Results

| Track | Config | Sliding BPB | Artifact | Params | Steps | Training Time |
|-------|--------|-------------|----------|--------|-------|---------------|
| **Binary** | 15L 768d relu² 4×MLP fp8 smear | **1.1239** | 15.60MB | 106.2M | 50,000 | 7,763s |
| Baseline | 9L 512d SP1024 int8+zlib | 1.1749 | 15.81MB | 17.1M | 500,000 | 14,400s |

These have been accepted and added into the main openai/parameter-golf repo. Check last section.


---

## Record: 1.1570 BPB — 73.7M Ternary U-Net Transformer

> Full record details and compliance: [`records/track_10min_16mb/`](records/track_10min_16mb/)

**BitNet b1.58 + 10L + NeoMuon + 4x relu² MLP + Factored Tied Embedding + Poly5 Softcap + YaRN 2048 + 8192 BPE + FP8 QAT + Base-3 LZMA + Stride-16 Sliding Eval**

**val_bpb: 1.1570** (3-seed mean sliding, std 0.0007) | **15.99 MB** max artifact | 8×H100 SXM, 599s

> Full experiment log covering 250+ runs, ablations, and decision rationale: [RESULTS.md](RESULTS.md). Complete training logs: [logs/cuda/](logs/cuda/).

### Results (3 seeds, 8×H100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s16) | val_bpb | RT bpb | Artifact |
|------|-------|---------|-------------------|---------|--------|----------|
| 42 | 6,530 | 91.7 | **1.1565** | 1.1816 | 1.1837 | 15,993,853 bytes |
| 1337 | 6,520 | 91.9 | 1.1568 | 1.1825 | 1.1839 | 15,995,705 bytes |
| 7 | 6,530 | 91.8 | 1.1578 | 1.1823 | 1.1850 | 15,992,753 bytes |
| **Mean** | **6,527** | **91.8** | **1.1570** | **1.1821** | **1.1842** | **15,994,104 bytes** |
| **Std** | **5** | **0.1** | **0.0007** | **0.0005** | **0.0007** | **1,498 bytes** |

### Architecture

- 10 transformer layers, dim=768, 8 heads, 4 KV heads (GQA), head_dim=96
- BitNet b1.58 ternary quantisation: weights {-1, 0, +1}, ~1.6 bits/param, per-group (128) absmean scaling
- 4x MLP expansion (hidden=3072) with **relu²** activation, fused gate+up projection
- U-Net encoder/decoder with learned skip weights (ones-init) and per-block residual mix from input embedding
- Factored tied embedding: 8192×254 bottleneck with learned 254-to-768 and 768-to-254 projections
- Polynomial softcap (degree 5, cap=10) with Z-loss regularisation (1e-4)
- YaRN positional encoding (max_len=2048, ROPE_BASE=5000)
- Fused QKV projection (single TernaryLinear)
- FlashAttention-3 (Hopper native kernels)
- 73.7M parameters, 15.92MB artifact (64.9M ternary + 2.5M fp8 + 70KB code)

### Key Techniques

**Architecture**
- **Width over depth:** 768d/10L outperforms 512d/25L — faster steps (91ms vs 127ms) yield 6,530 vs 4,720 steps in 600s
- **4x relu² MLP:** relu² is -0.024 bpb over relu at zero cost; 4x width adds -0.008 bpb over 3x at same step budget
- **EMBED_DIM=254:** frees ~4MB for wider MLP; 254 = 256-2 to fit code within the byte budget

**Training**
- **NeoMuon** with 3 Newton-Schulz steps: compensates for ternary STE gradient attenuation; 3 steps equivalent to 5 at convergence (+190 free steps)
- **Fused QKV + fused relu²:** ~4-6ms/step saving (~180 extra training steps)
- **FlashAttention-3:** -9% step time (~380 free steps)
- **524k batch tokens:** optimal for ternary STE — 262k too noisy, 1M loses gradient updates

**Evaluation**
- **Temperature scaling (T=0.90):** auto-selected via 5-point grid on training tokens (not manually chosen); relu² logits slightly underconfident
- **Sliding window (stride=16):** full context per scored token, ~0.025 bpb over chunked eval

**Compression**
- **Base-3 + LZMA (preset=9):** 5 trits/byte packing, 39% reduction over int8+zlib; auto-compared against bitmask per run
- **FP8 QAT (e4m3):** halves fp_params (~5MB to ~2.5MB), only 0.002 bpb RT penalty
- **Shrinkage fix:** corrects ternary zero-fraction scale mismatch, eliminating all roundtrip gaps

### Compliance

- [x] 3 seeds run on 8×H100 SXM
- [x] All 3 seeds train in <=600s (max: 599.7s)
- [x] All 3 seeds artifact <=16,000,000 bytes (max: 15,995,705)
- [x] Sliding window eval stride=16, consistent (std=0.0007)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute

---

## Notable Non-Record: 1.1239 BPB — 106.2M Asymmetric Binary U-Net Transformer

> Full non-record details and compliance: [`records/track_non_record_16mb/`](records/track_non_record_16mb/)

**1-bit Quantisation + 15L (7 Encoder - 8 Decoder) + NeoMuon + 4x relu² MLP + SmearGate + Factored Tied Embedding + Poly5 Softcap + YaRN 2048 + 8192 BPE + FP8 QAT + LZMA + Stride-16 Sliding Eval**

**val_bpb: 1.1239** (sliding, seed=42) | **15.67 MB** artifact | 8×H100 SXM, 50k steps (~2.15h)

> This is a **non-record submission** — training exceeds the 10-minute wallclock constraint (50,000 steps / ~2.15 hours). Submitted to demonstrate the compression frontier: 106.2M parameters in 15.67MB via 1-bit quantisation. Over 120M possible with FP4 (implemented) at a worse bpb. Full experiment log: [RESULTS.md](RESULTS.md). Complete training logs: [logs/cuda/](logs/cuda/).

### Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| Sliding BPB (s16) | **1.1239** |
| val_bpb | 1.1497 |
| RT bpb | 1.1516 |
| Steps | 50,000 |
| ms/step | 155.3 |
| Training time | 7,763s (~2.15h) |
| optimal_T | 0.90 |
| Artifact | 15,670,651 bytes (15.67MB) |
| Parameters | 106,154,616 |

Binary reaches better absolute quality but requires ~13x more training time. Within the 10-minute budget, binary's best fitting run (14L, 4,820 steps) scores 1.1824 sliding — 0.025 bpb worse than ternary. The zero state is worth more at convergence than the 60% parameter density advantage.

### Architecture

- 15 transformer layers, dim=768, 8 heads, 4 KV heads (GQA), head_dim=96
- Binary quantisation: weights {-1, +1}, 1 bit/param, per-group (128) absmean scaling
- 4x MLP expansion (hidden=3072) with **relu²** activation, fused gate+up projection
- U-Net encoder/decoder with learned skip weights (ones-init) and per-block residual mix from input embedding
- **SmearGate:** causal cumulative mean blending with learned tanh gate, zero-init for safe residual start
- Factored tied embedding: 8192×254 bottleneck with learned projections
- Polynomial softcap (degree 5, cap=10) with Z-loss regularisation (1e-4)
- YaRN positional encoding (max_len=2048, ROPE_BASE=5000)
- Fused QKV projection
- FlashAttention-3 (Hopper native kernels)
- 106.2M parameters, 15.67MB artifact (97.3M binary + 2.5M fp8 + 70KB code)

### Key Techniques

**Architecture**
- **Binary quantisation:** 1 bit/param packs 60% more parameters per MB than ternary (1.6 bits/param), allowing 15 layers vs 10 within similar budget
- **4x relu² MLP:** relu² strictly dominates relu; 4x width outperforms 3x even with fewer layers at matched budget
- **SmearGate:** blends each position with causal cumulative mean; adds 22ms/step overhead but provides -0.007 bpb at scale; viable here because the run is not wallclock-constrained

**Training**
- **NeoMuon** with 3 Newton-Schulz steps
- **50,000 steps unconstrained:** at 4,000 steps (the 10-minute equivalent) binary lags ternary by 0.025 bpb; extended training closes the gap and surpasses it
- **524k batch tokens:** optimal gradient quality / step count tradeoff

**Evaluation**
- **Temperature scaling (T=0.90):** auto-selected via calibrated grid (not manually chosen)
- **Sliding window (stride=16):** full context per scored token

**Compression**
- **Bit-packing + LZMA (preset=9):** binary weights pack at exactly 1 bit/param before LZMA entropy coding
- **FP8 QAT (e4m3):** for non-binary parameters; clean roundtrip, no shrinkage correction needed (no zero state)
- **No EMA:** despite clean binary roundtrip math, EMA still hurts quality by 0.03 bpb in practice

### Compliance

- [x] Artifact <=16,000,000 bytes (15,670,651)
- [x] Sliding window eval stride=16
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
- [x] Train time: **non-record submission** (7,763s / 2.2h / 50,000 steps)

---

## Architecture Overview

**Ternary U-Net Transformer** with BitNet b1.58 quantisation ({-1, 0, +1}, ~1.6 bits/param). Key components: U-Net skip connections with learned weights, factored embedding (EMBED_DIM=254), fused QKV and relu² projections, polynomial softcap with Z-loss regularisation, YaRN positional encoding, and Muon optimizer with Newton-Schulz orthogonalisation.

Compression: Base-3 packing + LZMA (preset=9), achieving 39% reduction over int8+zlib. FP8 QAT for non-ternary parameters. Artifact fits within 16,000,000 bytes including code.

Full experiment log with 250+ runs across dev, scaling, and final configurations: see [RESULTS.md](RESULTS.md).

---

## Repository Structure

```
.
├── train_gpt_cuda_ternary.py    # Main ternary training script (8×H100)
├── train_gpt_cuda_binary.py     # Binary quantisation variant
├── train_gpt_mlx_ternary.py     # MLX ternary (Apple Silicon dev)
├── train_gpt_mlx.py             # MLX baseline
├── train_gpt_mlx_float.py       # MLX float reference
├── train_gpt_mps.py             # MPS backend (Apple Silicon)
│
├── run_cuda_ternary.sh          # Launch: ternary on 8×H100
├── run_cuda_binary.sh           # Launch: binary on 8×H100
├── run_local_mlx_ternary.sh     # Launch: ternary on Apple Silicon (MLX)
├── run_local_mlx_float.sh       # Launch: float on Apple Silicon (MLX)
├── run_local_mps.sh             # Launch: MPS backend
├── run_mlx_ternary.sh           # MLX ternary (remote)
├── run_mlx_float.sh             # MLX float (remote)
│
├── setup.sh                     # Full environment setup (conda + deps + data)
├── RESULTS.md                   # Complete experiment log (250+ runs)
├── README.md                    # This file
│
├── models/
│   ├── final_model.ternary.ptz  # Final ternary artifact
│   ├── final_model.binary.ptz   # Final binary artifact
│   ├── old_model.pt             # Early checkpoint
│   └── old_model.int8.ptz       # Early int8 checkpoint
│
├── research/
│   ├── tversky_investigation.ipynb    # Tversky similarity analysis
│   ├── microbenchmark.ipynb           # H100 kernel timing
│   ├── charcnn.ipynb                  # ByteCNN embedding experiment
│   ├── asymmetric_tokenizer_test.ipynb # Asymmetric tokenizer study
│   └── plots/
│       └── tversky_sweep.png
│
├── logs/                        # Training logs (~250 runs, not listed)
│   └── cuda/                    # H100 run logs
│
├── records/
│   ├── track_10min_16mb/        # Official submission README + artifacts
│   └── track_non_record_16mb/   # Non-record submission README + artifacts
│
└── data/
    ├── tokenizers/              # SentencePiece BPE models (1k–8k vocab)
    └── original_data_scripts/   # Dataset preparation scripts
```

---

## Quick Start

### 1. Environment Setup (GPU Cloud)

For a fresh instance (e.g. Lambda, RunPod, etc.) with 8×H100:

```bash
git clone https://github.com/<your-fork>/parameter-golf.git
cd parameter-golf
bash setup.sh
```

This installs Miniconda, creates a `golf` conda environment with Python 3.13, installs all dependencies (PyTorch, FlashAttention-3, SentencePiece), and downloads the FineWeb 10B dataset (sp8192 tokenisation).

### 2. Train Ternary Model (Submission)

```bash
conda activate golf
bash run_cuda_ternary.sh
```

The script runs for 599 seconds on 8×H100, producing a ~15.92MB artifact that fits within the 16MB budget. Training completes ~6,530 steps at 91.8ms/step, followed by temperature-scaled sliding window evaluation (stride=16, T=0.90).

Expected output: **~1.157 bpb** sliding, **~1.182 bpb** val.

### 3. Train Binary Model (Extended)

```bash
conda activate golf
bash run_cuda_binary.sh
```

The binary variant trains for 50,000 steps (~2 hours) with smear enabled. Produces a 15.60MB artifact with 97.3M binary-quantised parameters.

Expected output: **~1.124 bpb** sliding, **~1.150 bpb** val.

### 4. Local Development (Apple Silicon)

```bash
bash run_local_mlx_ternary.sh   # MLX backend, 100 steps
bash run_local_mps.sh           # MPS backend
```

Local runs use reduced batch sizes and step counts for rapid iteration. Results at 100 steps are directionally useful but not representative of full convergence.

---

## Configuration

All hyperparameters are set via environment variables in the shell scripts. Key parameters for the ternary submission:

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_LAYERS | 10 | Minimum viable depth at 768d |
| MODEL_DIM | 768 | Width beats depth at this budget |
| MLP_MULT | 4 | 4× relu² MLP, fused gate+up |
| EMBED_DIM | 254 | 256-2 to fit within byte budget |
| VOCAB_SIZE | 8192 | 8k BPE, largest single improvement |
| FP_STORAGE | FP8 | Halves fp_params, enables wider MLP |
| MUON_WD | 0.0 | WD=0 optimal for wide MLP |
| TRAIN_BATCH_TOKENS | 524288 | Optimal gradient quality / step count tradeoff |
| MAX_WALLCLOCK_SECONDS | 599 | 1s safety margin |
| ACTIVATION | relu2 | Strictly dominates relu; free improvement |
| SLIDING_EVAL_STRIDE | 16 | Maximum context per scored token |
| TEMP_SCALING | 1 | Auto-calibrates to T=0.90 |

See [RESULTS.md](RESULTS.md) for the complete experiment log covering all 250+ runs, ablations, and the rationale behind every locked decision.

---

## Research Notebooks

The `research/` directory contains Jupyter notebooks used to isolate and understand specific techniques before committing H100 compute:

- **tversky_investigation.ipynb** — Synthetic-data analysis of Tversky similarity for directional vs LM tasks. Confirmed that asymmetric similarity only benefits tasks with directional feature relationships.
- **microbenchmark.ipynb** — Standalone H100 kernel timing for STE, RoPE, softcap, and CE+Z-loss variants. Demonstrated that torch.compile fusion invalidates standalone microbenchmark conclusions.
- **charcnn.ipynb** — ByteCNN embedding generator experiment. Showed embedding collapse from CNN inductive bias.
- **asymmetric_tokenizer_test.ipynb** — 8k BPE input / 256-byte output tokenizer. Confirmed byte independence assumption is mathematically incorrect.



## PRs
### PR640
<img width="907" height="431" alt="PR640" src="https://github.com/user-attachments/assets/0f89b8eb-c365-4a54-b18e-5cc37d4f8bfc" />

### PR641
<img width="983" height="606" alt="PR641" src="https://github.com/user-attachments/assets/e95d459e-2743-4289-b8b2-b40735e190de" />

---

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
