# Parameter Golf - OpenAI Challenge

**Author:** Ciprian-Florin Ifrim | **Date:** March 2026

Submission for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100 SXM GPUs**, evaluated by bits-per-byte (BPB) on the FineWeb validation set.

---

## Results

| Track | Config | Sliding BPB | Artifact | Params | Constraint |
|-------|--------|-------------|----------|--------|------------|
| **Ternary (submission)** | 10L 768d relu² 4×MLP fp8 | **1.1570** (mean, 3 seeds) | 15.92MB | 73.7M | 599s on 8×H100 |
| **Binary (unconstrained)** | 15L 768d relu² 4×MLP fp8 smear | **1.1239** | 15.60MB | 97.3M | 50k steps (~2h) |
| Baseline | 9L 512d int8+zlib | 1.2244 | — | — | — |

Ternary submission improves over baseline by **0.067 bpb** within the competition budget. Three-seed standard deviation: 0.0007 bpb.

---

## Architecture

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
├── records/                     # Competition submission records
│   ├── track_10min_16mb/        # Official submissions
│   └── track_non_record_16mb/   # Non-record submissions
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

## Submission Reproducibility

Three-seed validation (seeds 1337, 42, 7):

| Seed | Steps | val_bpb | RT bpb | Sliding bpb | Budget |
|------|-------|---------|--------|-------------|--------|
| 1337 | 6520 | 1.1825 | 1.1839 | 1.1568 | 16.00/16.00MB |
| 42 | 6530 | 1.1816 | 1.1837 | 1.1565 | 15.99/16.00MB |
| 7 | 6530 | 1.1823 | 1.1850 | 1.1578 | 15.99/16.00MB |
| **Mean** | **6527** | **1.1821** | **1.1842** | **1.1570** | |
| **Std** | **5** | **0.0005** | **0.0007** | **0.0007** | |

All seeds fit within the 16,000,000 byte budget. Standard deviation of 0.0007 bpb confirms high reproducibility.

---

## Research Notebooks

The `research/` directory contains Jupyter notebooks used to isolate and understand specific techniques before committing H100 compute:

- **tversky_investigation.ipynb** — Synthetic-data analysis of Tversky similarity for directional vs LM tasks. Confirmed that asymmetric similarity only benefits tasks with directional feature relationships.
- **microbenchmark.ipynb** — Standalone H100 kernel timing for STE, RoPE, softcap, and CE+Z-loss variants. Demonstrated that torch.compile fusion invalidates standalone microbenchmark conclusions.
- **charcnn.ipynb** — ByteCNN embedding generator experiment. Showed embedding collapse from CNN inductive bias.
- **asymmetric_tokenizer_test.ipynb** — 8k BPE input / 256-byte output tokenizer. Confirmed byte independence assumption is mathematically incorrect.

---

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
