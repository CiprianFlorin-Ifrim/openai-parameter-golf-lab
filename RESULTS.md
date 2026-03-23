# Parameter Golf — Complete Experiment Log

## Challenge
Train best LM in 16MB artifact, evaluated by BPB on FineWeb val. 10 min training on 8xH100, 10 min eval.
- Baseline: **1.2244 bpb** | SOTA: **~1.2029 bpb**

## Hardware
- **Dev:** RTX 5090 32GB, single GPU | **Final:** 8xH100 SXM 80GB
- **RTX 5090 constraint:** Triton shared memory ceiling 101KB/SM (vs 228KB on H100). Blocks several techniques.

---

## Current Best Configuration

```bash
NUM_LAYERS=12         MODEL_DIM=512        NUM_HEADS=8
NUM_KV_HEADS=4        MLP_MULT=2           VOCAB_SIZE=8192
ACTIVATION=swiglu     LOGIT_SOFTCAP=10     QK_GAIN_INIT=2.25
ROPE_BASE=10000       ATTN_PROJ_TYPE=tversky
TVERSKY_NUM_FEATURES=128  TVERSKY_TERNARY_PROTOS=1
TIE_EMBEDDINGS=1      FP_STORAGE=0         EVAL_DEPTH_RECURRENCE=2
MTP_HEADS=0           UNTIE_AT_FRACTION=0.0
```

**Best 100-step result:** val_bpb **1.9898**, RT bpb **1.9898**, artifact **7.69MB**

---

## Complete Run Table

### Phase 0 — Ternary vs Binary (500 steps, 16L 512d, 1k vocab)

| Run | Config | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|--------|---------|--------|----------|---------|--------|-------|
| 17 | Ternary baseline | 1.7110 | 1.7300 | 23.95MB | 1312 | BASELINE | int8+zlib compression |
| 18 | Binary {-1,+1} | 1.7121 | 1.7316 | 23.93MB | 1309 | PARKED | -0.0016 worse; revisit at max scale |

**Finding:** Binary parked. Zero state provides modest but consistent benefit. Binary may regain advantage at larger scale (100M params at 1 bit/param vs 60M at 1.6 bits/param).

### Phase 1 — Training Techniques (100 steps, 16L 512d, 1k vocab)

| Run | Config | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|--------|---------|--------|----------|---------|--------|-------|
| 19 | Ternary baseline 16L 512d 1k | 2.3371 | 2.3793 | 7.33MB | 1346 | OLD BASELINE | |
| 20 | Untie lm_head at 2/3 | 2.3569 | 2.3983 | 8.13MB | — | DEFERRED | +0.019; head trains only 34 steps + 70s recompile |
| 21 | Value embeddings | — | — | — | — | BLOCKED | RTX 5090 Triton smem limit |
| 22 | Smear module | 2.3593 | 2.3985 | 7.33MB | 1339 | DEFERRED | +0.022; gate needs many steps from zero init |

### Phase 1 — Training Techniques (100 steps, 9L 512d, 1k vocab)

| Run | Config | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|--------|---------|--------|----------|---------|--------|-------|
| 23 | Baseline 9L 512d 1k | 2.4483 | 2.4768 | 4.45MB | 737 | BASELINE | Switched from 16L due to Triton |
| 24 | + Polynomial softcap | 2.3981 | 2.4438 | 4.45MB | 738 | **ACCEPTED** | **-0.033 rt** |
| 25 | + Seq length schedule | 2.4633 | 2.5106 | 4.45MB | varies | DEFERRED | Recompile cost at 100 steps |
| 26 | + NorMuon | 2.4018 | 2.4104 | 4.40MB | 743 | **ACCEPTED** | **-0.033 rt**; 5x smaller RT gap (0.046->0.009) |
| 27 | + Grad accum delay | 2.6298 | 2.6571 | 4.40MB | 741 | DEFERRED | +0.232; needs 2000+ steps |

### Vocabulary Sweep (100 steps, 9L 512d)

| Run | Vocab | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|-------|---------|--------|----------|---------|--------|-------|
| 23 | 1024 | 2.4483 | 2.4768 | 4.45MB | 737 | BASELINE | |
| 28 | 4096 | 2.0930 | 2.0974 | 6.68MB | 783 | TESTED | -0.32 vs 1k |
| 29 | 8192 | 1.9946 | 1.9990 | 9.64MB | 838 | **ACCEPTED** | **-0.42 vs 1k**; largest single win |

### Activation Sweep (100 steps, 9L 512d, 8k vocab)

| Run | Activation | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|-----------|---------|--------|----------|---------|--------|-------|
| 29 | relu2 | 1.9946 | 1.9990 | 9.64MB | 838 | BASELINE | |
| 30 | relu | 1.9846 | 1.9879 | 9.63MB | 830 | **ACCEPTED** | **-0.011**; simpler, faster |
| 31 | SwiGLU | 1.9704 | 1.9743 | 10.70MB | 960 | **ACCEPTED** | **-0.025**; +1MB, +15% slower |
| 32 | SwiGLU + MTP(2) | 1.9627 | 1.9672 | 10.69MB | 1111 | **ACCEPTED** | **-0.032**; +16% slower; training-only |

### Embedding Factorization Sweep (100 steps, 9L 512d, 8k vocab)

| Run | EMBED_DIM | val_bpb | RT bpb | RT gap | Artifact | ms/step | Status | Notes |
|-----|-----------|---------|--------|--------|----------|---------|--------|-------|
| 33a | 0 (=512) | 1.9931 | 1.9962 | 0.003 | 9.63MB | 832 | BASELINE | No factoring |
| 33b | 512 | 1.9877 | 1.9912 | 0.004 | 9.63MB | 830 | — | Seed noise vs 33a |
| 33c | 256 | 2.0538 | 2.1339 | 0.080 | 6.68MB | 741 | — | Huge RT gap (embed_proj ternarized) |
| 33d | 128 | 2.0697 | 2.0733 | 0.004 | 5.28MB | 724 | VIABLE | Clean RT; for max-depth configs |
| 33e | 64 | 2.0936 | 2.0968 | 0.003 | 4.49MB | 717 | VIABLE | Maximum budget headroom |
| 33f | 1024 | 2.0709 | 2.1845 | 0.114 | 15.60MB | 847 | REJECTED | Over budget |
| 33g | 2048 | 2.1415 | 2.3456 | 0.204 | 27.46MB | 985 | REJECTED | Way over budget |
| 33h | 4096 | 2.3123 | 2.5933 | 0.281 | 50.77MB | 1280 | REJECTED | Absurd size |

**Finding:** Full-size embedding (512) best. Factoring at 128/64 viable only if freed budget spent on extra layers.

---

## Tversky Neural Network Investigation

Based on Doumbouya et al. (2025), "Tversky Neural Networks."

### Attempt 1: Simplified Implementation (FAILED — missing contrast terms)

Runs used `relu(x @ features.t()) @ prototypes.t()`, missing three-term Tversky formula. Also misapplied to K/V projections instead of output proj.

| Run | Config | val_bpb | Status | Notes |
|-----|--------|---------|--------|-------|
| 34* | Tversky embed_proj, EMBED_DIM=0 | 1.9832 | INVALID | Tversky not activated |
| 35* | Tversky embed_proj, EMBED_DIM=128 | 2.8318 | FAILED | Missing contrast terms |
| 36* | Tversky c_v, 16 features | 2.8314 | FAILED | Rank bottleneck + wrong location |
| 37* | Tversky c_v, 64 features | 2.8320 | FAILED | Same issues |

*Runs nullified; numbering restarted from 34 with corrected implementation.*

### Attempt 2: Tversky Logit Head (FAILED — zero-gradient init)

| Run | Features | val_bpb | RT bpb | Status | Notes |
|-----|----------|---------|--------|--------|-------|
| 34 | 16 | 3.5245 | 3.5260 | REJECTED | Loss stuck at random (~9.0) |
| 35 | 128 | 3.5158 | 3.5204 | REJECTED | Same at all feature counts |
| 37 | 128 (both logit+attn) | 3.5148 | 3.5200 | REJECTED | Logit head poisons training |

**Root cause:** Random prototype init with sigmoid produces near-uniform outputs. The logit head is the sole gradient source — zero gradients mean nothing trains.

### Attempt 3: Correct Attn Output Proj, FP16 — Feature Count Sweep

Proper three-term formula: `S = theta*f(A∩B) - alpha*f(A-B) - beta*f(B-A)` with product intersection, ignorematch difference, soft sigmoid membership. Memory-efficient matmul decomposition.

| Run | Features | val_bpb | RT bpb | RT gap | Artifact | ms/step | Status |
|-----|----------|---------|--------|--------|----------|---------|--------|
| — | Baseline (no Tversky) | 1.9751 | 1.9751 | 0.000 | 5.33MB | 723 | BASELINE |
| 38 | 16 | 1.9877 | 2.0186 | 0.031 | 5.46MB | 758 | — |
| 39 | 32 | 1.9843 | 2.0133 | 0.029 | 5.57MB | 753 | — |
| 40 | 64 | 1.9790 | 2.0097 | 0.031 | 5.79MB | 755 | — |
| 41 | **128** | **1.9427** | 1.9865 | 0.044 | 6.20MB | 789 | **BEST RAW** |
| 42 | 256 | 1.9737 | 2.0863 | 0.113 | 5.63MB | 830 | — |
| 43 | 512 | 2.0036 | 2.0965 | 0.093 | 5.90MB | 900 | — |

**Finding:** 128 features optimal. 16-64 too few features for expressiveness. 256+ diminishing returns and harder to train at 100 steps.

### Roundtrip Gap Fixes

| Run | Fix | val_bpb | RT bpb | RT gap | Notes |
|-----|-----|---------|--------|--------|-------|
| 43 | fp16 precision sim (`.half().float()`) | 1.9443 | 1.9867 | 0.042 | Did not fix gap |
| 44 | **Quantization shrinkage fix** | 1.9425 | **1.9425** | **0.000** | **Eliminated ALL RT gaps** |
| 45 | 256 features + shrinkage fix | 1.9616 | 1.9616 | 0.000 | Confirms fix is universal |

**Shrinkage fix:** When Q contains zeros, `mean(|Q|) < 1.0`, so re-quantizing loaded weights produces smaller scale. Fix: inflate by `1/mean(|Q|)` during dequantization. Applies universally to all ternary models.

### Attempt 4: Full Ternary Tversky (FAILED — sigmoid saturation)

Quantized BOTH prototypes AND features to ternary via STE.

| Run | Features | val_bpb | RT bpb | vs Baseline | Status |
|-----|----------|---------|--------|-------------|--------|
| 46 | 128 | 1.9960 | — | +0.021 worse | REJECTED |
| 47 | 256 | 2.0558 | 2.0559 | +0.081 worse | REJECTED |
| 48 | 64 | 2.0244 | 2.0244 | +0.049 worse | REJECTED |

**Root cause:** Multiplying two ternary matrices creates chunky discrete dot products. `sigmoid(int * 5.0)` saturates to exactly 0.0 or 1.0, killing gradient flow through the membership indicators.

### Attempt 5: Half-Ternary (Ternary Protos + FP16 Local Features) — SUCCESS

Kept Feature Bank as continuous FP16 (preserves smooth sigmoid gradients). Only applied ternary STE to Prototypes.

| Run | Config | val_bpb | RT bpb | Artifact | Status |
|-----|--------|---------|--------|----------|--------|
| N/A | 9L Half-Ternary Local 128F | **1.9582** | **1.9582** | 6.25MB | **ACCEPTED** |
| 52 | 12L Half-Ternary Local 128F | **1.9695** | **1.9695** | 7.76MB | **ACCEPTED** |

### Feature Sharing Study

Paper recommended sharing feature banks. Empirical testing showed "semantic bottleneck."

| Run | Config | val_bpb | Artifact | Status | Notes |
|-----|--------|---------|----------|--------|-------|
| 50 | 9L 1-Pool Shared | 2.0419 | 5.50MB | REJECTED | Semantic bottleneck |
| — | 9L 3-Pool Shared | 2.0186 | 5.67MB | REJECTED | Better than 1 pool, worse than local |
| N/A | 9L Local (per-layer) | **1.9582** | 6.25MB | **ACCEPTED** | Each layer needs its own feature space |
| 54 | 12L 6-Pool Shared | 2.0083 | 7.18MB | REJECTED | Confirms local > shared at 12L |

### Layer Scaling

| Run | Config | val_bpb | Artifact | Status | Notes |
|-----|--------|---------|----------|--------|-------|
| 53 | 12L Standard Baseline | 1.9723 | 6.62MB | BASELINE | |
| 52 | 12L Half-Ternary Tversky | **1.9695** | 7.76MB | **ACCEPTED** | Tversky advantage persists at depth |

---

## Compression Experiments

### Ternary Packing (Base-3)

5 ternary values per byte (1.6 bits/trit) + best-of-3 compression (LZMA/zlib/zstd).

| Method | 9L 512d 8k |
|--------|-----------|
| int8 + zlib | 15.80MB |
| base-3 + LZMA | **9.63MB** |

39% reduction. LZMA consistently wins for ternary data.

### Split Bitmask (Run 55) — REJECTED

Hypothesis: split ternary into non_zero mask + sign mask for better LZMA patterns. Reality: zero-valued weights store meaningless "ghost" sign bits injecting random noise.

| Run | Config | Artifact | vs Base-3 | Status |
|-----|--------|----------|-----------|--------|
| 55 | 12L Split Bitmask | 8.39MB | **+630KB worse** | REJECTED |
| 56 | 12L Base-3 + no lm_head + pruned code | 7.80MB | — | **ACCEPTED** |

### Dead Weight Elimination

With `tie_embeddings=1`, the `lm_head` tensor (4.19M params of dead zeros) was still in state_dict. Intercepting and removing before serialization saved significant space.

### FP Storage Compression Sweep

Storage-only change: train in fp16, compress to fp8/fp4 for artifact, upcast on load.

| Run | Storage | val_bpb | RT bpb | RT gap | Artifact | Saved | Status |
|-----|---------|---------|--------|--------|----------|-------|--------|
| 58 | fp16 | 1.9955 | 1.9955 | 0.000 | 7.74MB | — | **DEFAULT** |
| 61 | fp8 | 1.9948 | 2.0033 | 0.0085 | 6.38MB | 1.36MB | CONFIGURABLE |
| 62 | fp8 + training sim | 2.2233 | 2.2329 | — | 6.31MB | — | REJECTED |
| 63 | fp4 | 1.9959 | 2.0327 | 0.037 | 5.67MB | 2.07MB | CONFIGURABLE |

**Finding:** FP8 training simulation destroys embedding quality (3-bit mantissa too coarse for 8192-token embedding). FP8/FP4 storage-only viable as budget escape valves.

---

## Depth Recurrence

### Training Recurrence

| Run | Config | val_bpb | RT bpb | ms/step | Status | Notes |
|-----|--------|---------|--------|---------|--------|-------|
| 57 | 12L x2 recurrence (half batch) | 2.0685 | 2.0685 | 1024 | DEFERRED | OOM full batch on 5090 |
| 58 | 12L x1 baseline (half batch) | 1.9955 | 1.9955 | 643 | REFERENCE | |

### Test-Time Compute Scaling (Eval-Only Recurrence)

Train with `DEPTH_RECURRENCE=1`, set higher recurrence only for eval. Zero training cost.

| Run | Eval Recurrence | RT bpb | Delta | Status | Notes |
|-----|----------------|--------|-------|--------|-------|
| 58 | 1 (baseline) | 2.5244 | — | — | 10-step comparison |
| 59 | 2 | 2.5240 | **-0.0004** | **ACCEPTED** | Free improvement |
| 60 | 3 | 2.5243 | -0.0001 | — | Worse than 2; diminishing returns |

---

## Hyperparameter Sweeps (100 steps, 12L 512d, 8k vocab)

All sweeps use the accepted stack (SwiGLU, Tversky 128, half-ternary local, shrinkage fix).

### MLP Width Sweep

| Run | MLP_MULT | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|----------|---------|--------|----------|---------|--------|-------|
| 58 | **2** | **1.9955** | **1.9955** | 7.75MB | 643 | **KEEP** | Best bpb/budget ratio |
| 64 | 3 | 1.9972 | 1.9972 | 9.08MB | 755 | REJECTED | +0.0017; +1.3MB wasted |
| 65 | 4 | 1.9992 | 1.9992 | 10.39MB | 737 | REJECTED | +0.0037; +2.6MB wasted |

**Finding:** Extra MLP width is undertrained at 100 steps AND wastes budget. The 6:1 MLP-to-attention ratio from SwiGLU at MLP_MULT=2 is already sufficient.

### KV Heads Sweep

| Run | KV_HEADS | val_bpb | RT bpb | Artifact | ms/step | Status | Notes |
|-----|----------|---------|--------|----------|---------|--------|-------|
| 58 | **4 (GQA)** | **1.9955** | **1.9955** | 7.75MB | 643 | **KEEP** | |
| 66 | 8 (MHA) | 2.0148 | 2.0147 | 8.46MB | 649 | REJECTED | +0.019; +0.7MB |

**Finding:** Full MHA wastes parameters at 512d. 4 KV heads provide sufficient attention diversity.

### ROPE_BASE Sweep

| Run | ROPE_BASE | val_bpb | RT bpb | Delta vs 10k | Status | Notes |
|-----|-----------|---------|--------|-------------|--------|-------|
| 70 | 5000 | 1.9959 | 1.9959 | +0.0004 | — | |
| 73 | **10000** | **1.9931** | **1.9931** | — | **KEEP** | Best at 100 steps |
| 69 | 20000 | 2.0008 | 2.0009 | +0.0053 | DEFERRED | Retest at 10k steps |
| 68 | 50000 | 2.0017 | 2.0017 | +0.0062 | DEFERRED | Retest at 10k steps |

**Finding:** Higher rope base helps long-range dependencies at convergence but hurts at 100 steps where the model barely sees distant positions. Retest at longer training.

### LOGIT_SOFTCAP Sweep

| Run | SOFTCAP | val_bpb | RT bpb | Delta vs 30 | Status | Notes |
|-----|---------|---------|--------|-------------|--------|-------|
| 74 | 5 | 1.9942 | 1.9942 | -0.0013 | — | Too restrictive |
| 73 | **10** | **1.9931** | **1.9931** | **-0.0024** | **ACCEPTED** | Sweet spot |
| 72 | 20 | 1.9935 | 1.9935 | -0.0020 | — | Close second |
| 58 | 30 (old default) | 1.9955 | 1.9955 | — | OLD DEFAULT | |
| 71 | 50 | 1.9957 | 1.9958 | +0.0003 | — | Nearly linear; no benefit |

**Finding:** SOFTCAP=10 is the sweet spot. Tighter cap provides better gradient signal for the polynomial approximation. 5 is too restrictive, 50 is too loose.

### QK_GAIN_INIT Sweep

| Run | QK_GAIN | val_bpb | RT bpb | Delta vs 1.5 | Status | Notes |
|-----|---------|---------|--------|-------------|--------|-------|
| 75 | 1.0 | 2.0007 | 2.0007 | +0.0076 | REJECTED | Too soft; attention too diffuse |
| 73 | 1.5 (old default) | 1.9931 | 1.9931 | — | OLD DEFAULT | |
| 76 | 2.0 | 1.9936 | 1.9936 | +0.0005 | — | Essentially tied with 1.5 |
| 81 | 2.15 | 1.9913 | 1.9913 | -0.0018 | — | Good, not peak |
| 79 | **2.25** | **1.9898** | **1.9898** | **-0.0033** | **ACCEPTED** | **New best; sharper attention** |
| 77 | 2.5 | 1.9915 | 1.9915 | -0.0016 | — | Past peak |
| 80 | 2.75 | 1.9975 | 1.9975 | +0.0044 | — | Overshot |
| 78 | 3.0 | 2.0011 | 2.0011 | +0.0080 | REJECTED | Way too sharp |

**Finding:** Clear inverted-U response curve. 2.25 is the peak — sharper attention than default helps the model focus, but 2.75+ overshoots and destabilizes. The -0.0033 improvement is zero-cost.

---

## Cumulative Accepted Stack

| Technique | BPB Impact | Size Impact | Speed Impact |
|-----------|-----------|-------------|-------------|
| Polynomial softcap | -0.033 | 0 | 0 |
| NorMuon | -0.033 (RT) | -50KB | 0 |
| 8k vocabulary | -0.42 | +5.2MB | +12% |
| relu (over relu2) | -0.011 | 0 | -1% |
| SwiGLU (configurable) | -0.014 | +1MB | +15% |
| MTP (configurable) | -0.007 | 0 (training-only) | +16% |
| Quantization shrinkage fix | eliminates RT gap | 0 | 0 |
| Half-Ternary Tversky attn proj | -0.017 | +0.9MB | +10% |
| 12L depth | -0.003 | +1.5MB | proportional |
| Dead lm_head removal | 0 | saves space | 0 |
| LOGIT_SOFTCAP=10 | -0.0024 | 0 | 0 |
| QK_GAIN_INIT=2.25 | -0.0033 | 0 | 0 |
| EVAL_DEPTH_RECURRENCE=2 | -0.0004 | 0 | eval-only |

---

## Rejected Techniques

| Technique | Reason |
|-----------|--------|
| Full ternary Tversky | Sigmoid saturation kills gradients |
| Tversky logit head | Random init -> zero gradients; model doesn't train |
| Shared Tversky feature banks | Semantic bottleneck across layers |
| Split bitmask compression | Ghost sign entropy leak (+630KB) |
| FP8 training simulation | Too aggressive; destroys embedding quality |
| MLP_MULT=3,4 | Worse bpb despite more params; budget wasted |
| NUM_KV_HEADS=8 (MHA) | +0.019 worse than GQA; +0.7MB |
| QK_GAIN_INIT=1.0 | +0.0076 worse; attention too soft |
| QK_GAIN_INIT=3.0 | +0.008 worse; attention too sharp |

## Configurable Features

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| ACTIVATION | swiglu, relu, relu2 | swiglu | MLP activation |
| MTP_HEADS | 0, 2 | 0 | Multi-token prediction (training-only) |
| EMBED_DIM | 0, 64, 128, 256 | 0 | Factored embedding (0=model_dim) |
| UNTIE_AT_FRACTION | 0.0-1.0 | 0.0 | Fraction to untie lm_head |
| SLIDING_EVAL | 0, 1 | 0 | Sliding window eval |
| ATTN_PROJ_TYPE | standard, tversky | tversky | Attn output projection |
| TVERSKY_NUM_FEATURES | int | 128 | Feature bank size |
| TVERSKY_FEATURE_POOLS | int | 0 | 0=local, N=shared pools |
| TVERSKY_TERNARY_PROTOS | 0, 1 | 1 | Ternary quantize prototypes |
| FP_STORAGE | 0, fp8, fp4 | 0 | Non-ternary storage precision |
| EVAL_DEPTH_RECURRENCE | int | 2 | Block iterations at eval time |
| DEPTH_RECURRENCE | int | 1 | Block iterations during training |
| LOGIT_SOFTCAP | float | 10 | Logit capping strength |
| QK_GAIN_INIT | float | 2.25 | Initial attention sharpness |
| BITNET_GROUP_SIZE | int | 64 | Ternary quantization group size |
| SEED | int | 1337 | Random seed |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-03-20 | Park binary | -0.0016 worse at equal config |
| 2025-03-20 | Base-3 + LZMA compression | 39% reduction over int8+zlib |
| 2025-03-20 | Polynomial softcap | -0.033, zero cost |
| 2025-03-20 | NorMuon | 5x smaller RT gap |
| 2025-03-20 | relu over relu2 | -0.011, faster |
| 2025-03-20 | SwiGLU configurable | -0.014 but +15% slower |
| 2025-03-20 | MTP configurable | -0.007 but +16% slower |
| 2025-03-20 | 8k vocabulary | -0.42, largest win |
| 2025-03-20 | Shrinkage fix | Eliminates ALL RT gaps universally |
| 2025-03-20 | Half-ternary Tversky | -0.017, scales to 12L |
| 2025-03-20 | Local per-layer features | Beats all sharing schemes |
| 2025-03-20 | Remove dead lm_head | Free space with tied embeddings |
| 2025-03-20 | LOGIT_SOFTCAP=10 | -0.0024, zero cost |
| 2025-03-20 | QK_GAIN_INIT=2.25 | **-0.0033**, zero cost; sharper attention optimal |
| 2025-03-20 | EVAL_DEPTH_RECURRENCE=2 | -0.0004, free (eval-only) |
| 2025-03-20 | Keep MLP_MULT=2 | 3 and 4 worse despite more params |
| 2025-03-20 | Keep GQA (4 KV heads) | MHA worse at 512d |
| 2025-03-20 | Keep ROPE_BASE=10000 | All alternatives worse at 100 steps; retest at 10k |
| 2025-03-20 | FP8/FP4 storage configurable | Trade RT gap for artifact space |

---

## Pending Experiments

### Immediate — Layer Scaling
- [ ] 16L 512d (fill budget with depth)
- [ ] 20L 512d
- [ ] 24L 512d (may need FP8/factored embedding)

### Final Assembly (H100 10k steps)
- [ ] Scale to max layers under 16MB
- [ ] Enable: UNTIE_AT_FRACTION=0.66, MTP_HEADS=2
- [ ] Retest: ROPE_BASE=50000 (for 8k context at convergence)
- [ ] Retest: LOGIT_SOFTCAP=10 vs 20 at convergence
- [ ] Depth recurrence (no OOM on H100)
- [ ] Value embeddings, residual decay (no Triton limit on H100)
- [ ] Sequence length schedule (amortized recompile)
- [ ] Binary vs ternary at max params
- [ ] TTT with LoRA during eval (causal, no leakage)
- [ ] FP8 storage if budget tight
- [ ] Final 8xH100 10-minute submission
