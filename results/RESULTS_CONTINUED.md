# Parameter Golf — Continued Experiment Log

**Author:** Ciprian-Florin Ifrim
**Date:** March 2026 (post-submission continued research)

---

## Overview

This document continues from RESULTS.md and covers post-submission research conducted after the initial P-series submission (sliding bpb 1.1570, 10L 768d relu² 4×MLP fp8, EMBED_DIM=254). The work focused on four areas: infrastructure improvements for interruptible compute, roundtrip error analysis and serialization improvements, scale estimator research (absmean vs RMS vs non-zero absmean), and embedding dimension scaling to improve quality within budget.

**Final result of this research period:**

| Config | Sliding bpb | val_bpb | RT gap | Artifact | Seed std |
|--------|-------------|---------|--------|----------|----------|
| Original submission (P-series) | 1.1570 | 1.1821 | 0.0021 | 15.92MB | 0.0007 |
| **New config (10L EMBED=312 RMS)** | **1.1539** | **1.1803** | **0.0011** | **15.88MB** | **0.0002** |

Improvement of **0.0031 bpb** on sliding evaluation, with tighter roundtrip, smaller artifact, and higher cross-seed reproducibility.

---

## 1. Checkpointing for Interruptible Compute

The training pods used for H100 compute are interruptible — the instance can be preempted at any point, but the attached HDD storage persists across restarts. Without checkpointing, an interrupted run loses all training progress and must restart from scratch.

Checkpointing was implemented as a periodic save of the full training state: model weights (float32), all optimizer states (Muon momentum buffers, Adam first/second moments), CPU and CUDA RNG states, elapsed training time, step counter, and schedule flags (_seq_switched, _batch_switched, _untied). Saves are atomic — the payload is written to a `.tmp` file then renamed, so a mid-write preemption never produces a corrupt checkpoint.

On resume, the script auto-discovers the latest checkpoint by lexicographic sort on step-numbered filenames, restores all state, and logs `checkpoint found: <path>` then `checkpoint loaded, starting from step X (accumulated_train_time:Xms)`. If no checkpoint exists it logs `no checkpoint found, starting from scratch` and proceeds normally.

**Wallclock continuity:** `training_time_ms` resumes from the checkpoint value rather than resetting to zero, so the wallclock cap and warmdown schedule are measured against total accumulated time across all restarts, not just the current session. This is critical for the submission config where the exact 599s budget matters.

**Budget exhaustion guards:** If the checkpoint's accumulated time already exceeds the wallclock cap, or the checkpoint step already meets or exceeds `ITERATIONS`, the script logs a skip message and proceeds directly to serialization without entering the training loop. This handles the common case where a pod restarts after training has already completed.

A subtle fix was required for the per-step average log line: the original `approx_ms / step` formula gives nonsense when resuming mid-run (total elapsed divided by global step count). A `steps_this_session` counter was added that divides only the current session's wall time.

The binary training script received the same checkpointing with one addition: EMA model state (`ema_model`, `_ema_started`, `_ema_steps`) is saved and restored alongside everything else, so EMA decay continues from exactly where it left off.

---

## 2. Roundtrip Error Investigation

### 2.1 Observed Problem

After extended training runs (150k steps for the binary model, ~50k for longer ternary experiments), the roundtrip error — the gap between `val_bpb` before and after ternary quantization and reload — grew significantly beyond the ~0.002 bpb seen at the 6,530-step submission config. At 150k ternary steps, the gap reached 0.039 bpb.

The binary model at 50k steps (2.15 hours) showed a much smaller gap of 0.0019 bpb under identical FP8 storage settings. This asymmetry pointed to a ternary-specific mechanism rather than FP8 storage quality.

### 2.2 Root Cause: Shrinkage Correction Amplification

The dequantization formula applies a shrinkage correction to compensate for zeros reducing the group mean:

```
reconstructed = q * (scale / q_absmean)
```

where `q_absmean = mean(|Q|)` over the group of {-1, 0, +1} values. For a group with zero fraction `z`, `q_absmean = 1 - z`, making the correction factor `1/(1-z)`.

The stored scale is the only external input to this formula. In the original code, scale was stored as FP16 (`scale.half().squeeze(-1)`), introducing a fixed relative rounding error of approximately `scale × 2^-11` per group. This error is then multiplied by the correction factor `1/(1-z)` on load.

As training progresses, ternary models with Muon optimisation push more weights toward zero — zero fraction rises from ~0.25 at early training to ~0.34 at the 6,530-step submission point and potentially higher at 50k+ steps. The correction factor grows correspondingly:

| zero_frac | AbsMean correction 1/(1-z) |
|-----------|---------------------------|
| 0.25 | 1.33× |
| 0.30 | 1.43× |
| 0.35 | 1.54× |
| 0.40 | 1.67× |
| 0.50 | 2.00× |

Binary weights are always {-1, +1} with no zeros, so `q_absmean = 1.0` always, making the correction factor exactly 1.0. This explains the asymmetry — binary has no amplification mechanism regardless of training length.

### 2.3 Fix: BF16 Scale Storage

FP16 uses a 5-bit exponent (range ±65504) and 10-bit mantissa. BF16 uses an 8-bit exponent (range identical to FP32, ±3.4×10^38) and 7-bit mantissa. For scale values — which are positive absmean magnitudes that can span several orders of magnitude across different weight matrices — the exponent range matters more than mantissa precision. A scale value of 0.001 and a scale value of 10.0 both need accurate magnitude representation; FP16's limited exponent range introduces systematic rounding in the mid-range values that are most common.

The fix changes scale storage from `.half()` to `.bfloat16()` everywhere in the serialization path. This costs zero bytes (both formats are 2 bytes per value) and zero budget impact. The dequantization path already calls `.float()` on load, which handles both FP16 and BF16 inputs identically, preserving backward compatibility with existing checkpoints.

The same fix was applied to the FP4 quantization path (`quantize_to_int4`) where scale had also been stored as FP16, and to the fallback serialization branch for small parameters (gains, biases, skip weights), which was changed from `t.half()` to `t.bfloat16()`.

### 2.4 Benchmark Validation

A microbenchmark notebook was developed to measure roundtrip error empirically across all schemes. The benchmark trains a small ternary model (dim=256, mlp_mult=4, 2 blocks, ~1.3M ternary params) for up to 50k steps and evaluates roundtrip error at checkpoints matching real training milestones. This was necessary because an earlier version of the benchmark used synthetic weights with a broken zero_frac generator — the actual zero_frac was 0.309 at every target level — producing meaningless error comparisons. The model-based approach lets zero_frac emerge from real training dynamics.

Key finding: at the zero_frac levels observed in the submission config (0.24–0.34), the difference between FP16 and BF16 scales is not detectable in mean absolute error on a per-matrix basis. The amplification effect only becomes visible in aggregate across all layers and through the complete model's output distribution. BF16 is the correct choice for long runs but the submission config was not showing material error even with FP16 — the original 0.039 bpb gap was from a 150k-step run, far outside the submission regime.

---

## 3. Scale Estimator Research: AbsMean vs RMS vs Non-Zero AbsMean

### 3.1 Motivation

The absmean scale estimator used throughout training and serialization is borrowed from the original BitNet b1.58 paper. It was chosen for hardware efficiency but has a known limitation: the correction factor `1/(1-z)` grows without bound as zero fraction increases. Two alternative estimators were investigated:

**RMS (Root Mean Square):** `scale = sqrt(mean(w²))`. Theoretically optimal under a squared-error reconstruction objective. The correction factor at load time is `scale / q_rms = 1/sqrt(1-z)`, which grows as the square root of the absmean factor — at zero_frac=0.50, absmean amplifies FP16 error 2×, RMS only 1.41×.

**Non-zero AbsMean:** Computes absmean over only the non-zero weights after quantization. Since the non-zero subset of {-1, +1} always has absmean = 1.0, the correction formula collapses to `q * scale` — zero amplification regardless of zero fraction.

### 3.2 Speed Measurements

From the microbenchmark at real submission matrix sizes (40 STE calls per step, 10L model):

| Estimator | ms/call (mlp_fc) | Per-step overhead vs absmean | Steps lost in 599s |
|-----------|-----------------|-----------------------------|--------------------|
| AbsMean | 0.021ms | 0 | 0 |
| RMS | 0.026ms | +0.19ms | ~13 |
| Non-zero AbsMean | 0.053ms | +1.29ms | ~91 |

RMS costs 0.21% of the step budget. Non-zero absmean costs 1.40% — approximately 2× slower due to the additional mask and conditional summation operations that don't fuse as cleanly under torch.compile.

### 3.3 RMS Training Experiment

RMS was applied to all three locations simultaneously — training-time STE in `TernaryLinear.forward`, serialization-time scale in `q_sd`, and load-time correction in `deq_sd`. Applying it to only serialization and load but not training would create an inconsistency: the model trains with absmean-calibrated weights but the stored scale is RMS-calibrated, requiring the correction to bridge two different conventions and introducing error rather than reducing it.

**10-minute comparison (seed 42):**

| Config | Steps | val_bpb | RT gap | Artifact | zero_frac |
|--------|-------|---------|--------|----------|-----------|
| AbsMean (original) | 6,530 | 1.1816 | 0.0021 | 15.92MB | 0.336 |
| RMS | 6,580 | 1.1863 | 0.0012 | 15.41MB | 0.236 |

The RMS model shows a 0.0047 bpb regression in val_bpb despite completing 50 more steps (step_avg 91.3ms vs 91.5ms — RMS is marginally faster, not slower as predicted by the microbenchmark, likely due to torch.compile fusing the pow(2)/mean/sqrt sequence efficiently). The RT gap is tighter (0.0012 vs 0.0021) and the artifact is 510KB smaller.

The critical observation is the zero_frac divergence: 0.236 for RMS vs 0.336 for absmean at equivalent step counts. RMS produces a higher quantization threshold (RMS ≥ absmean by Jensen's inequality), which rounds more weights to zero earlier in training. This changes the training dynamics — the model operates with a different sparsity regime than it was implicitly tuned for. The lower zero_frac under RMS compresses better (explaining the smaller artifact) but suggests the model is representing information differently, and the 0.0047 bpb gap reflects this.

**Conclusion:** RMS is theoretically superior for long runs where zero_frac grows high enough to create material amplification error. For the submission config (599s, zero_frac ~0.34), the training dynamic change dominates the serialization improvement, and absmean remains the better practical choice. RMS is recommended for future models trained from scratch with RMS throughout, not as a drop-in replacement for absmean-trained models.

### 3.4 Non-Zero AbsMean

Not tested at full training scale. The 1.40% step cost and 2× speed penalty make it unattractive relative to RMS, and the benchmark showed it is structurally worse in mean reconstruction error at early zero_frac levels (1.05–1.08× worse than absmean at zero_frac=0.25). The non-zero scale overstates the true group magnitude because it ignores the contribution of zeroed weights to the reconstruction, biasing the scale upward. This is a fundamental property of the estimator, not a tuning issue.

---

## 4. Embedding Dimension Scaling

### 4.1 Motivation

The original submission used EMBED_DIM=254, chosen to fit the artifact within budget while providing a meaningful embedding bottleneck (128 was optimal at dev scale; 256 improved quality at convergence; 254 trimmed to fit). After RMS serialization reduced the ternary blob significantly (15.92MB → 15.41MB, freeing 510KB), there was headroom to explore larger embedding dimensions without exceeding the 16MB budget.

The embedding matrices (`tok_emb` at 8192×EMBED_DIM and the projection pair) are stored in FP8 rather than ternary. Increasing EMBED_DIM increases the fp blob, not the ternary blob. This means the tradeoff is directly: richer token representations at the cost of fp storage.

### 4.2 Embedding Dimension Sweep

All runs: 10L 768d relu² 4×MLP fp8, RMS scales, seed=42, 599s wallclock.

| EMBED_DIM | val_bpb | RT gap | Artifact | Steps | Budget |
|-----------|---------|--------|----------|-------|--------|
| 254 (original absmean) | 1.1816 | 0.0021 | 15.92MB | 6,530 | FITS |
| 254 (RMS) | 1.1863 | 0.0012 | 15.41MB | 6,580 | FITS |
| 312 | 1.1802 | 0.0009 | 15.95MB | 6,530 | FITS |
| 320 | 1.1812 | 0.0018 | 16.02MB | 6,550 | OVER |
| 328 | — | — | — | — | — |
| 336 | — | — | — | — | — |
| 344 | 1.1758 | 0.0005 | 16.21MB | 6,520 | OVER |
| 352 | 1.1804 | 0.0017 | 16.03MB | 6,500 | OVER |
| 384 | 1.1754 | 0.0007 | 16.27MB | 6,500 | OVER |

The quality improvement is clear and monotonic up to EMBED_DIM=384 — larger embedding dimensions consistently improve val_bpb and tighten the RT gap. The RT gap improvement is particularly notable: at EMBED_DIM=344 the gap is only 0.0005 bpb, essentially zero. This is consistent with the theory that richer per-token representations reduce the sensitivity of the model output to the small quantization errors introduced by the ternary roundtrip.

The budget constraint is driven entirely by the fp blob growing with EMBED_DIM: each unit increase in EMBED_DIM adds `8192 × 1 = 8,192` FP8 bytes to `tok_emb`, plus two smaller projection matrices. After LZMA compression this is approximately 55-60KB per 8-unit step. EMBED_DIM=312 (divisible by 8, head_dim compatible) lands at 15.95MB with 48KB headroom — the largest embedding that fits within the 16MB budget.

Note: EMBED_DIM values not divisible by 8 were not tested. The embedding projection matrices feed into MODEL_DIM=768 operations that are optimized for 8/16-aligned dimensions on H100 tensor cores; misaligned dimensions would introduce padding overhead and potentially slower kernels.

### 4.3 Layer Count Experiments

The 510KB artifact saving from RMS also motivated attempting 11 and 12 layer configs, since each additional layer adds approximately 6.5M ternary params (~550KB compressed at the observed compression ratio).

**11-layer experiments:**

| Config | val_bpb | RT gap | Artifact | Steps | Notes |
|--------|---------|--------|----------|-------|-------|
| 11L kv=4 group=128 | 1.1800 | 0.0013 | 16.84MB | 6,000 | 840KB over |
| 11L kv=4 group=256 | 1.1801 | 0.0010 | 16.52MB | 6,020 | 520KB over |
| 11L kv=2 group=128 | 1.1847 | 0.0012 | 16.18MB | 6,300 | 182KB over |
| 11L kv=2 group=256 | 1.1850 | 0.0013 | 15.93MB | 6,300 | **FITS** |

Every 11L configuration that fits budget (kv=2, group=256) is worse than the 10L original. The extra layer adds capacity but simultaneously: reduces steps per run (step_avg increases from 91.8ms to ~95ms), degrades attention quality (halving KV heads from 4 to 2), and coarsens quantization (doubling group size from 128 to 256). The 11L result of 1.1850 vs the 10L original of 1.1816 confirms that the additional depth does not compensate for these three simultaneous penalties.

12-layer experiments were attempted with various embedding strategies (ternary embeddings, low-rank factorization) but the ternary blob alone at 12L (~12.5-16.6MB depending on embedding strategy) leaves insufficient budget for fp parameters and code overhead. No 12L configuration achieved both budget compliance and quality improvement.

### 4.4 Low-Rank Embedding Factorization

To decouple embedding quality from FP8 storage size, a low-rank factorized embedding was implemented and tested. Instead of storing a full `vocab_size × EMBED_DIM` matrix in FP8, the approach stores `vocab_size × R` (the lookup table) and `R × EMBED_DIM` (a learned projection), where R << EMBED_DIM. At rank R=64 with EMBED_DIM=512, storage drops from 4.2MB to ~560KB — a 7.5× reduction.

The output head reconstruction with tied embeddings requires computing `W_full = W_rank @ W_proj.T` at inference time rather than using `tok_emb.weight` directly.

Testing at 12L with R=64, EMBED_DIM=512 produced val_bpb=1.2672 — substantially worse than any 10L result. The rank-64 bottleneck was too tight: 64 dimensions for 8192 tokens provides insufficient representational capacity, and the ternary blob at 12L still exceeded budget (17.12MB). At R=128 or higher the storage savings diminish while the quality cost remains. The approach is theoretically sound but the competition budget does not provide enough room to benefit from it.

### 4.5 Ternary Embedding Experiment

The FP8 exclusion of `tok_emb` from ternary quantization was briefly lifted to test whether treating the embedding as ternary could free sufficient budget for 12L. At EMBED_DIM=512 with full ternary embedding (4.19M params → ~880KB compressed), training proceeded normally and val_bpb reached 1.1698 — the best single-seed result of all experiments. However, the roundtrip gap was catastrophic: 1.1698 → 1.3820, a gap of 0.1122 bpb.

The cause is the lookup table semantics of embeddings. Each token's row is only 512 values split into 4 groups of 128, giving very coarse scale granularity per token. The shrinkage correction amplifies errors severely because individual token rows have highly non-uniform weight magnitudes (different tokens occupy very different regions of the embedding space). The original BitNet b1.58 paper explicitly excludes embeddings from ternary quantization for this reason. The result confirmed the design decision but also confirmed that 12L with a richer embedding would outperform 10L if the serialization problem could be solved — which it cannot without either a different compression scheme or a much larger per-group scale resolution.

---

## 5. Final Submission Configuration

**Config: 10L 768d relu² 4×MLP fp8, RMS scales, BF16 storage, EMBED_DIM=312**

```bash
NUM_LAYERS=10           MODEL_DIM=768          NUM_HEADS=8
NUM_KV_HEADS=4          MLP_MULT=4             VOCAB_SIZE=8192
ACTIVATION=relu2        LOGIT_SOFTCAP=10       SOFTCAP_TYPE=poly
QK_GAIN_INIT=2.25       ROPE_BASE=5000         ROPE_TYPE=yarn
YARN_MAX_LEN=2048       EMBED_DIM=312          TIE_EMBEDDINGS=1
BITNET_GROUP_SIZE=128   FP_STORAGE=FP8         MUON_WD=0.0
MATRIX_LR=0.04          SCALAR_LR=0.02         TIED_EMBED_LR=0.02
MUON_BACKEND_STEPS=3    MUON_MOMENTUM=0.95     WARMDOWN_FRACTION=0.2
MAX_WALLCLOCK_SECONDS=599
SLIDING_EVAL=1          SLIDING_EVAL_STRIDE=16 TEMP_SCALING=1
TRAIN_BATCH_TOKENS=524288
```

**3-Seed Validation:**

| Seed | Steps | val_bpb | RT bpb | RT gap | Sliding bpb | Artifact |
|------|-------|---------|--------|--------|-------------|----------|
| 42 | 6,540 | 1.1805 | 1.1824 | 0.0019 | 1.1542 | 15.88MB |
| 1337 | 6,530 | 1.1803 | 1.1811 | 0.0008 | 1.1540 | 15.88MB |
| 7 | 6,530 | 1.1802 | 1.1808 | 0.0006 | 1.1535 | 15.87MB |
| **Mean** | **6,533** | **1.1803** | **1.1814** | **0.0011** | **1.1539** | **15.88MB** |
| **Std** | **5** | **0.0002** | **0.0008** | | **0.0004** | |

**Comparison vs original P-series submission:**

| Metric | Original | New | Delta |
|--------|----------|-----|-------|
| val_bpb | 1.1821 | 1.1803 | −0.0018 |
| RT bpb | 1.1842 | 1.1814 | −0.0028 |
| Sliding bpb | 1.1570 | 1.1539 | **−0.0031** |
| RT gap | 0.0021 | 0.0011 | −0.0010 |
| Artifact | 15.99MB | 15.88MB | −110KB |
| Seed std (sliding) | 0.0007 | 0.0004 | more stable |
| Params | 73.7M | 74.3M | +0.6M |

The larger EMBED_DIM (312 vs 254) adds 0.6M parameters exclusively in the fp8 embedding path, not in the ternary model. The seed standard deviation improvement (0.0007 → 0.0004 bpb) indicates that richer token representations reduce the sensitivity of the model to random initialisation, which is consistent with the embedding bottleneck at 254 dimensions being a source of training instability.

---

## 6. Serialization Improvements Summary

The following serialization changes were made relative to the original P-series code. All are backward-compatible — existing checkpoints and artifacts with the old fp16 storage load correctly.

| Change | Location | Rationale |
|--------|----------|-----------|
| Scale storage: FP16 → BF16 | `q_sd` ternary branch | BF16 exponent range eliminates magnitude rounding that gets amplified by shrinkage correction at high zero_frac |
| Scale computation: `.half().float()` → pure FP32 | `q_sd` ternary branch | Removes the round-trip through FP16 that biased the computed scale |
| Fallback storage: `fp16` → `bf16` | `q_sd` else branch | Consistent precision for small parameters (gains, biases, skip weights) |
| FP4 scale storage: FP16 → BF16 | `quantize_to_int4` | Same magnitude rounding issue; dormant unless FP_STORAGE=FP4 |
| RMS scale estimator (optional) | `TernaryLinear`, `q_sd`, `deq_sd` | Theoretically superior correction stability at high zero_frac; disabled for submission due to training dynamic changes |

---

## 7. Rejected Directions

**MLP_MULT=3 with 11L:** The ternary blob at 11L with 3x MLP still exceeded budget. The parameter reduction per layer was insufficient to offset the additional layer. val_bpb not tested.

**group_size=256 with 10L:** At dev scale, coarsening the group size from 128 to 256 saved ~550KB compressed with no detectable val_bpb cost at 150 steps. At full convergence this held — the quality cost was within noise. However the budget saving was insufficient to make any otherwise-failing config fit, making it a neutral option.

**NUM_KV_HEADS=2 on 10L:** Not tested in isolation. All kv=2 experiments were done at 11L where the step improvement from reduced attention cost was needed. The 0.003 bpb quality cost observed at 11L kv=2 vs 11L kv=4 suggests kv=2 on 10L would likely regress the submission by a similar margin.

**Non-zero absmean scale:** 92% slower than absmean per STE call, structurally worse in reconstruction error at submission-range zero_frac levels. Not viable.

**Ternary embeddings:** Large roundtrip error due to per-token scale granularity. The quality gain (val_bpb 1.1698 is excellent) is entirely negated by the serialization error. Would require a fundamentally different per-token scale scheme to be viable.

**Low-rank embedding factorization:** Correct approach theoretically but the rank needed for competitive quality (R≥128) provides insufficient budget savings to justify the architectural complexity at 10L. More promising for future work where embedding quality is the primary bottleneck.
