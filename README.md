# Parameter Golf — Brainquiver Experiments

Private experiment tracking for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

Target: beat the current SOTA bpb on FineWeb val set, under 16MB artifact, 10min on 8xH100.

---

## Local Reference Setup

- Hardware: Apple Mac Studio M1 Max (MLX)
- Batch: 16,384 tokens, grad_accum=2, seq_len=1024
- Val: 500,000 tokens (capped for speed)
- Steps: 1,000 (local) - time-limited (H100)

---

## Results

| # | Run | val_bpb (1k steps) | Artifact | Changes vs prev | Notes |
|---|-----|--------------------|----------|-----------------|-------|
| 1 | Baseline | 1.7288 | 14.1MB | — | OpenAI naive baseline, exact hyperparams |
| 2 | Tuned Optimizer | 1.8601 | 15.0MB | Muon momentum 0.99, LR halved, longer warmup/warmdown | Worse at 1k steps — likely better at 13k steps on H100 |

---

## Planned Experiments

| # | Change | Status |
|---|--------|--------|
| 2 | Tuned Optimizer | pending |
| 3 | NorMuon | pending |
| 4 | Polynomial softcap | pending |
| 5 | Sequence length schedule | pending |
| 6 | Untie lm_head at 2/3 | pending |
| 7 | Smear | pending |
| 8 | Value embeddings | pending |
| 9 | Sliding window eval | pending |
| 10 | Document-isolated eval | pending |

---

## Shell Config

Base shell script used for all 1k-step local runs:

```bash
RUN_ID=run_name \
TRAIN_BATCH_TOKENS=16384 \
GRAD_ACCUM_STEPS=2 \
TRAIN_SEQ_LEN=1024 \
ITERATIONS=1000 \
WARMDOWN_ITERS=200 \
WARMUP_STEPS=20 \
VAL_LOSS_EVERY=500 \
VAL_BATCH_SIZE=16384 \
VAL_MAX_TOKENS=500000 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
TRAIN_LOG_EVERY=1 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_mlx.py
```

Optimizer defaults (baseline):
- MATRIX_LR=0.04, SCALAR_LR=0.04, TIED_EMBED_LR=0.05
- MUON_MOMENTUM=0.95, MUON_MOMENTUM_WARMUP_START=0.85
- MUON_MOMENTUM_WARMUP_STEPS=500, WARMDOWN_ITERS=1200