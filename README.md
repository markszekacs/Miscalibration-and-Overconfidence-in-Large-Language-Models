# Multi-Model LLM Calibration Pipeline — 6 Models

Extension of the single-model Bayesian calibration paper.

## Models

| Model | Provider | Logprobs | Output/M | Est. cost (817 q) |
|---|---|---|---|---|
| gpt-4.1-nano | OpenAI | ✅ | $0.40 | ~$0.03 |
| gpt-4o-mini | OpenAI | ✅ | $0.60 | ~$0.05 (already done) |
| gpt-4.1-mini | OpenAI | ✅ | $1.60 | ~$0.13 |
| gpt-4.1 | OpenAI | ✅ | $8.00 | ~$0.65 |
| claude-haiku-4-5 | Anthropic | ❌ | $5.00 | ~$0.41 |
| claude-sonnet-4-6 | Anthropic | ❌ | $15.00 | ~$1.23 |
| **Judge** (gpt-4o-mini × 5 new models) | OpenAI | — | $0.60 | ~$0.25 |
| **Total** | | | | **~$2.75** |

---

## Pipeline

```
TruthfulQA.csv
      │
      ├─ run_pipeline.py   → results_multimodel.csv   (4 OpenAI models)
      ├─ run_claude.py     → results_multimodel.csv   (2 Claude models, appended)
      │
      ▼
results_multimodel.csv
      │
      ├─ judge.py          → results_judged.csv
      │    (merges original manual labels for gpt-4o-mini automatically)
      │
      ▼
results_judged.csv
      │
      └─ hierarchical_model.py
           → trace_hier_conf.nc      (all 6 models, confidence only)
           → trace_hier_full.nc      (4 OpenAI models, conf + logprob)
           → forest_beta_conf.png
           → forest_beta_logp.png
           → calibration_by_model.png
```

---

## Step 1: Run OpenAI models

```bash
# Run the 3 new OpenAI models (skip gpt-4o-mini if merging original results)
python run_pipeline.py --models gpt-4.1-nano gpt-4.1-mini gpt-4.1

# Or all 4 if starting fresh
python run_pipeline.py --models gpt-4.1-nano gpt-4o-mini gpt-4.1-mini gpt-4.1

# Resume after interruption
python run_pipeline.py --models gpt-4.1-nano gpt-4.1-mini gpt-4.1 --resume

# Quick test (20 questions each)
python run_pipeline.py --models gpt-4.1-nano gpt-4.1-mini --limit 20
```

## Step 2: Run Claude models

```bash
pip install anthropic   # if not installed

python run_claude.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6

# Resume
python run_claude.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6 --resume
```

## Step 3: Merge original gpt-4o-mini results (optional but saves ~$0.05)

```python
import pandas as pd

# Load original manually-labeled results
df_orig = pd.read_excel("results_with_logprobs.xlsx")
df_orig["model"] = "gpt-4o-mini"

# Load new multi-model results
df_new = pd.read_csv("results_multimodel.csv")

# If you ran gpt-4o-mini again, drop it from new results
df_new = df_new[df_new["model"] != "gpt-4o-mini"]

# Combine
combined = pd.concat([df_orig, df_new], ignore_index=True)
combined.to_csv("results_multimodel.csv", index=False)
```

## Step 4: Judge correctness

```bash
# With original manual labels (skips re-judging gpt-4o-mini)
python judge.py \
    --input results_multimodel.csv \
    --output results_judged.csv \
    --original-labels results_with_logprobs.xlsx

# Resume if interrupted
python judge.py \
    --input results_multimodel.csv \
    --output results_judged.csv \
    --original-labels results_with_logprobs.xlsx \
    --resume
```

## Step 5: Hierarchical Bayesian model

```bash
# Full pipeline (Model A: all 6, Model B: 4 OpenAI)
python hierarchical_model.py --input results_judged.csv

# Confidence only (faster, if you want to skip Model B)
python hierarchical_model.py --input results_judged.csv --conf-only

# More samples for final paper figures
python hierarchical_model.py --input results_judged.csv --draws 2000 --chains 4
```

---

## Model structure

### Model A — Confidence only (all 6 models)
```
correctness_i ~ Bernoulli(sigmoid(alpha_m[i] + beta_conf_m[i] * conf_z_i))

alpha_m     = mu_alpha     + offset_alpha_m    * sigma_alpha
beta_conf_m = mu_beta_conf + offset_conf_m     * sigma_beta_conf

Hyperpriors:  mu_* ~ Normal(0, 1),  sigma_* ~ HalfNormal(1)
```

### Model B — Full (4 OpenAI models, logprobs available)
```
logit(p_i) = alpha_m[i] + beta_conf_m[i] * conf_z_i + beta_logp_m[i] * logp_z_i
```

### Key interpretations
- `mu_beta_conf` — population effect of confidence across all LLMs
- `sigma_beta_conf` — how much models differ in confidence calibration
- `beta_conf[m]` — shrunk per-model estimate (partial pooling)
- Forest plots show whether the confidence → correctness relationship
  is consistent across models or varies (and in which direction)

---

## Note on Claude models and logprobs

The Anthropic API does not expose token-level log-probabilities.
Claude rows have `avg_logprob = NaN` and are excluded from Model B.
In the paper:

> "For Claude models, only self-reported confidence is available as an
> uncertainty signal, as the Anthropic API does not expose token-level
> log-probabilities. These models are therefore included in the
> confidence-only specification only."
