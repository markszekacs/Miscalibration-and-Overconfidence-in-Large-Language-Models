"""
hierarchical_model.py
=====================
Bayesian hierarchical logistic regression across 6 LLMs.
Crossed random effects: model × category.
 
Model specifications:
 
  Model A — Confidence only, crossed (all 6 models):
    logit(p_i) = alpha_m[i] + gamma_c[i] + beta_conf_m[i] * confidence_z_i
 
    where gamma_c is a category-level random intercept (partial pooling):
      gamma_c ~ Normal(0, sigma_gamma)
      sigma_gamma ~ HalfNormal(1)
 
  Model B — Confidence + logprob, crossed (4 OpenAI models only):
    logit(p_i) = alpha_m[i] + gamma_c[i] + beta_conf_m[i] * confidence_z_i
                                          + beta_logp_m[i] * avg_logprob_z_i
 
Key outputs:
  - trace_hier_conf.nc         ArviZ trace, Model A
  - trace_hier_full.nc         ArviZ trace, Model B
  - forest_beta_conf.png       Forest plot: beta_conf per model
  - forest_gamma.png           Forest plot: gamma per category (difficulty)
  - calibration_by_model.png   Per-model calibration curves
 
Usage:
  python hierarchical_model.py --input results_judged.csv
  python hierarchical_model.py --input results_judged.csv --conf-only
  python hierarchical_model.py --input results_judged.csv --draws 500 --chains 1
"""
 
import argparse
 
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# ---------------------------------------------------------------------------
# Model order for plots: nano → mini → 4o-mini → 4.1 → haiku → sonnet
# (capability gradient left to right)
MODEL_ORDER = [
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
]
 
MODEL_LABELS = {
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1": "GPT-4.1",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}
 
OPENAI_MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
 
PLOT_COLOR = "#1f2f4a"
 
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})
# ---------------------------------------------------------------------------
 
 
def standardize(series: pd.Series) -> tuple[pd.Series, float, float]:
    mu = series.mean()
    sigma = series.std()
    return (series - mu) / sigma, mu, sigma
 
 
def encode_models(df: pd.DataFrame, model_col: str = "model"):
    """Return integer index array and ordered list of model names."""
    present = [m for m in MODEL_ORDER if m in df[model_col].unique()]
    # Add any models not in MODEL_ORDER (shouldn't happen, but safe)
    for m in df[model_col].unique():
        if m not in present:
            present.append(m)
    idx_map = {m: i for i, m in enumerate(present)}
    return df[model_col].map(idx_map).values, present
 
 
# ---------------------------------------------------------------------------
# PyMC models
# ---------------------------------------------------------------------------
 
def build_conf_only_model(df: pd.DataFrame, model_idx: np.ndarray, model_names: list,
                          cat_idx: np.ndarray, cat_names: list):
    coords = {"model": model_names, "category": cat_names}
    with pm.Model(coords=coords) as model:
        # ── Model-level hyperpriors ──────────────────────────────────────────
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        mu_beta_conf = pm.Normal("mu_beta_conf", mu=0, sigma=1)
        sigma_beta_conf = pm.HalfNormal("sigma_beta_conf", sigma=1)
 
        # ── Category-level hyperprior ────────────────────────────────────────
        # gamma_c captures inherent difficulty of each TruthfulQA category
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1)
 
        # ── Non-centered parameterization ────────────────────────────────────
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, dims="model")
        beta_conf_offset = pm.Normal("beta_conf_offset", mu=0, sigma=1, dims="model")
        gamma_offset = pm.Normal("gamma_offset", mu=0, sigma=1, dims="category")
 
        alpha = pm.Deterministic("alpha",
                                 mu_alpha + alpha_offset * sigma_alpha,
                                 dims="model")
        beta_conf = pm.Deterministic("beta_conf",
                                     mu_beta_conf + beta_conf_offset * sigma_beta_conf,
                                     dims="model")
        # gamma: category random intercept (difficulty)
        gamma = pm.Deterministic("gamma",
                                 gamma_offset * sigma_gamma,
                                 dims="category")
 
        logit_p = (alpha[model_idx]
                   + gamma[cat_idx]
                   + beta_conf[model_idx] * df["confidence_z"].values)
        p = pm.math.sigmoid(logit_p)
        pm.Bernoulli("y_obs", p=p, observed=df["correctness"].values)
 
    return model
 
 
def build_full_model(df: pd.DataFrame, model_idx: np.ndarray, model_names: list,
                     cat_idx: np.ndarray, cat_names: list):
    coords = {"model": model_names, "category": cat_names}
    with pm.Model(coords=coords) as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        mu_beta_conf = pm.Normal("mu_beta_conf", mu=0, sigma=1)
        sigma_beta_conf = pm.HalfNormal("sigma_beta_conf", sigma=1)
        mu_beta_logp = pm.Normal("mu_beta_logp", mu=0, sigma=1)
        sigma_beta_logp = pm.HalfNormal("sigma_beta_logp", sigma=1)
 
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1)
 
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, dims="model")
        beta_conf_offset = pm.Normal("beta_conf_offset", mu=0, sigma=1, dims="model")
        beta_logp_offset = pm.Normal("beta_logp_offset", mu=0, sigma=1, dims="model")
        gamma_offset = pm.Normal("gamma_offset", mu=0, sigma=1, dims="category")
 
        alpha = pm.Deterministic("alpha",
                                 mu_alpha + alpha_offset * sigma_alpha,
                                 dims="model")
        beta_conf = pm.Deterministic("beta_conf",
                                     mu_beta_conf + beta_conf_offset * sigma_beta_conf,
                                     dims="model")
        beta_logp = pm.Deterministic("beta_logp",
                                     mu_beta_logp + beta_logp_offset * sigma_beta_logp,
                                     dims="model")
        gamma = pm.Deterministic("gamma",
                                 gamma_offset * sigma_gamma,
                                 dims="category")
 
        logit_p = (alpha[model_idx]
                   + gamma[cat_idx]
                   + beta_conf[model_idx] * df["confidence_z"].values
                   + beta_logp[model_idx] * df["avg_logprob_z"].values)
        p = pm.math.sigmoid(logit_p)
        pm.Bernoulli("y_obs", p=p, observed=df["correctness"].values)
 
    return model
 
 
# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
 
def plot_forest(trace, var_name: str, model_names: list, filename: str, title: str):
    """Forest plot of a per-model coefficient."""
    posterior = trace.posterior[var_name]
    means = posterior.mean(dim=("chain", "draw")).values
    lower = posterior.quantile(0.025, dim=("chain", "draw")).values
    upper = posterior.quantile(0.975, dim=("chain", "draw")).values
 
    labels = [MODEL_LABELS.get(m, m) for m in model_names]
    y = np.arange(len(model_names))
 
    fig, ax = plt.subplots(figsize=(6, 0.7 * len(model_names) + 1.5), dpi=200)
 
    ax.barh(y, upper - lower, left=lower, height=0.4,
            color=PLOT_COLOR, alpha=0.25, label="95% CI")
    ax.scatter(means, y, color=PLOT_COLOR, zorder=5, s=30)
    ax.axvline(0, linestyle="--", linewidth=0.9, color="#444444")
 
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Posterior mean and 95% credible interval")
    ax.set_title(title)
 
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(axis="x", alpha=0.12)
    ax.set_facecolor("#fafafa")
 
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {filename}")
 
 
def plot_calibration_grid(df: pd.DataFrame, model_names: list, filename: str):
    """Per-model calibration curves in a grid."""
    n = len(model_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
 
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             dpi=150)
    axes = axes.flatten()
 
    for i, model in enumerate(model_names):
        ax = axes[i]
        df_m = df[df["model"] == model].copy()
 
        if len(df_m) == 0:
            ax.set_visible(False)
            continue
 
        df_m["conf_bin"] = pd.cut(df_m["confidence"], bins=5)
        bin_means = df_m.groupby("conf_bin", observed=False)["confidence"].mean()
        bin_acc = df_m.groupby("conf_bin", observed=False)["correctness"].mean()
        bin_n = df_m.groupby("conf_bin", observed=False).size()
 
        ax.plot(bin_means.values, bin_acc.values,
                color=PLOT_COLOR, linewidth=1.4, marker="o", markersize=4)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=9.5)
        ax.set_xlabel("Reported confidence", fontsize=8.5)
        ax.set_ylabel("Observed accuracy", fontsize=8.5)
        ax.set_facecolor("#fafafa")
        ax.grid(alpha=0.08)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
 
        # Annotate overall accuracy
        acc = df_m["correctness"].mean()
        ax.text(0.05, 0.92, f"Acc={acc:.2f}  N={len(df_m)}",
                transform=ax.transAxes, fontsize=7.5, color="#555555")
 
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    plt.suptitle("Calibration by model", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {filename}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Hierarchical Bayesian calibration — 6 models")
    parser.add_argument("--input", default="results_judged.csv")
    parser.add_argument("--conf-only", action="store_true",
                        help="Skip full model (use if no logprob data available)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()
 
    # ── Load & clean ───────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df = df.dropna(subset=["confidence", "correctness", "model"]).copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").astype(int)
    df = df.dropna(subset=["confidence", "correctness"]).copy()
 
    print(f"Total observations: {len(df)}")
    print(f"\nAccuracy and N by model:")
    summary = df.groupby("model")["correctness"].agg(accuracy="mean", n="count")
    print(summary.round(3).to_string())
 
    # ── Standardize ────────────────────────────────────────────────────────
    df["confidence_z"], conf_mu, conf_sd = standardize(df["confidence"])
 
    # ── Calibration grid (raw, before modeling) ────────────────────────────
    model_names_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    plot_calibration_grid(df, model_names_present, "calibration_by_model.png")
 
    # ── Category encoding ─────────────────────────────────────────────────
    df = df.dropna(subset=["category"]).copy()
    cat_names = sorted(df["category"].unique().tolist())
    cat_map = {c: i for i, c in enumerate(cat_names)}
    cat_idx = df["category"].map(cat_map).values
    print(f"\nCategories ({len(cat_names)}): {cat_names}")
    print("\nAccuracy by category:")
    print(df.groupby("category")["correctness"].agg(accuracy="mean", n="count").round(3).to_string())
 
    # ── Model A: confidence only + category RE (all 6 models) ─────────────
    model_idx, model_names = encode_models(df)
    print(f"\nModel A — confidence only + category RE ({len(model_names)} models)")
 
    pymc_conf = build_conf_only_model(df, model_idx, model_names, cat_idx, cat_names)
    with pymc_conf:
        try:
            trace_conf = pm.sample(
                draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.chains,
                target_accept=0.9, random_seed=42,
                nuts_sampler="numpyro",
            )
        except Exception:
            # Fallback if numpyro not installed
            trace_conf = pm.sample(
                draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.chains,
                target_accept=0.9, random_seed=42,
            )
 
    print("\n=== Model A summary ===")
    print(az.summary(trace_conf,
                     var_names=["mu_beta_conf", "sigma_beta_conf", "sigma_gamma", "beta_conf"],
                     hdi_prob=0.95).to_string())
 
    az.to_netcdf(trace_conf, "trace_hier_conf.nc")
    print("Saved trace_hier_conf.nc")
 
    plot_forest(trace_conf, "beta_conf", model_names,
                "forest_beta_conf.png",
                r"$\beta_{\mathrm{conf}}$ by model (hierarchical, 95% CI)")
 
    plot_forest(trace_conf, "gamma", cat_names,
                "forest_gamma.png",
                r"$\gamma_{\mathrm{cat}}$ by category (difficulty, 95% CI)")
 
    # Population-level summary
    print("\n=== Population hyperparameters (Model A) ===")
    print(az.summary(trace_conf,
                     var_names=["mu_alpha", "mu_beta_conf",
                                "sigma_alpha", "sigma_beta_conf", "sigma_gamma"],
                     hdi_prob=0.95).to_string())
 
    # ── Model B: full model (OpenAI models only) ───────────────────────────
    if not args.conf_only:
        df_full = df[df["model"].isin(OPENAI_MODELS)].copy()
        df_full = df_full.dropna(subset=["avg_logprob"]).copy()
        df_full["avg_logprob"] = pd.to_numeric(df_full["avg_logprob"], errors="coerce")
        df_full = df_full.dropna(subset=["avg_logprob"]).copy()
        df_full["avg_logprob_z"], logp_mu, logp_sd = standardize(df_full["avg_logprob"])
 
        model_idx_full, model_names_full = encode_models(df_full)
        print(f"\nModel B — full ({len(model_names_full)} OpenAI models: {model_names_full})")
        print(f"  Observations: {len(df_full)}")
 
        cat_names_full = sorted(df_full["category"].dropna().unique().tolist())
        cat_map_full = {c: i for i, c in enumerate(cat_names_full)}
        cat_idx_full = df_full["category"].map(cat_map_full).values
 
        pymc_full = build_full_model(df_full, model_idx_full, model_names_full, cat_idx_full, cat_names_full)
        with pymc_full:
            try:
                trace_full = pm.sample(
                    draws=args.draws, tune=args.tune,
                    chains=args.chains, cores=args.chains,
                    target_accept=0.9, random_seed=42,
                    nuts_sampler="numpyro",
                )
            except Exception:
                trace_full = pm.sample(
                    draws=args.draws, tune=args.tune,
                    chains=args.chains, cores=args.chains,
                    target_accept=0.9, random_seed=42,
                )
 
        print("\n=== Model B summary ===")
        print(az.summary(trace_full,
                         var_names=["mu_beta_conf", "mu_beta_logp",
                                    "sigma_beta_conf", "sigma_beta_logp",
                                    "beta_conf", "beta_logp"],
                         hdi_prob=0.95).to_string())
 
        az.to_netcdf(trace_full, "trace_hier_full.nc")
        print("Saved trace_hier_full.nc")
 
        plot_forest(trace_full, "beta_logp", model_names_full,
                    "forest_beta_logp.png",
                    r"$\beta_{\mathrm{logp}}$ by model (hierarchical, 95% CI)")
 
        print("\n=== Population hyperparameters (Model B) ===")
        print(az.summary(trace_full,
                         var_names=["mu_alpha", "mu_beta_conf", "mu_beta_logp",
                                    "sigma_alpha", "sigma_beta_conf", "sigma_beta_logp"],
                         hdi_prob=0.95).to_string())
 
 
if __name__ == "__main__":
    main()
