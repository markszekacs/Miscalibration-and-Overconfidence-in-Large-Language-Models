"""
analysis.py
===========
Summary statistics for all 6 models:
  - Accuracy
  - Mean confidence
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Calibration table per model
  - Overconfidence gap (mean confidence - accuracy)

Usage:
  python analysis.py --input results_judged.csv
  python analysis.py --input results_judged.csv --bins 10
"""

import argparse
import numpy as np
import pandas as pd

MODEL_ORDER = [
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
]

MODEL_LABELS = {
    "gpt-4.1-nano":            "GPT-4.1 Nano",
    "gpt-4o-mini":             "GPT-4o Mini",
    "gpt-4.1-mini":            "GPT-4.1 Mini",
    "gpt-4.1":                 "GPT-4.1",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-6":       "Claude Sonnet 4.6",
}


def ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 5) -> float:
    """
    Expected Calibration Error.
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return ece_val


def mce(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 5) -> float:
    """
    Maximum Calibration Error.
    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        errors.append(abs(bin_acc - bin_conf))

    return max(errors) if errors else np.nan


def calibration_table(df_model: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Per-bin calibration breakdown for one model."""
    df_model = df_model.copy()
    df_model["conf_bin"] = pd.cut(df_model["confidence"], bins=n_bins)

    table = (df_model.groupby("conf_bin", observed=False)
             .agg(
                 mean_confidence=("confidence", "mean"),
                 accuracy=("correctness", "mean"),
                 n=("correctness", "size"),
             )
             .reset_index())

    table["gap"] = table["mean_confidence"] - table["accuracy"]
    return table


def main():
    parser = argparse.ArgumentParser(description="Accuracy and calibration summary")
    parser.add_argument("--input", default="results_judged.csv")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--output", default="calibration_summary.csv",
                        help="Save model-level summary to CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["confidence", "correctness"]).copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").astype(int)
    df = df.dropna(subset=["confidence", "correctness"]).copy()

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    # Add any unlisted models
    for m in df["model"].unique():
        if m not in models_present:
            models_present.append(m)

    # ── Model-level summary ────────────────────────────────────────────────
    rows = []
    for model in models_present:
        df_m = df[df["model"] == model]
        conf = df_m["confidence"].values
        corr = df_m["correctness"].values

        rows.append({
            "model": MODEL_LABELS.get(model, model),
            "n": len(df_m),
            "accuracy": corr.mean(),
            "mean_confidence": conf.mean(),
            "overconfidence_gap": conf.mean() - corr.mean(),
            "ece": ece(conf, corr, args.bins),
            "mce": mce(conf, corr, args.bins),
            "has_logprobs": df_m["avg_logprob"].notna().any(),
        })

    summary = pd.DataFrame(rows)

    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(summary.drop(columns=["has_logprobs"]).to_string(index=False, float_format="{:.3f}".format))

    summary.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # ── Per-model calibration tables ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("CALIBRATION BY BIN (per model)")
    print("=" * 70)

    for model in models_present:
        df_m = df[df["model"] == model]
        label = MODEL_LABELS.get(model, model)
        table = calibration_table(df_m, args.bins)

        print(f"\n{label}  (N={len(df_m)}, accuracy={df_m['correctness'].mean():.3f}, "
              f"ECE={ece(df_m['confidence'].values, df_m['correctness'].values, args.bins):.3f})")
        print(table.to_string(index=False, float_format="{:.3f}".format))

    # ── Logprob summary (OpenAI models only) ──────────────────────────────
    df_logp = df[df["avg_logprob"].notna()].copy()
    if len(df_logp) > 0:
        print("\n" + "=" * 70)
        print("AVG LOGPROB SUMMARY (OpenAI models only)")
        print("=" * 70)
        logp_summary = (df_logp.groupby("model")
                        .agg(
                            mean_logprob=("avg_logprob", "mean"),
                            mean_prob_score=("prob_score", "mean"),
                            corr_logprob_correctness=("avg_logprob",
                                lambda x: x.corr(df_logp.loc[x.index, "correctness"]))
                        )
                        .reset_index())
        logp_summary["model"] = logp_summary["model"].map(
            lambda m: MODEL_LABELS.get(m, m))
        print(logp_summary.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
