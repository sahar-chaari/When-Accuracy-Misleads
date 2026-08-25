"""
Deliverable 3 — Cross-axis rank correlation.

Builds a table: 11 configurations × 4 axis scores
  (accuracy, AUARC, faithfulness sup-rate, harm cost H).
Computes Spearman ρ between all axis pairs.
Harm cost H is inverted (lower = better), so we negate it before ranking
so that higher rank = better on all axes.

Axis scores:
  Accuracy   — from Table 1 (macro-avg across answer choices)
  AUARC      — from Table 2
  Faithful   — supported-rate from Table 3 (col "Sup%" for each bench)
  Harm cost  — H from Table 5, negated so higher = better

Saves results/analysis/analysis_03_rank_correlation.json
"""

import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from shared_inputs import validate_inputs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Axis scores from paper tables ──────────────────────────────────────────
# Format: (system, benchmark) → score
# Accuracy (Table 1): use macro-F1 column where available; otherwise accuracy
# Using the F1 columns (MedMCQA: acc, F1; PubMedQA: acc, F1)
# MedMCQA accuracy col, PubMedQA accuracy col

ACCURACY = {
    # MedMCQA (acc, F1) — use F1 for macro
    ("IRCoT 7B",    "MedMCQA"):  50.4,   # F1
    ("IRCoT 32B",   "MedMCQA"):  51.1,
    ("CRAG 7B",     "MedMCQA"):  37.1,
    ("CRAG 32B",    "MedMCQA"):  44.8,
    ("HippoRAG 7B", "MedMCQA"):  48.2,
    ("HippoRAG 32B","MedMCQA"):  50.8,
    # PubMedQA (acc, F1) — use F1
    ("IRCoT 7B",    "PubMedQA"): 42.4,
    ("IRCoT 32B",   "PubMedQA"): 52.7,
    ("CRAG 7B",     "PubMedQA"): 43.1,
    ("CRAG 32B",    "PubMedQA"): 40.2,
    ("HippoRAG 7B", "PubMedQA"): 46.3,
    ("HippoRAG 32B","PubMedQA"): 52.3,
}

AUARC = {
    ("IRCoT 7B",    "MedMCQA"):  0.452472,
    ("IRCoT 32B",   "MedMCQA"):  0.595158,
    ("CRAG 7B",     "MedMCQA"):  0.513260,
    ("CRAG 32B",    "MedMCQA"):  0.445165,
    ("HippoRAG 7B", "MedMCQA"):  0.512523,
    ("HippoRAG 32B","MedMCQA"):  0.697379,
    ("IRCoT 7B",    "PubMedQA"): 0.607405,
    ("IRCoT 32B",   "PubMedQA"): 0.471208,
    ("CRAG 7B",     "PubMedQA"): 0.430780,
    ("CRAG 32B",    "PubMedQA"): 0.505146,
    ("HippoRAG 7B", "PubMedQA"): 0.355717,
    ("HippoRAG 32B","PubMedQA"): 0.448587,
}

# Faithfulness — supported% (col "Sup%", from Table 3)
# MedMCQA: supported%
FAITHFUL = {
    ("IRCoT 7B",    "MedMCQA"):   5.0,
    ("IRCoT 32B",   "MedMCQA"):   3.3,
    ("CRAG 7B",     "MedMCQA"):   1.7,
    ("CRAG 32B",    "MedMCQA"):   8.3,
    ("HippoRAG 7B", "MedMCQA"):   3.3,
    ("HippoRAG 32B","MedMCQA"):   5.0,
    ("IRCoT 7B",    "PubMedQA"): 11.7,
    ("IRCoT 32B",   "PubMedQA"): 18.3,
    ("CRAG 7B",     "PubMedQA"):  6.7,
    ("CRAG 32B",    "PubMedQA"): 11.7,
    ("HippoRAG 7B", "PubMedQA"): 18.3,
    ("HippoRAG 32B","PubMedQA"): 21.7,
}

# Harm cost H (Table 5) — lower = better; we negate so higher = better
HARM_H = {
    ("IRCoT 7B",    "MedMCQA"):  0.596,
    ("IRCoT 32B",   "MedMCQA"):  0.504,
    ("CRAG 7B",     "MedMCQA"):  0.652,
    ("CRAG 32B",    "MedMCQA"):  0.546,
    ("HippoRAG 7B", "MedMCQA"):  0.496,
    ("HippoRAG 32B","MedMCQA"):  0.504,
    ("IRCoT 7B",    "PubMedQA"): 0.559,
    ("IRCoT 32B",   "PubMedQA"): 0.461,
    ("CRAG 7B",     "PubMedQA"): 0.549,
    ("CRAG 32B",    "PubMedQA"): 0.603,
    ("HippoRAG 7B", "PubMedQA"): 0.485,
    ("HippoRAG 32B","PubMedQA"): 0.480,
}

# Ordered list of 12 configurations
CONFIGS = [
    ("IRCoT 7B",    "MedMCQA"),
    ("IRCoT 32B",   "MedMCQA"),
    ("CRAG 7B",     "MedMCQA"),
    ("CRAG 32B",    "MedMCQA"),
    ("HippoRAG 7B", "MedMCQA"),
    ("HippoRAG 32B","MedMCQA"),
    ("IRCoT 7B",    "PubMedQA"),
    ("IRCoT 32B",   "PubMedQA"),
    ("CRAG 7B",     "PubMedQA"),
    ("CRAG 32B",    "PubMedQA"),
    ("HippoRAG 7B", "PubMedQA"),
    ("HippoRAG 32B","PubMedQA"),
]

AXES = ["Macro-F1", "AUARC", "Faithful (Sup%)", "Harm (-H)"]


def build_score_matrix():
    """Returns (active_configs, mat) — rows with any None value are skipped."""
    active, rows = [], []
    for key in CONFIGS:
        vals = [ACCURACY.get(key), AUARC.get(key), FAITHFUL.get(key), HARM_H.get(key)]
        if any(v is None for v in vals):
            continue   # skip pending configurations
        active.append(key)
        rows.append([vals[0], vals[1], vals[2], -vals[3]])  # negate H
    return active, np.array(rows, dtype=float)


def main():
    validate_inputs()

    active_configs, mat = build_score_matrix()
    n = len(active_configs)
    pending = [k for k in CONFIGS if k not in active_configs]

    print("=" * 70)
    print(f"CROSS-AXIS SCORE TABLE  (n={n} configurations; {len(pending)} pending)")
    print("=" * 70)
    if pending:
        print(f"  PENDING: {', '.join(f'{s}/{b[:3]}' for s,b in pending)}\n")
    header = f"  {'Config':<22}  {'Acc(F1)':>8}  {'AUARC':>6}  {'Sup%':>5}  {'-H':>7}"
    print(header)
    print("  " + "-" * 60)
    for i, (sys_name, bench) in enumerate(active_configs):
        label = f"{sys_name}/{bench[:3]}"
        print(f"  {label:<22}  {mat[i,0]:>8.1f}  {mat[i,1]:>6.3f}  {mat[i,2]:>5.1f}  {mat[i,3]:>7.3f}")

    # Spearman correlation between all axis pairs
    axis_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pair_labels = [
        ("F1 vs AUARC",   "f1-auarc"),
        ("F1 vs Sup%",    "f1-faithful"),
        ("F1 vs -H",      "f1-harm"),
        ("AUARC vs Sup%", "auarc-faithful"),
        ("AUARC vs -H",   "auarc-harm"),
        ("Sup% vs -H",    "faithful-harm"),
    ]

    print("\n")
    print("=" * 70)
    print(f"SPEARMAN RANK CORRELATIONS  (n={n} configurations)")
    print("  Note: H negated -> '-H' so all axes point higher=better")
    print("=" * 70)
    print(f"  {'Pair':<20}  {'rho':>6}  {'p-val':>8}  {'Signif'}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*10}")

    corr_results = []
    for (i, j), (label, key) in zip(axis_pairs, pair_labels):
        rho, pval = stats.spearmanr(mat[:, i], mat[:, j])
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        print(f"  {label:<20}  {rho:>+6.3f}  {pval:>8.4f}  {sig}")
        corr_results.append({
            "pair": label, "key": key,
            "rho": round(float(rho), 4), "p_value": round(float(pval), 4),
            "significant_05": bool(pval < 0.05),
        })

    # Plain-language summary
    print("\n")
    print("=" * 70)
    print("PLAIN-LANGUAGE SUMMARY")
    print("=" * 70)
    for r in corr_results:
        direction = "positive" if r["rho"] > 0 else "negative"
        strength = "strong" if abs(r["rho"]) > 0.6 else ("moderate" if abs(r["rho"]) > 0.3 else "weak")
        sig_str = f"p={r['p_value']:.3f}" + (" [significant]" if r["significant_05"] else " [not significant]")
        print(f"  {r['pair']}: rho={r['rho']:+.3f} ({strength} {direction}), {sig_str}, n={n}")

    print(f"\n  CAVEAT: n={n} is small; all p-values should be interpreted cautiously.")
    if pending:
        print(f"  Re-run once pending harm H values are finalized to get n={len(CONFIGS)}.")

    # Save
    table_data = []
    for i, (sys_name, bench) in enumerate(active_configs):
        table_data.append({
            "system": sys_name, "benchmark": bench,
            "accuracy_f1": mat[i, 0], "auarc": mat[i, 1],
            "faithful_sup_pct": mat[i, 2], "neg_harm_h": round(mat[i, 3], 4),
        })

    out = {
        "n_configurations": n,
        "note_harm": "H values are negated (-H) so higher = better for all axes",
        "score_table": table_data,
        "spearman_correlations": corr_results,
    }
    out_json = os.path.join(RESULTS_DIR, "analysis_03_rank_correlation.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_json}")


if __name__ == "__main__":
    main()
