"""
Deliverable 2 — Bootstrap confidence intervals on AUARC.

Bootstrap is performed on the 60 per-question (correct, confidence) pairs.
For each resample we compute the threshold-sweep AUARC using the same method
as the paper: for each threshold in {0.0, 0.1, …, 1.0}, take questions with
confidence >= threshold as "answered", compute accuracy, then integrate over
(coverage, accuracy) with np.trapz (None acc at zero coverage → 0.0).

Saves results/analysis/analysis_02_bootstrap_auarc.json
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shared_inputs import FILES, load_rows, validate_inputs

SEED = 42
N_BOOT = 1000
THRESHOLDS = [round(t / 10, 1) for t in range(11)]  # 0.0 … 1.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)


def auarc_threshold_sweep(pairs):
    """
    pairs: list of (correct: int, conf: float) — conf may be None (→ abstain).
    Returns AUARC using threshold-sweep; thresholds where no questions are answered
    (acc=None) are SKIPPED, matching the computation for 10/11 paper values.
    """
    n = len(pairs)
    covs, accs = [], []
    for t in THRESHOLDS:
        answered = [(c, cf) for (c, cf) in pairs if cf is not None and cf >= t]
        if not answered:
            continue   # skip zero-coverage point (skip-None convention)
        cov = len(answered) / n if n else 0.0
        acc = sum(c for c, _ in answered) / len(answered)
        covs.append(cov)
        accs.append(acc)
    if not covs:
        return 0.0
    covs_arr = np.array(covs)
    accs_arr = np.array(accs)
    order = np.argsort(covs_arr)
    return float(abs(np.trapz(accs_arr[order], covs_arr[order])))


def bootstrap_auarc(rows, n_boot=N_BOOT, seed=SEED):
    """rows: list of (qid, correct, conf). conf None → abstain."""
    pairs = [(c, conf) for (_, c, conf) in rows]
    n = len(pairs)
    rng = np.random.default_rng(seed)
    # Point estimate from full set
    point = auarc_threshold_sweep(pairs)
    # Bootstrap
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [pairs[i] for i in idx]
        boot_vals.append(auarc_threshold_sweep(sample))
    arr = np.array(boot_vals)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return point, float(np.mean(arr)), float(lo), float(hi), arr


def main():
    validate_inputs()

    boot_store = {}
    point_store = {}
    ci_table = []

    print("=" * 70)
    print(f"BOOTSTRAP AUARC  (threshold-sweep, n_boot={N_BOOT}, seed={SEED})")
    print("=" * 70)
    print(f"\n  {'System':<16}  {'Bench':<10}  {'AUARC':>6}  {'Boot mean':>9}  {'95% CI':>18}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*18}")

    for sys_name, bench, rel in FILES:
        rows = load_rows(rel)
        if not rows:
            print(f"  {sys_name:<16}  {bench:<10}  PENDING")
            ci_table.append({
                "system": sys_name, "benchmark": bench,
                "auarc_point": None, "auarc_boot_mean": None,
                "ci_lo": None, "ci_hi": None,
                "note": "PENDING - run artifact not available",
            })
            continue
        point, mean_a, lo, hi, arr = bootstrap_auarc(rows)
        boot_store[(sys_name, bench)] = arr
        point_store[(sys_name, bench)] = point
        print(f"  {sys_name:<16}  {bench:<10}  {point:.3f}  {mean_a:.3f}      [{lo:.3f}, {hi:.3f}]")
        ci_table.append({
            "system": sys_name, "benchmark": bench,
            "auarc_point": round(point, 4),
            "auarc_boot_mean": round(mean_a, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        })

    # ── Pairwise inversion tests ─────────────────────────────────────────────
    # Claimed inversions in paper (AUARC axis):
    #   A: HippoRAG 32B is top-ranked on MedMCQA (0.697) but rank 4 on PubMedQA (0.449)
    #      while IRCoT 7B is rank 4 on MedMCQA (0.452) but rank 1 on PubMedQA (0.607)
    #   B: IRCoT 32B > IRCoT 7B on MedMCQA AUARC (0.595 vs 0.452),
    #      reversed on PubMedQA (0.471 vs 0.607)

    INVERSIONS = [
        {
            "label": "HippoRAG 32B #1 -> #4 (MedMCQA -> PubMedQA); IRCoT 7B #4 -> #1",
            "desc": "HippoRAG 32B > IRCoT 7B on MedMCQA AUARC; IRCoT 7B > HippoRAG 32B on PubMedQA AUARC",
            "pair_bench1": (("HippoRAG 32B", "MedMCQA"), ("IRCoT 7B",    "MedMCQA")),
            "pair_bench2": (("IRCoT 7B",    "PubMedQA"), ("HippoRAG 32B","PubMedQA")),
        },
        {
            "label": "IRCoT 32B > IRCoT 7B on MedMCQA; reversed on PubMedQA",
            "desc": "IRCoT 32B > IRCoT 7B on MedMCQA AUARC; IRCoT 7B > IRCoT 32B on PubMedQA AUARC",
            "pair_bench1": (("IRCoT 32B", "MedMCQA"), ("IRCoT 7B", "MedMCQA")),
            "pair_bench2": (("IRCoT 7B", "PubMedQA"), ("IRCoT 32B","PubMedQA")),
        },
    ]

    print("\n")
    print("=" * 70)
    print("PAIRWISE INVERSION ROBUSTNESS TESTS")
    print("=" * 70)

    inversion_results = []
    for inv in INVERSIONS:
        A1, B1 = inv["pair_bench1"]
        A2, B2 = inv["pair_bench2"]
        d1 = boot_store[A1] - boot_store[B1]  # positive means A1 > B1 (desired on bench1)
        d2 = boot_store[A2] - boot_store[B2]  # positive means A2 > B2 (desired on bench2)
        holds = (d1 > 0) & (d2 > 0)
        pct = 100.0 * holds.mean()
        lo1, hi1 = np.percentile(d1, [2.5, 97.5])
        lo2, hi2 = np.percentile(d2, [2.5, 97.5])

        print(f"\n  {inv['label']}")
        print(f"    {inv['desc']}")
        print(f"    Delta({A1[0]} - {B1[0]}) on {A1[1]}:   "
              f"mean={d1.mean():+.3f}, 95%CI=[{lo1:+.3f},{hi1:+.3f}]")
        print(f"    Delta({A2[0]} - {B2[0]}) on {A2[1]}: "
              f"mean={d2.mean():+.3f}, 95%CI=[{lo2:+.3f},{hi2:+.3f}]")
        print(f"    Both inversions hold in {pct:.1f}% of {N_BOOT} resamples")

        inversion_results.append({
            "label": inv["label"],
            "desc": inv["desc"],
            "delta_bench1": {
                "pair": f"{A1[0]} - {B1[0]} on {A1[1]}",
                "mean": round(float(d1.mean()), 4),
                "ci": [round(lo1, 4), round(hi1, 4)],
            },
            "delta_bench2": {
                "pair": f"{A2[0]} - {B2[0]} on {A2[1]}",
                "mean": round(float(d2.mean()), 4),
                "ci": [round(lo2, 4), round(hi2, 4)],
            },
            "inversion_holds_pct": round(pct, 1),
            "n_boot": N_BOOT, "seed": SEED,
        })

    out = {
        "seed": SEED, "n_boot": N_BOOT,
        "method": "threshold-sweep [0.0,0.1,...,1.0], conf=None->abstain, zero-coverage threshold skipped (skip-None)",
        "ci_table": ci_table,
        "inversion_tests": inversion_results,
    }
    out_json = os.path.join(RESULTS_DIR, "analysis_02_bootstrap_auarc.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_json}")


if __name__ == "__main__":
    main()
