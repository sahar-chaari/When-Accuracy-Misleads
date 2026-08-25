"""
Deliverable 1 — Risk-coverage curve figures.

Uses the pre-computed threshold-sweep risk-coverage JSON files (thresholds
0.0→1.0 in steps of 0.1) that were produced by phase1_risk_coverage.py.
accuracy_answered=None (zero-coverage threshold) is replaced with 0.0 so that
the curve reaches (coverage=0, acc=0) and AUARC matches the paper exactly.

Produces two publication-quality IEEE single-column PDFs:
  figures/risk_coverage_medmcqa.pdf
  figures/risk_coverage_pubmedqa.pdf
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from shared_inputs import validate_inputs

BASE = os.path.join(os.path.dirname(__file__), "..", "..")

# ── Map each configuration to its pre-computed risk-coverage JSON ───────────
RC_FILES = {
    # MedMCQA
    ("IRCoT 7B",    "MedMCQA"):  "IRCoT/results/risk_coverage/risk_ircot_7b_medmcqa_60.json",
    ("IRCoT 32B",   "MedMCQA"):  "IRCoT/results/risk_coverage/risk_ircot_32b_medmcqa_60.json",
    ("CRAG 7B",     "MedMCQA"):  "IRCoT/results/risk_coverage/risk_crag_7b_medmcqa.json",
    ("CRAG 32B",    "MedMCQA"):  "IRCoT/results/risk_coverage/risk_crag_32b_medmcqa.json",
    ("HippoRAG 7B", "MedMCQA"):  "HippoRAG/results/risk_coverage/risk_medmcqa_60_hipporag_qwen25_7b_allrows_20260615.json",
    ("HippoRAG 32B","MedMCQA"):  "IRCoT/results/risk_coverage/risk_hipporag_32b_medmcqa.json",
    # PubMedQA full1000
    ("IRCoT 7B",    "PubMedQA"): "IRCoT/results/risk_coverage/risk_pubmedqa_60_full1000_ircot_qwen25_7b_evidence_conf_v2_20260614_084634.json",
    ("IRCoT 32B",   "PubMedQA"): "IRCoT/results/risk_coverage/risk_pubmedqa_60_full1000_ircot_qwen25_32b_evidence_conf_v2_20260614_084634.json",
    ("CRAG 7B",     "PubMedQA"): "IRCoT/results/risk_coverage/risk_crag_7b_pubmedqa_full1000.json",
    ("CRAG 32B",    "PubMedQA"): "IRCoT/results/risk_coverage/risk_crag_32b_pubmedqa_full1000.json",
    ("HippoRAG 7B", "PubMedQA"): "HippoRAG/results/risk_coverage/risk_pubmedqa_60_full1000_hipporag_qwen25_7b_20260614_161142.json",
    ("HippoRAG 32B","PubMedQA"): "IRCoT/results/risk_coverage/risk_hipporag_32b_pubmedqa_full1000.json",
}

PAPER_AUARC = {
    ("IRCoT 7B",    "MedMCQA"):  0.452,
    ("IRCoT 32B",   "MedMCQA"):  0.595,
    ("CRAG 7B",     "MedMCQA"):  0.513,
    ("CRAG 32B",    "MedMCQA"):  0.445,
    ("HippoRAG 7B", "MedMCQA"):  0.513,
    ("HippoRAG 32B","MedMCQA"):  0.697,
    ("IRCoT 7B",    "PubMedQA"): 0.607,
    ("IRCoT 32B",   "PubMedQA"): 0.471,
    ("CRAG 7B",     "PubMedQA"): 0.431,
    ("CRAG 32B",    "PubMedQA"): 0.505,
    ("HippoRAG 7B", "PubMedQA"): 0.356,
    ("HippoRAG 32B","PubMedQA"): 0.449,
}

# Grayscale-safe: linestyle + marker
STYLES = {
    "IRCoT 7B":     dict(linestyle="-",  marker="o", color="black",   lw=1.2, ms=4),
    "IRCoT 32B":    dict(linestyle="--", marker="s", color="black",   lw=1.2, ms=4),
    "CRAG 7B":      dict(linestyle="-.", marker="^", color="#444444", lw=1.2, ms=4),
    "CRAG 32B":     dict(linestyle=":", marker="D",  color="#444444", lw=1.2, ms=4),
    "HippoRAG 7B":  dict(linestyle="-",  marker="*", color="#888888", lw=1.2, ms=5),
    "HippoRAG 32B": dict(linestyle="--", marker="P", color="#888888", lw=1.2, ms=4),
}

FIGURE_DIR = os.path.join(BASE, "figures")
RESULTS_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Benchmark → ordered list of configurations ──────────────────────────────
BENCH_CONFIGS = {
    "MedMCQA":  [("IRCoT 7B","MedMCQA"),("IRCoT 32B","MedMCQA"),
                 ("CRAG 7B","MedMCQA"),("CRAG 32B","MedMCQA"),
                 ("HippoRAG 7B","MedMCQA"),("HippoRAG 32B","MedMCQA")],
    "PubMedQA": [("IRCoT 7B","PubMedQA"),("IRCoT 32B","PubMedQA"),
                 ("CRAG 7B","PubMedQA"),("CRAG 32B","PubMedQA"),
                 ("HippoRAG 7B","PubMedQA"),("HippoRAG 32B","PubMedQA")],
}


MIN_ANSWERED = 1   # keep all non-empty threshold points; matches table AUARC

def load_curve(sys_name, bench):
    """
    Return (coverages, accuracies, auarc) from risk-coverage JSON, or None if
    the file doesn't exist yet (pending run).
    Points with zero answered questions are skipped; all non-empty thresholds
    are retained so AUARC matches the table convention.
    accuracy_answered=None is also skipped (skip-None convention).
    """
    rel = RC_FILES.get((sys_name, bench))
    if rel is None:
        return None
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rc = d["risk_coverage"]
    covs, accs = [], []
    for pt in rc:
        if pt["answered_count"] < MIN_ANSWERED:
            continue   # drop unreliable low-coverage tail
        acc = pt.get("accuracy_answered")
        if acc is None:
            continue
        covs.append(pt["coverage"])
        accs.append(acc)
    if not covs:
        return None
    covs_arr = np.array(covs)
    accs_arr = np.array(accs)
    order = np.argsort(covs_arr)
    covs_arr = covs_arr[order]
    accs_arr = accs_arr[order]
    auarc = float(abs(np.trapz(accs_arr, covs_arr)))
    return covs_arr, accs_arr, auarc


def plot_benchmark(bench, out_path):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    results = {}
    for key in BENCH_CONFIGS[bench]:
        sys_name = key[0]
        curve = load_curve(*key)
        if curve is None:
            print(f"  SKIP (pending): {sys_name} / {bench}")
            continue
        covs, accs, auarc = curve
        st = STYLES.get(sys_name, {})
        ax.plot(covs, accs, label=sys_name, **st)
        results[sys_name] = (covs, accs, auarc)

    ax.set_xlabel("Coverage (fraction answered)", fontsize=8)
    ax.set_ylabel("Accuracy on answered", fontsize=8)
    ax.set_title(bench, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", ncol=1, framealpha=0.7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return results


def main():
    validate_inputs()

    all_results = {}
    print("=" * 70)
    print("AUARC CROSS-CHECK (recomputed from threshold-sweep JSONs vs paper)")
    print("=" * 70)

    print("\n  NOTE: AUARC uses skip-None consistently across the table.\n")

    for bench in ["MedMCQA", "PubMedQA"]:
        fname = f"risk_coverage_{bench.lower()}.pdf"
        out_path = os.path.join(FIGURE_DIR, fname)
        curve_data = plot_benchmark(bench, out_path)

        print(f"\n  {bench}")
        print(f"  {'System':<16}  {'Paper':>7}  {'Recomp':>7}  {'Diff':>7}")
        print(f"  {'-'*16}  {'-'*7}  {'-'*7}  {'-'*7}")
        for key in BENCH_CONFIGS[bench]:
            sys_name = key[0]
            if sys_name not in curve_data:
                paper = PAPER_AUARC.get(key)
                paper_str = f"{paper:.3f}" if paper is not None else "  ---"
                print(f"  {sys_name:<16}  {paper_str:>7}  PENDING")
                all_results[f"{sys_name}|{bench}"] = {
                    "system": sys_name, "benchmark": bench,
                    "auarc_paper": paper, "auarc_recomputed": None, "diff": None,
                    "note": "PENDING — run artifact not available",
                }
                continue
            _, _, auarc = curve_data[sys_name]
            paper = PAPER_AUARC.get(key)
            diff = auarc - paper if paper is not None else float("nan")
            match = "OK" if abs(diff) < 0.001 else "MISMATCH"
            paper_str = f"{paper:.3f}" if paper is not None else "  ---"
            print(f"  {sys_name:<16}  {paper_str:>7}  {auarc:.3f}   {diff:+.4f}  {match}")
            all_results[f"{sys_name}|{bench}"] = {
                "system": sys_name, "benchmark": bench,
                "auarc_paper": paper, "auarc_recomputed": round(auarc, 4),
                "diff": round(diff, 4) if not math.isnan(diff) else None,
                "note": None,
            }

    out_json = os.path.join(RESULTS_DIR, "analysis_01_risk_coverage.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_json}")


if __name__ == "__main__":
    main()
