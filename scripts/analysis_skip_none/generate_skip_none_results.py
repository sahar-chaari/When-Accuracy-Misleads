#!/usr/bin/env python3
"""Generate skip-None AUARC tables, ranks, correlations, and figures.

This script intentionally writes to new paths only:
  - ieee/results/analysis_skip_none/
  - ieee/figures_skip_none/

AUARC matches the bootstrap convention used in
scripts/analysis/analysis_02_bootstrap_auarc.py:
  - thresholds are 0.0, 0.1, ..., 1.0
  - rows with confidence=None are abstained
  - thresholds with zero answered questions are skipped
  - AUARC is trapezoidal area over (coverage, answered-set accuracy)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "analysis_skip_none"
FIGURE_DIR = ROOT / "figures_skip_none"
MPLCONFIG_DIR = Path("/tmp/ieee_skip_none_mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from shared_inputs import FILES, load_rows  # noqa: E402

THRESHOLDS = [round(t / 10, 1) for t in range(11)]

SYSTEM_ORDER = ["IRCoT 7B", "IRCoT 32B", "CRAG 7B", "CRAG 32B", "HippoRAG 7B", "HippoRAG 32B"]
BENCHES = ["MedMCQA", "PubMedQA"]
KEYS = [(system, bench) for bench in BENCHES for system in SYSTEM_ORDER]

ACCURACY_ACC = {
    ("IRCoT 7B", "MedMCQA"): 50.0,
    ("IRCoT 32B", "MedMCQA"): 51.7,
    ("CRAG 7B", "MedMCQA"): 36.7,
    ("CRAG 32B", "MedMCQA"): 41.7,
    ("HippoRAG 7B", "MedMCQA"): 48.3,
    ("HippoRAG 32B", "MedMCQA"): 51.7,
    ("IRCoT 7B", "PubMedQA"): 45.0,
    ("IRCoT 32B", "PubMedQA"): 53.3,
    ("CRAG 7B", "PubMedQA"): 43.3,
    ("CRAG 32B", "PubMedQA"): 40.0,
    ("HippoRAG 7B", "PubMedQA"): 50.0,
    ("HippoRAG 32B", "PubMedQA"): 51.7,
}

ACCURACY_F1 = {
    ("IRCoT 7B", "MedMCQA"): 50.4,
    ("IRCoT 32B", "MedMCQA"): 51.1,
    ("CRAG 7B", "MedMCQA"): 37.1,
    ("CRAG 32B", "MedMCQA"): 44.8,
    ("HippoRAG 7B", "MedMCQA"): 48.2,
    ("HippoRAG 32B", "MedMCQA"): 50.8,
    ("IRCoT 7B", "PubMedQA"): 42.4,
    ("IRCoT 32B", "PubMedQA"): 52.7,
    ("CRAG 7B", "PubMedQA"): 43.1,
    ("CRAG 32B", "PubMedQA"): 40.2,
    ("HippoRAG 7B", "PubMedQA"): 46.3,
    ("HippoRAG 32B", "PubMedQA"): 52.3,
}

FAITHFUL_SUP = {
    ("IRCoT 7B", "MedMCQA"): 5.0,
    ("IRCoT 32B", "MedMCQA"): 3.3,
    ("CRAG 7B", "MedMCQA"): 1.7,
    ("CRAG 32B", "MedMCQA"): 8.3,
    ("HippoRAG 7B", "MedMCQA"): 3.3,
    ("HippoRAG 32B", "MedMCQA"): 5.0,
    ("IRCoT 7B", "PubMedQA"): 11.7,
    ("IRCoT 32B", "PubMedQA"): 18.3,
    ("CRAG 7B", "PubMedQA"): 6.7,
    ("CRAG 32B", "PubMedQA"): 11.7,
    ("HippoRAG 7B", "PubMedQA"): 18.3,
    ("HippoRAG 32B", "PubMedQA"): 21.7,
}

HARM_H = {
    ("IRCoT 7B", "MedMCQA"): 0.596,
    ("IRCoT 32B", "MedMCQA"): 0.504,
    ("CRAG 7B", "MedMCQA"): 0.652,
    ("CRAG 32B", "MedMCQA"): 0.546,
    ("HippoRAG 7B", "MedMCQA"): 0.496,
    ("HippoRAG 32B", "MedMCQA"): 0.504,
    ("IRCoT 7B", "PubMedQA"): 0.559,
    ("IRCoT 32B", "PubMedQA"): 0.461,
    ("CRAG 7B", "PubMedQA"): 0.549,
    ("CRAG 32B", "PubMedQA"): 0.603,
    ("HippoRAG 7B", "PubMedQA"): 0.485,
    ("HippoRAG 32B", "PubMedQA"): 0.480,
}

OLD_AUARC_TABLE = {
    ("IRCoT 7B", "MedMCQA"): "0.452",
    ("IRCoT 32B", "MedMCQA"): "0.595",
    ("CRAG 7B", "MedMCQA"): "0.513",
    ("CRAG 32B", "MedMCQA"): "0.445",
    ("HippoRAG 7B", "MedMCQA"): "0.513",
    ("HippoRAG 32B", "MedMCQA"): "0.697",
    ("IRCoT 7B", "PubMedQA"): "0.607",
    ("IRCoT 32B", "PubMedQA"): "0.471",
    ("CRAG 7B", "PubMedQA"): "0.431",
    ("CRAG 32B", "PubMedQA"): "0.505",
    ("HippoRAG 7B", "PubMedQA"): "0.356",
    ("HippoRAG 32B", "PubMedQA"): "0.449",
}

# Colour encodes model family; line width + style encode model size
# (32B = thick solid, 7B = thin dashed). Per-family markers keep the
# curves separable in grayscale.
STYLES = {
    "IRCoT 7B": dict(color="#1f77b4", marker="o", linestyle="--", lw=1.3, ms=3.5, markeredgewidth=0.0, alpha=0.95),
    "IRCoT 32B": dict(color="#1f77b4", marker="o", linestyle="-", lw=2.4, ms=4.5, markeredgewidth=0.0, alpha=0.95),
    "CRAG 7B": dict(color="#d62728", marker="^", linestyle="--", lw=1.3, ms=3.5, markeredgewidth=0.0, alpha=0.95),
    "CRAG 32B": dict(color="#d62728", marker="^", linestyle="-", lw=2.4, ms=4.5, markeredgewidth=0.0, alpha=0.95),
    "HippoRAG 7B": dict(color="#2ca02c", marker="s", linestyle="--", lw=1.3, ms=3.5, markeredgewidth=0.0, alpha=0.95),
    "HippoRAG 32B": dict(color="#2ca02c", marker="s", linestyle="-", lw=2.4, ms=4.5, markeredgewidth=0.0, alpha=0.95),
}


def fmt3(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def threshold_curve(rows):
    if not rows:
        return None
    pairs = [(correct, conf) for _, correct, conf in rows]
    n = len(pairs)
    points = []
    for threshold in THRESHOLDS:
        answered = [(correct, conf) for correct, conf in pairs if conf is not None and conf >= threshold]
        if not answered:
            continue
        coverage = len(answered) / n
        accuracy = sum(correct for correct, _ in answered) / len(answered)
        points.append(
            {
                "threshold": threshold,
                "coverage": round(coverage, 6),
                "accuracy_answered": round(accuracy, 6),
                "answered_count": len(answered),
            }
        )
    if not points:
        return {"auarc": 0.0, "points": []}
    covs = np.array([p["coverage"] for p in points])
    accs = np.array([p["accuracy_answered"] for p in points])
    order = np.argsort(covs)
    auarc = float(abs(np.trapz(accs[order], covs[order])))
    return {"auarc": auarc, "points": points}


def competition_ranks_by_benchmark(scores):
    ranks = {}
    for bench in BENCHES:
        valid = {
            key: score
            for key, score in scores.items()
            if key[1] == bench and score is not None
        }
        ranks.update(
            {key: 1 + sum(other > score for other in valid.values()) for key, score in valid.items()}
        )
    return ranks


def bold_best(value_text, value, best):
    if value is None:
        return "---"
    return rf"\textbf{{{value_text}}}" if math.isclose(value, best, rel_tol=0, abs_tol=5e-7) else value_text


def table_fragments(auarc, ranks, correlations):
    best_med = max(v for (s, b), v in auarc.items() if b == "MedMCQA" and v is not None)
    best_pub = max(v for (s, b), v in auarc.items() if b == "PubMedQA" and v is not None)

    auarc_lines = [
        r"\multirow{2}{*}{IRCoT}     & 7B  & "
        + bold_best(fmt3(auarc[("IRCoT 7B", "MedMCQA")]), auarc[("IRCoT 7B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("IRCoT 7B", "PubMedQA")]), auarc[("IRCoT 7B", "PubMedQA")], best_pub)
        + r" \\",
        r"                            & 32B & "
        + bold_best(fmt3(auarc[("IRCoT 32B", "MedMCQA")]), auarc[("IRCoT 32B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("IRCoT 32B", "PubMedQA")]), auarc[("IRCoT 32B", "PubMedQA")], best_pub)
        + r" \\",
        r"\multirow{2}{*}{CRAG}      & 7B  & "
        + bold_best(fmt3(auarc[("CRAG 7B", "MedMCQA")]), auarc[("CRAG 7B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("CRAG 7B", "PubMedQA")]), auarc[("CRAG 7B", "PubMedQA")], best_pub)
        + r" \\",
        r"                            & 32B & "
        + bold_best(fmt3(auarc[("CRAG 32B", "MedMCQA")]), auarc[("CRAG 32B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("CRAG 32B", "PubMedQA")]), auarc[("CRAG 32B", "PubMedQA")], best_pub)
        + r" \\",
        r"\multirow{2}{*}{HippoRAG~2}& 7B  & "
        + bold_best(fmt3(auarc[("HippoRAG 7B", "MedMCQA")]), auarc[("HippoRAG 7B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("HippoRAG 7B", "PubMedQA")]), auarc[("HippoRAG 7B", "PubMedQA")], best_pub)
        + r" \\",
        r"                            & 32B & "
        + bold_best(fmt3(auarc[("HippoRAG 32B", "MedMCQA")]), auarc[("HippoRAG 32B", "MedMCQA")], best_med)
        + " & "
        + bold_best(fmt3(auarc[("HippoRAG 32B", "PubMedQA")]), auarc[("HippoRAG 32B", "PubMedQA")], best_pub)
        + r" \\",
    ]

    rank_lines = []
    for family, systems in [
        ("IRCoT", ["IRCoT 7B", "IRCoT 32B"]),
        ("CRAG", ["CRAG 7B", "CRAG 32B"]),
        ("HippoRAG~2", ["HippoRAG 7B", "HippoRAG 32B"]),
    ]:
        for i, system in enumerate(systems):
            row_name = rf"\multirow{{2}}{{*}}{{{family}}}" if i == 0 else " " * 24
            size = "7B" if system.endswith("7B") else "32B"
            vals = []
            for bench in BENCHES:
                key = (system, bench)
                acc_rank = ranks["accuracy"].get(key)
                auarc_rank = ranks["auarc"].get(key)
                vals.append("---" if acc_rank is None else (rf"\textbf{{{acc_rank}}}" if acc_rank == 1 else str(acc_rank)))
                vals.append("---" if auarc_rank is None else (rf"\textbf{{{auarc_rank}}}" if auarc_rank == 1 else str(auarc_rank)))
            rank_lines.append(f"{row_name} & {size} & " + " & ".join(vals) + r" \\")

    corr_lines = []
    for item in correlations:
        sig = "*" if item["p_value"] < 0.05 else r"\textrm{ns}"
        label = item["pair"].replace(" vs ", r" vs.\ ").replace("Sup%", r"Sup\%").replace("-H", r"$-H$")
        corr_lines.append(
            rf"{label:<18} & ${item['rho']:+.2f}$ & ${item['p_value']:.3f}$\;{sig} \\"
        )

    return {
        "auarc_table_rows": "\n".join(auarc_lines),
        "ranking_table_rows": "\n".join(rank_lines),
        "spearman_table_rows": "\n".join(corr_lines),
    }


def plot_curves(curves):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for bench in BENCHES:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for system in SYSTEM_ORDER:
            key = (system, bench)
            curve = curves.get(f"{system}|{bench}")
            if not curve or not curve["points"]:
                continue
            points = sorted(curve["points"], key=lambda p: p["coverage"])
            covs = [p["coverage"] for p in points]
            accs = [p["accuracy_answered"] for p in points]
            ax.plot(covs, accs, label=system, **STYLES[system])
        ax.set_xlabel("Coverage (fraction answered)", fontsize=8)
        ax.set_ylabel("Accuracy on answered", fontsize=8)
        ax.set_title(bench, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best", ncol=1, framealpha=0.7)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout(pad=0.4)
        stem = f"risk_coverage_{bench.lower()}_skip_none"
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rel_by_key = {(system, bench): rel for system, bench, rel in FILES}
    curves = {}
    auarc = {}
    for key in KEYS:
        rows = load_rows(rel_by_key[key])
        curve = threshold_curve(rows)
        curves[f"{key[0]}|{key[1]}"] = curve
        auarc[key] = None if curve is None else curve["auarc"]

    ranks = {
        "accuracy": competition_ranks_by_benchmark(ACCURACY_ACC),
        "auarc": competition_ranks_by_benchmark(auarc),
    }

    active = []
    matrix = []
    for key in KEYS:
        vals = [ACCURACY_F1.get(key), auarc.get(key), FAITHFUL_SUP.get(key), HARM_H.get(key)]
        if any(v is None for v in vals):
            continue
        active.append(key)
        matrix.append([vals[0], vals[1], vals[2], -vals[3]])
    matrix = np.array(matrix, dtype=float)

    pairs = [
        (0, 1, "F1 vs AUARC", "f1-auarc"),
        (0, 2, "F1 vs Sup%", "f1-faithful"),
        (0, 3, "F1 vs -H", "f1-harm"),
        (1, 2, "AUARC vs Sup%", "auarc-faithful"),
        (1, 3, "AUARC vs -H", "auarc-harm"),
        (2, 3, "Sup% vs -H", "faithful-harm"),
    ]
    correlations = []
    for i, j, label, key in pairs:
        rho, p_value = stats.spearmanr(matrix[:, i], matrix[:, j])
        correlations.append(
            {
                "pair": label,
                "key": key,
                "rho": round(float(rho), 4),
                "p_value": round(float(p_value), 4),
                "significant_05": bool(p_value < 0.05),
            }
        )

    changes = []
    for key in KEYS:
        old = OLD_AUARC_TABLE.get(key)
        new = fmt3(auarc[key])
        if old != new:
            changes.append(
                {
                    "system": key[0],
                    "benchmark": key[1],
                    "old_table_value": old,
                    "new_skip_none_value": new,
                    "exact_skip_none_value": None if auarc[key] is None else round(auarc[key], 6),
                }
            )

    fragments = table_fragments(auarc, ranks, correlations)
    plot_curves(curves)

    out = {
        "method": "threshold-sweep [0.0,0.1,...,1.0], confidence None -> abstain, zero-answer thresholds skipped",
        "auarc": {f"{s}|{b}": None if v is None else round(v, 6) for (s, b), v in auarc.items()},
        "ranks": {
            "accuracy": {f"{s}|{b}": rank for (s, b), rank in ranks["accuracy"].items()},
            "auarc": {f"{s}|{b}": rank for (s, b), rank in ranks["auarc"].items()},
        },
        "curves": curves,
        "score_table_for_spearman": [
            {
                "system": system,
                "benchmark": bench,
                "accuracy_f1": matrix[i, 0],
                "auarc": matrix[i, 1],
                "faithful_sup_pct": matrix[i, 2],
                "neg_harm_h": matrix[i, 3],
            }
            for i, (system, bench) in enumerate(active)
        ],
        "spearman_correlations": correlations,
        "changed_auarc_table_cells": changes,
        "fragments": fragments,
    }

    with (RESULTS_DIR / "skip_none_analysis.json").open("w") as f:
        json.dump(out, f, indent=2)

    with (RESULTS_DIR / "skip_none_changed_cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "benchmark", "old_table_value", "new_skip_none_value", "exact_skip_none_value"],
        )
        writer.writeheader()
        writer.writerows(changes)

    with (RESULTS_DIR / "skip_none_paper_fragments.tex").open("w") as f:
        for name, text in fragments.items():
            f.write(f"% {name}\n{text}\n\n")

    print(f"Wrote {RESULTS_DIR / 'skip_none_analysis.json'}")
    print(f"Wrote {RESULTS_DIR / 'skip_none_changed_cells.csv'}")
    print(f"Wrote {RESULTS_DIR / 'skip_none_paper_fragments.tex'}")
    print(f"Wrote figures under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
