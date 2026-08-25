#!/usr/bin/env python3
"""Create simple tables and SVG plots for copied Phase 1 result bundles.

This script intentionally uses only the Python standard library so it works on
the local machine without installing pandas/matplotlib.
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable


PUBMEDQA_LABELS = ("yes", "no", "maybe", "unknown")


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def find_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}
            )
    return sorted(files)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def pct(value: float | int | None) -> str:
    if value is None:
        return ""
    try:
        return f"{100 * float(value):.1f}%"
    except Exception:
        return ""


def num(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def row_get_confidence(row: dict) -> float | None:
    for field in ("self_reported_confidence", "confidence", "parsed_confidence", "model_confidence"):
        value = row.get(field)
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            return float(value)
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if 0 <= parsed <= 1:
                return parsed
    return None


def row_gold(row: dict) -> str:
    value = row.get("gold_short_answer")
    if value:
        return str(value).lower()
    golds = row.get("gold_answers")
    if isinstance(golds, list) and golds:
        return str(golds[0]).lower()
    return ""


def row_pred(row: dict) -> str:
    value = row.get("parsed_label")
    if value:
        return str(value).lower()
    text = str(row.get("predicted_answer", "")).lower()
    for label in ("yes", "no", "maybe"):
        if f"answer: {label}" in text:
            return label
    words = text.replace(":", " ").replace(".", " ").replace(",", " ").split()
    for label in ("yes", "no", "maybe"):
        if label in words:
            return label
    return "unknown" if text else ""


def classify_json_file(path: Path, obj: dict) -> str:
    name = path.name.lower()
    if "risk_coverage" in obj or name.startswith("risk_"):
        return "risk_json"
    if "confusion" in obj and obj.get("benchmark") == "pubmedqa":
        return "pubmedqa_metrics"
    if obj.get("benchmark") == "hotpotqa" and ("exact_match" in obj or "f1" in obj):
        return "hotpotqa_metrics"
    if "evidence_exact_quote_count" in obj or "supporting_evidence_present_rate" in obj:
        return "evidence_compliance"
    if "harm_cost_summary" in obj:
        return "manifest_or_postprocess"
    if "avg_base_harm_cost" in obj or "severity_labeled_count" in obj:
        return "harm_summary"
    return "json"


def classify_jsonl_file(path: Path, rows: list[dict]) -> str:
    name = path.name.lower()
    if not rows:
        return "jsonl_empty"
    keys = set().union(*(row.keys() for row in rows[:10]))
    if "manual_review_label" in keys or "manual_faithfulness_label" in keys or name.startswith("manual_review"):
        return "faithfulness_review"
    if "severity_class_a_b_c" in {key.lower() for key in keys} or "severity_template" in name:
        return "severity_template"
    if "base_harm_cost" in keys:
        return "harm_scaffold"
    if "predicted_answer" in keys and ("question_id" in keys or "id" in keys):
        return "model_outputs"
    return "jsonl"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = ["file", "kind", "benchmark", "rows"]
    seen = set()
    fields = []
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_frame(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )


def save_bar_svg(path: Path, title: str, values: dict[str, float | int], ylabel: str = "count") -> None:
    if not values:
        return
    labels = list(values)
    nums = [float(values[label] or 0) for label in labels]
    width = max(560, 90 * len(labels) + 120)
    height = 380
    left, right, top, bottom = 70, 25, 55, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = max(nums) if nums else 1
    max_y = max_y if max_y > 0 else 1
    bar_w = chart_w / max(1, len(labels)) * 0.68
    parts = [
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<text x="18" y="{top + chart_h/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + chart_h/2:.1f})" font-family="Arial" font-size="12">{html.escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>',
    ]
    for tick in range(5):
        yv = max_y * tick / 4
        y = top + chart_h - (yv / max_y) * chart_h
        parts.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.2g}</text>')
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f97316", "#14b8a6"]
    for idx, (label, value) in enumerate(zip(labels, nums)):
        cx = left + chart_w * (idx + 0.5) / len(labels)
        h = (value / max_y) * chart_h
        x = cx - bar_w / 2
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:g}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + chart_h + 22:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(label)}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_frame(width, height, "\n".join(parts)), encoding="utf-8")


def save_line_svg(path: Path, title: str, series: list[tuple[str, list[tuple[float, float | None]]]]) -> None:
    series = [(name, [(x, y) for x, y in points if y is not None]) for name, points in series]
    series = [(name, points) for name, points in series if points]
    if not series:
        return
    width, height = 720, 430
    left, right, top, bottom = 70, 165, 55, 65
    chart_w = width - left - right
    chart_h = height - top - bottom
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(1.0, max(ys))
    if max_x == min_x:
        max_x += 1
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0d9488"]

    def px(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * chart_w

    def py(y: float) -> float:
        return top + chart_h - (y - min_y) / (max_y - min_y) * chart_h

    parts = [
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>',
        f'<text x="{left + chart_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">threshold</text>',
    ]
    for tick in range(6):
        xval = min_x + (max_x - min_x) * tick / 5
        x = px(xval)
        parts.append(f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" y2="{top + chart_h + 4}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{xval:.1f}</text>')
    for tick in range(6):
        yval = min_y + (max_y - min_y) * tick / 5
        y = py(yval)
        parts.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yval:.1f}</text>')
    for idx, (name, points) in enumerate(series):
        color = colors[idx % len(colors)]
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        ly = top + 18 + idx * 22
        parts.append(f'<line x1="{left + chart_w + 28}" y1="{ly}" x2="{left + chart_w + 52}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + chart_w + 58}" y="{ly+4}" font-family="Arial" font-size="12">{html.escape(name[:22])}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_frame(width, height, "\n".join(parts)), encoding="utf-8")


def save_confusion_svg(path: Path, title: str, confusion: dict) -> None:
    labels = ["yes", "no", "maybe", "unknown"]
    rows = ["yes", "no", "maybe"]
    cell = 72
    left, top = 100, 70
    width = left + cell * len(labels) + 45
    height = top + cell * len(rows) + 70
    values = [[int(confusion.get(g, {}).get(p, 0)) for p in labels] for g in rows]
    max_v = max([v for row in values for v in row] or [1]) or 1
    parts = [
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<text x="22" y="{top + cell*len(rows)/2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + cell*len(rows)/2:.1f})" font-family="Arial" font-size="12">gold</text>',
        f'<text x="{left + cell*len(labels)/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">prediction</text>',
    ]
    for j, label in enumerate(labels):
        parts.append(f'<text x="{left + j*cell + cell/2:.1f}" y="{top-14}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    for i, gold in enumerate(rows):
        parts.append(f'<text x="{left-12}" y="{top + i*cell + cell/2 + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{gold}</text>')
        for j, pred in enumerate(labels):
            value = values[i][j]
            shade = 245 - int(175 * value / max_v)
            fill = f"rgb({shade},{shade + 5},{255})"
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x + cell/2:.1f}" y="{y + cell/2 + 5:.1f}" text-anchor="middle" font-family="Arial" font-size="15">{value}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_frame(width, height, "\n".join(parts)), encoding="utf-8")


def summarize_model_outputs(path: Path, rows: list[dict], base: Path, plot_dir: Path) -> dict:
    benchmark = str(rows[0].get("benchmark", "")) if rows else ""
    preds = [row_pred(row) for row in rows]
    golds = [row_gold(row) for row in rows]
    confs = [row_get_confidence(row) for row in rows]
    available_confs = [c for c in confs if c is not None]
    correct = [g == p for g, p in zip(golds, preds) if g and p]
    pred_counts = dict(collections.Counter(preds))
    stem = path.stem[:80].replace("/", "_")
    save_bar_svg(plot_dir / f"labels_{stem}.svg", f"Predicted labels: {path.name}", pred_counts, "rows")
    if available_confs:
        bins = collections.Counter(f"{min(9, int(c * 10)) / 10:.1f}-{(min(9, int(c * 10)) + 1) / 10:.1f}" for c in available_confs)
        save_bar_svg(plot_dir / f"confidence_{stem}.svg", f"Confidence histogram: {path.name}", dict(sorted(bins.items())), "rows")
    return {
        "file": rel(path, base),
        "kind": "model_outputs",
        "benchmark": benchmark,
        "rows": len(rows),
        "accuracy_from_rows": sum(correct) / len(correct) if correct else None,
        "confidence_available": len(available_confs),
        "avg_confidence": sum(available_confs) / len(available_confs) if available_confs else None,
        "prediction_counts": json.dumps(pred_counts, sort_keys=True),
    }


def analyze(root_paths: list[Path], out_dir: Path, base: Path) -> tuple[list[dict], list[Path]]:
    files = find_files(root_paths)
    summaries: list[dict] = []
    risk_series: list[tuple[str, list[tuple[float, float | None]], list[tuple[float, float | None]]]] = []
    faithfulness_backlog = {"filled": 0, "missing": 0}
    severity_backlog = {"filled": 0, "missing": 0}
    plot_dir = out_dir / "plots"

    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            obj = load_json(path)
            if not isinstance(obj, dict):
                continue
            kind = classify_json_file(path, obj)
            row = {"file": rel(path, base), "kind": kind, "benchmark": obj.get("benchmark", ""), "rows": obj.get("num_examples") or obj.get("rows") or ""}
            for key in (
                "accuracy",
                "macro_f1",
                "exact_match",
                "f1",
                "avg_retrieval_recall",
                "avg_retrieval_recall_pmid",
                "error_count",
                "avg_confidence",
                "supporting_evidence_present_rate",
                "supporting_evidence_found_flag_rate",
                "supporting_evidence_verified_by_exporter_rate",
                "evidence_exact_quote_rate",
                "avg_base_harm_cost",
                "avg_confidence_weighted_base_harm_cost",
                "severity_labeled_count",
                "severity_missing_count",
            ):
                if key in obj:
                    row[key] = obj.get(key)
            summaries.append(row)
            if kind == "pubmedqa_metrics":
                if isinstance(obj.get("prediction_counts"), dict):
                    save_bar_svg(plot_dir / f"prediction_counts_{path.stem}.svg", f"Prediction counts: {path.name}", obj["prediction_counts"], "rows")
                if isinstance(obj.get("confusion"), dict):
                    save_confusion_svg(plot_dir / f"confusion_{path.stem}.svg", f"Confusion: {path.name}", obj["confusion"])
            continue
        if suffix == ".csv" and path.name.lower().startswith("risk_"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            coverage_points = []
            accuracy_points = []
            for row in rows:
                try:
                    threshold = float(row.get("threshold", ""))
                except ValueError:
                    continue
                coverage_points.append((threshold, float(row["coverage"]) if row.get("coverage") else None))
                acc = row.get("accuracy_answered") or row.get("exact_match_answered")
                accuracy_points.append((threshold, float(acc) if acc not in (None, "") else None))
            risk_series.append((path.stem, coverage_points, accuracy_points))
            summaries.append({"file": rel(path, base), "kind": "risk_csv", "rows": len(rows)})
            continue
        if suffix == ".jsonl":
            rows = load_jsonl(path)
            kind = classify_jsonl_file(path, rows)
            if kind == "model_outputs":
                summaries.append(summarize_model_outputs(path, rows, base, plot_dir))
            elif kind == "faithfulness_review":
                missing = sum(
                    not (
                        str(row.get("manual_review_label", "")).strip()
                        or str(row.get("manual_faithfulness_label", "")).strip()
                    )
                    for row in rows
                )
                filled = len(rows) - missing
                faithfulness_backlog["missing"] += missing
                faithfulness_backlog["filled"] += filled
                summaries.append({"file": rel(path, base), "kind": kind, "rows": len(rows), "manual_labels_filled": filled, "manual_labels_missing": missing})
            elif kind == "severity_template":
                missing = sum(not str(row.get("severity_class_A_B_C", "")).strip() for row in rows)
                filled = len(rows) - missing
                severity_backlog["missing"] += missing
                severity_backlog["filled"] += filled
                summaries.append({"file": rel(path, base), "kind": kind, "rows": len(rows), "severity_labels_filled": filled, "severity_labels_missing": missing})
            elif kind == "harm_scaffold":
                values = [row.get("base_harm_cost") for row in rows if isinstance(row.get("base_harm_cost"), (int, float))]
                summaries.append({"file": rel(path, base), "kind": kind, "rows": len(rows), "avg_base_harm_cost": sum(values) / len(values) if values else None})
            else:
                summaries.append({"file": rel(path, base), "kind": kind, "rows": len(rows)})

    line_series = []
    for name, coverage, accuracy in risk_series:
        line_series.append((f"{name}: coverage", coverage))
        line_series.append((f"{name}: answered acc", accuracy))
    save_line_svg(plot_dir / "risk_coverage_curves.svg", "Risk coverage curves", line_series)
    if faithfulness_backlog["filled"] or faithfulness_backlog["missing"]:
        save_bar_svg(plot_dir / "faithfulness_labeling_backlog.svg", "Faithfulness manual review backlog", faithfulness_backlog, "rows")
    if severity_backlog["filled"] or severity_backlog["missing"]:
        save_bar_svg(plot_dir / "severity_labeling_backlog.svg", "Severity labeling backlog", severity_backlog, "rows")
    return summaries, files


def report_markdown(summaries: list[dict], files: list[Path], base: Path, out_dir: Path) -> str:
    metrics = [row for row in summaries if "metrics" in str(row.get("kind", "")) or row.get("kind") in {"model_outputs", "evidence_compliance", "harm_summary", "risk_csv"}]
    labels = [row for row in summaries if row.get("kind") in {"faithfulness_review", "severity_template"}]
    lines = [
        "# Result Bundle Analysis",
        "",
        f"Files scanned: {len(files)}",
        f"Generated plots: `{rel(out_dir / 'plots', base)}`",
        "",
        "## Main Result Files",
        "",
        "| file | kind | rows | key numbers |",
        "|---|---:|---:|---|",
    ]
    for row in metrics:
        keys = []
        for key in (
            "accuracy",
            "macro_f1",
            "exact_match",
            "f1",
            "avg_retrieval_recall",
            "accuracy_from_rows",
            "avg_confidence",
            "supporting_evidence_present_rate",
            "supporting_evidence_found_flag_rate",
            "evidence_exact_quote_rate",
            "avg_base_harm_cost",
        ):
            if key in row and row[key] not in (None, ""):
                value = pct(row[key]) if "rate" in key or key in {"accuracy", "macro_f1", "exact_match", "f1", "avg_retrieval_recall", "accuracy_from_rows"} else num(row[key])
                keys.append(f"{key}={value}")
        lines.append(f"| `{row.get('file', '')}` | {row.get('kind', '')} | {row.get('rows', '')} | {'; '.join(keys)} |")
    lines.extend(["", "## What Still Needs Labeling", "", "| file | kind | filled | missing |", "|---|---:|---:|---:|"])
    for row in labels:
        filled = row.get("manual_labels_filled", row.get("severity_labels_filled", ""))
        missing = row.get("manual_labels_missing", row.get("severity_labels_missing", ""))
        lines.append(f"| `{row.get('file', '')}` | {row.get('kind', '')} | {filled} | {missing} |")
    lines.extend(
        [
            "",
            "## Reading The Plots",
            "",
            "- `risk_coverage_curves.svg`: as the confidence threshold rises, coverage should fall. If answered accuracy rises, confidence is useful for abstention.",
            "- `confusion_*.svg`: rows are gold labels and columns are predictions. A strong diagonal is good; off-diagonal blocks show systematic mistakes.",
            "- `prediction_counts_*.svg` and `labels_*.svg`: shows whether the model overuses one answer, such as `maybe`.",
            "- `confidence_*.svg`: shows whether confidence is spread out enough to support risk-coverage analysis.",
            "- `*_labeling_backlog.svg`: shows how much manual annotation remains.",
            "",
            "## Metric Meanings",
            "",
            "- PubMedQA `accuracy`: fraction of examples where parsed prediction equals gold `yes/no/maybe`.",
            "- PubMedQA `macro_f1`: F1 computed separately for `yes`, `no`, and `maybe`, then averaged. It penalizes label imbalance.",
            "- HotpotQA `exact_match`: answer string exactly matches a gold answer after normalization.",
            "- HotpotQA `f1`: token overlap F1 between prediction and best matching gold answer.",
            "- `avg_retrieval_recall`: average fraction of gold evidence recovered in retrieved passages.",
            "- Risk `coverage`: fraction of rows still answered at a confidence threshold.",
            "- Risk `accuracy_answered`: accuracy only among rows with confidence above the threshold.",
            "- Evidence compliance: checks whether the model supplied evidence and whether that evidence is an exact/normalized substring of retrieved passages. It is a quote-compliance check, not a semantic faithfulness proof.",
            "- Harm cost: an experimental penalty for answer errors. Severity-weighted harm is unavailable until A/B/C severity labels are filled.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("remote_results"), Path("results"), Path("annotations"), Path("manifests")])
    parser.add_argument("--out-dir", type=Path, default=Path("reports/result_analysis"))
    args = parser.parse_args()

    base = Path.cwd()
    roots = [path if path.is_absolute() else base / path for path in args.roots]
    out_dir = args.out_dir if args.out_dir.is_absolute() else base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, files = analyze(roots, out_dir, base)
    write_csv(out_dir / "result_files_summary.csv", summaries)
    (out_dir / "report.md").write_text(report_markdown(summaries, files, base, out_dir), encoding="utf-8")
    print(f"Scanned {len(files)} files")
    print(f"Wrote {out_dir / 'result_files_summary.csv'}")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote plots under {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
