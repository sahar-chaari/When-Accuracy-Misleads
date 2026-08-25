#!/usr/bin/env python3
"""Local CSV labeling interface for HippoRAG severity review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "annotations"
    / "harm_cost"
    / "review_workbooks"
    / "severity_labeling.csv"
)

REQUIRED_COLUMNS = [
    "review_priority",
    "question_id",
    "question",
    "gold_answer",
    "severity_class_A_B_C",
    "severity_rationale",
    "manual_checked",
    "notes",
]

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Severity Labeling</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f6;
      --text: #1f2933;
      --muted: #627084;
      --border: #d8dee8;
      --focus: #246bfe;
      --a: #256d4f;
      --a-bg: #e6f4ef;
      --b: #9a5b00;
      --b-bg: #fff3d8;
      --c: #af2f2f;
      --c-bg: #fde6e3;
      --done: #1d7a54;
      --shadow: 0 10px 28px rgba(21, 31, 46, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }

    button,
    textarea,
    input {
      font: inherit;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
    }

    .sidebar {
      border-right: 1px solid var(--border);
      background: #fbfcfe;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .side-head {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--border);
    }

    .title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }

    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 760;
    }

    .status {
      min-height: 24px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      text-align: right;
    }

    .progress-wrap {
      display: grid;
      gap: 7px;
    }

    .progress-meta {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 13px;
    }

    .progress {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--surface-2);
    }

    .progress > span {
      display: block;
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, #246bfe, #1d7a54);
      transition: width 180ms ease;
    }

    .filters {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 14px;
    }

    .filter {
      border: 1px solid var(--border);
      background: var(--surface);
      border-radius: 8px;
      min-height: 34px;
      color: var(--muted);
      cursor: pointer;
    }

    .filter.active {
      color: #ffffff;
      border-color: var(--focus);
      background: var(--focus);
    }

    .row-list {
      overflow: auto;
      padding: 10px;
      display: grid;
      gap: 8px;
    }

    .row-button {
      width: 100%;
      min-height: 76px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 11px;
      background: var(--surface);
      text-align: left;
      cursor: pointer;
      display: grid;
      gap: 7px;
    }

    .row-button:hover,
    .row-button.active {
      border-color: var(--focus);
      box-shadow: 0 0 0 2px rgba(36, 107, 254, 0.12);
    }

    .row-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .row-question {
      font-size: 13px;
      line-height: 1.35;
      color: var(--text);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .main {
      min-width: 0;
      padding: 24px;
      display: flex;
      align-items: stretch;
      justify-content: center;
    }

    .workspace {
      width: min(100%, 1040px);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 16px;
    }

    .topbar,
    .question-panel,
    .editor-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .topbar {
      min-height: 64px;
      padding: 14px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .meta {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .pill,
    .label-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .pill.priority {
      color: #7c3e00;
      border-color: #f3bf70;
      background: #fff2dc;
    }

    .label-badge.A {
      color: var(--a);
      border-color: #a7d8c7;
      background: var(--a-bg);
    }

    .label-badge.B {
      color: var(--b);
      border-color: #e5bd72;
      background: var(--b-bg);
    }

    .label-badge.C {
      color: var(--c);
      border-color: #efaaa4;
      background: var(--c-bg);
    }

    .nav {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .nav button,
    .save-button {
      min-height: 36px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      padding: 0 12px;
      cursor: pointer;
    }

    .nav button:hover,
    .save-button:hover {
      border-color: var(--focus);
      color: var(--focus);
    }

    .question-panel {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
      min-height: 300px;
    }

    .question {
      margin: 0;
      font-size: 26px;
      line-height: 1.24;
      font-weight: 760;
      overflow-wrap: anywhere;
    }

    .answers {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .answer-box {
      display: grid;
      gap: 6px;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfe;
    }

    .answer-box span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 740;
      text-transform: uppercase;
    }

    .answer-box strong {
      font-size: 18px;
      line-height: 1.35;
    }

    .labels {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .label-button {
      min-height: 118px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      background: var(--surface);
      cursor: pointer;
      display: grid;
      align-content: start;
      gap: 8px;
      text-align: left;
    }

    .label-button:hover,
    .label-button.selected {
      border-color: var(--focus);
      box-shadow: 0 0 0 2px rgba(36, 107, 254, 0.12);
    }

    .label-button .letter {
      font-size: 24px;
      font-weight: 820;
      line-height: 1;
    }

    .label-button .meaning {
      font-size: 13px;
      line-height: 1.36;
      color: var(--muted);
    }

    .label-button[data-label="A"] .letter {
      color: var(--a);
    }

    .label-button[data-label="B"] .letter {
      color: var(--b);
    }

    .label-button[data-label="C"] .letter {
      color: var(--c);
    }

    .editor-panel {
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 740;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 78px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 11px;
      color: var(--text);
      background: #fbfcfe;
      line-height: 1.4;
      text-transform: none;
      font-weight: 500;
    }

    textarea:focus,
    button:focus-visible {
      outline: 2px solid rgba(36, 107, 254, 0.28);
      outline-offset: 2px;
    }

    .checked {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
      text-transform: none;
      white-space: nowrap;
    }

    .checked input {
      width: 18px;
      height: 18px;
      accent-color: var(--focus);
    }

    .empty {
      padding: 24px;
      color: var(--muted);
    }

    @media (max-width: 860px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        min-height: auto;
        max-height: 42vh;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      .main {
        padding: 14px;
      }

      .topbar,
      .editor-panel {
        grid-template-columns: 1fr;
        flex-direction: column;
        align-items: stretch;
      }

      .labels,
      .answers,
      .editor-panel {
        grid-template-columns: 1fr;
      }

      .question {
        font-size: 22px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="side-head">
        <div class="title-row">
          <h1>Severity Labeling</h1>
          <div id="status" class="status">Loading</div>
        </div>
        <div class="progress-wrap">
          <div class="progress-meta">
            <span id="progressText">0 / 0 labeled</span>
            <span id="remainingText">0 left</span>
          </div>
          <div class="progress"><span id="progressBar"></span></div>
        </div>
        <div class="filters">
          <button class="filter active" data-filter="all">All</button>
          <button class="filter" data-filter="open">Open</button>
          <button class="filter" data-filter="priority">Review</button>
        </div>
      </div>
      <div id="rowList" class="row-list"></div>
    </aside>
    <main class="main">
      <section class="workspace">
        <div class="topbar">
          <div class="meta" id="meta"></div>
          <div class="nav">
            <button id="prevButton" type="button">Previous</button>
            <button id="nextOpenButton" type="button">Next Open</button>
            <button id="nextButton" type="button">Next</button>
          </div>
        </div>
        <section class="question-panel" id="questionPanel"></section>
        <section class="editor-panel">
          <label>Rationale
            <textarea id="rationaleInput"></textarea>
          </label>
          <label>Notes
            <textarea id="notesInput"></textarea>
          </label>
          <div>
            <label class="checked">
              <input type="checkbox" id="manualCheckedInput">
              Manual checked
            </label>
            <button class="save-button" id="saveDetailsButton" type="button">Save Details</button>
          </div>
        </section>
      </section>
    </main>
  </div>
  <script>
    const state = {
      rows: [],
      index: 0,
      filter: "all",
      saveTimer: null
    };

    const els = {
      status: document.getElementById("status"),
      progressText: document.getElementById("progressText"),
      remainingText: document.getElementById("remainingText"),
      progressBar: document.getElementById("progressBar"),
      rowList: document.getElementById("rowList"),
      meta: document.getElementById("meta"),
      questionPanel: document.getElementById("questionPanel"),
      rationaleInput: document.getElementById("rationaleInput"),
      notesInput: document.getElementById("notesInput"),
      manualCheckedInput: document.getElementById("manualCheckedInput"),
      saveDetailsButton: document.getElementById("saveDetailsButton"),
      prevButton: document.getElementById("prevButton"),
      nextButton: document.getElementById("nextButton"),
      nextOpenButton: document.getElementById("nextOpenButton")
    };

    function clean(value) {
      return (value || "").toString();
    }

    function isLabeled(row) {
      return ["A", "B", "C"].includes(clean(row.severity_class_A_B_C).trim());
    }

    function checkedValue(row) {
      return clean(row.manual_checked).toLowerCase() === "true";
    }

    function setStatus(text) {
      els.status.textContent = text;
    }

    function labelBadge(label) {
      const value = clean(label).trim();
      const shown = value || "-";
      const cls = value ? ` ${value}` : "";
      return `<span class="label-badge${cls}">${shown}</span>`;
    }

    function filteredRows() {
      return state.rows
        .map((row, index) => ({ row, index }))
        .filter(({ row }) => {
          if (state.filter === "open") return !isLabeled(row);
          if (state.filter === "priority") {
            const priority = clean(row.review_priority);
            return priority === "review_first" || priority === "error";
          }
          return true;
        });
    }

    function renderProgress() {
      const total = state.rows.length;
      const done = state.rows.filter(isLabeled).length;
      const pct = total ? Math.round((done / total) * 100) : 0;
      els.progressText.textContent = `${done} / ${total} labeled`;
      els.remainingText.textContent = `${Math.max(total - done, 0)} left`;
      els.progressBar.style.width = `${pct}%`;
    }

    function renderList() {
      const items = filteredRows();
      if (!items.length) {
        els.rowList.innerHTML = `<div class="empty">No rows</div>`;
        return;
      }
      els.rowList.innerHTML = items.map(({ row, index }) => {
        const active = index === state.index ? " active" : "";
        const priority = clean(row.review_priority) === "review_first" ? "Review" : clean(row.review_priority);
        const question = clean(row.question).replace(/[&<>"']/g, (ch) => ({
          "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
        }[ch]));
        return `
          <button class="row-button${active}" data-index="${index}" type="button">
            <div class="row-top">
              <span>${index + 1}. ${priority}</span>
              ${labelBadge(row.severity_class_A_B_C)}
            </div>
            <div class="row-question">${question}</div>
          </button>
        `;
      }).join("");

      els.rowList.querySelectorAll(".row-button").forEach((button) => {
        button.addEventListener("click", () => {
          state.index = Number(button.dataset.index);
          render();
        });
      });
    }

    function renderCurrent() {
      const row = state.rows[state.index];
      if (!row) {
        els.meta.innerHTML = "";
        els.questionPanel.innerHTML = `<div class="empty">No row selected</div>`;
        return;
      }
      const priority = clean(row.review_priority) === "review_first" ? "Review first" : clean(row.review_priority);
      els.meta.innerHTML = `
        <span class="pill priority">${priority}</span>
        <span class="pill">Row ${state.index + 1} of ${state.rows.length}</span>
        <span class="pill">PMID ${clean(row.question_id)}</span>
        ${labelBadge(row.severity_class_A_B_C)}
      `;
      const selected = clean(row.severity_class_A_B_C).trim();
      const predicted = clean(row.predicted_answer);
      const predictedBox = predicted ? `
          <div class="answer-box">
            <span>Predicted Answer</span>
            <strong>${predicted}</strong>
          </div>
        ` : "";
      els.questionPanel.innerHTML = `
        <p class="question">${clean(row.question)}</p>
        <div class="answers">
          <div class="answer-box">
            <span>Gold Answer</span>
            <strong>${clean(row.gold_answer) || "-"}</strong>
          </div>
          ${predictedBox}
        </div>
        <div class="labels">
          <button class="label-button ${selected === "A" ? "selected" : ""}" data-label="A" type="button">
            <span class="letter">A</span>
            <span class="meaning">No injury or health damage possible</span>
          </button>
          <button class="label-button ${selected === "B" ? "selected" : ""}" data-label="B" type="button">
            <span class="letter">B</span>
            <span class="meaning">Non-serious injury possible</span>
          </button>
          <button class="label-button ${selected === "C" ? "selected" : ""}" data-label="C" type="button">
            <span class="letter">C</span>
            <span class="meaning">Serious injury or death possible</span>
          </button>
        </div>
      `;
      els.rationaleInput.value = clean(row.severity_rationale);
      els.notesInput.value = clean(row.notes);
      els.manualCheckedInput.checked = checkedValue(row);

      els.questionPanel.querySelectorAll(".label-button").forEach((button) => {
        button.addEventListener("click", () => saveLabel(button.dataset.label));
      });
    }

    function render() {
      renderProgress();
      renderList();
      renderCurrent();
      const active = els.rowList.querySelector(".row-button.active");
      if (active) active.scrollIntoView({ block: "nearest" });
    }

    async function savePatch(patch, nextIndex = null) {
      setStatus("Saving");
      const response = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row_index: state.index, patch })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Save failed: ${response.status}`);
      }
      const data = await response.json();
      state.rows[data.row_index] = data.row;
      if (nextIndex !== null) state.index = nextIndex;
      setStatus("Saved");
      render();
    }

    async function saveLabel(label) {
      const next = nextOpenIndex(state.index + 1);
      await savePatch({
        severity_class_A_B_C: label,
        manual_checked: "True"
      }, next === -1 ? state.index : next);
    }

    function saveDetails() {
      savePatch({
        severity_rationale: els.rationaleInput.value,
        notes: els.notesInput.value,
        manual_checked: els.manualCheckedInput.checked ? "True" : "False"
      }).catch((error) => setStatus(error.message));
    }

    function scheduleDetailsSave() {
      window.clearTimeout(state.saveTimer);
      state.saveTimer = window.setTimeout(saveDetails, 500);
    }

    function clampIndex(index) {
      return Math.min(Math.max(index, 0), Math.max(state.rows.length - 1, 0));
    }

    function nextOpenIndex(start) {
      for (let i = start; i < state.rows.length; i += 1) {
        if (!isLabeled(state.rows[i])) return i;
      }
      for (let i = 0; i < start; i += 1) {
        if (!isLabeled(state.rows[i])) return i;
      }
      return -1;
    }

    async function loadRows() {
      const response = await fetch("/api/rows");
      if (!response.ok) throw new Error(`Load failed: ${response.status}`);
      const data = await response.json();
      state.rows = data.rows;
      const firstOpen = nextOpenIndex(0);
      state.index = firstOpen === -1 ? 0 : firstOpen;
      setStatus("Ready");
      render();
    }

    document.querySelectorAll(".filter").forEach((button) => {
      button.addEventListener("click", () => {
        state.filter = button.dataset.filter;
        document.querySelectorAll(".filter").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        renderList();
      });
    });

    els.prevButton.addEventListener("click", () => {
      state.index = clampIndex(state.index - 1);
      render();
    });
    els.nextButton.addEventListener("click", () => {
      state.index = clampIndex(state.index + 1);
      render();
    });
    els.nextOpenButton.addEventListener("click", () => {
      const next = nextOpenIndex(state.index + 1);
      if (next !== -1) {
        state.index = next;
        render();
      }
    });
    els.saveDetailsButton.addEventListener("click", saveDetails);
    els.rationaleInput.addEventListener("input", scheduleDetailsSave);
    els.notesInput.addEventListener("input", scheduleDetailsSave);
    els.manualCheckedInput.addEventListener("change", saveDetails);

    window.addEventListener("keydown", (event) => {
      const tag = event.target.tagName.toLowerCase();
      if (tag === "textarea" || tag === "input") return;
      const key = event.key.toUpperCase();
      if (["A", "B", "C"].includes(key)) {
        event.preventDefault();
        saveLabel(key).catch((error) => setStatus(error.message));
      } else if (event.key === "ArrowRight") {
        state.index = clampIndex(state.index + 1);
        render();
      } else if (event.key === "ArrowLeft") {
        state.index = clampIndex(state.index - 1);
        render();
      }
    });

    loadRows().catch((error) => setStatus(error.message));
  </script>
</body>
</html>
"""


class CsvStore:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        self.backup_path: Path | None = None

    def read(self) -> tuple[list[str], list[dict[str, str]]]:
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        for column in REQUIRED_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
                for row in rows:
                    row[column] = ""
        return fieldnames, rows

    def write(self, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backup_path is None and self.csv_path.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.backup_path = self.csv_path.with_name(f"{self.csv_path.name}.bak_{stamp}")
            shutil.copy2(self.csv_path, self.backup_path)

        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=str(self.csv_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(self.csv_path)

    def update_row(self, row_index: int, patch: dict[str, str]) -> dict[str, object]:
        with self.lock:
            fieldnames, rows = self.read()
            if row_index < 0 or row_index >= len(rows):
                raise ValueError(f"row_index out of range: {row_index}")

            for key in patch:
                if key not in fieldnames:
                    fieldnames.append(key)
                    for row in rows:
                        row.setdefault(key, "")

            label = patch.get("severity_class_A_B_C")
            if label is not None:
                label = label.strip().upper()
                if label and label not in {"A", "B", "C"}:
                    raise ValueError("severity_class_A_B_C must be A, B, C, or blank")
                patch["severity_class_A_B_C"] = label

            for key, value in patch.items():
                rows[row_index][key] = "" if value is None else str(value)

            self.write(fieldnames, rows)
            return {
                "row_index": row_index,
                "row": rows[row_index],
                "backup_path": str(self.backup_path) if self.backup_path else "",
            }


class Handler(BaseHTTPRequestHandler):
    store: CsvStore

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self.send_html()
            return
        if path == "/api/rows":
            try:
                fieldnames, rows = self.store.read()
                self.send_json(
                    {
                        "csv_path": str(self.store.csv_path),
                        "fieldnames": fieldnames,
                        "rows": rows,
                    }
                )
            except Exception as exc:  # pragma: no cover - surfaced in browser
                self.send_json({"error": str(exc)}, status=500)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            row_index = int(payload["row_index"])
            patch = dict(payload.get("patch") or {})
            self.send_json(self.store.update_row(row_index, patch))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file to label.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")

    Handler.store = CsvStore(csv_path)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Severity labeling UI: http://{args.host}:{args.port}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
