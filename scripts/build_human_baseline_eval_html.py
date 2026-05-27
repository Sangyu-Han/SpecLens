from __future__ import annotations

import argparse
import html
import json
import os
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SUITES = [
    {
        "model_slug": "clip_b16",
        "model_label": "CLIP-B/16",
        "suite_manifest": ROOT
        / "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/repeat_suites/clip50k_metric200pb_randompatch_renderfix_20260429/suite_manifest.json",
    },
    {
        "model_slug": "siglip_b16",
        "model_label": "SigLIP-B/16",
        "suite_manifest": ROOT
        / "outputs/autolabel_metric_runs/siglip_full_workspace/outputs/repeat_suites/siglip_metric200pb_randompatch_renderfix_20260429/suite_manifest.json",
    },
    {
        "model_slug": "dinov3_s16",
        "model_label": "DINOv3-S/16",
        "suite_manifest": ROOT
        / "outputs/autolabel_metric_runs/dinov3_full_workspace/outputs/repeat_suites/dinov3_metric200pb_randompatch_20260429_renderfix292/suite_manifest.json",
    },
]


CSS_TEXT = """
:root {
  color-scheme: light;
  --bg: #f7f6f2;
  --panel: #ffffff;
  --ink: #171717;
  --muted: #63635f;
  --line: #d8d6ce;
  --accent: #0f766e;
  --accent-soft: #dff3ef;
  --warn: #9a3412;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, select {
  font: inherit;
}
.shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 18px;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid var(--line);
  background: rgba(247, 246, 242, 0.96);
  backdrop-filter: blur(8px);
}
.topbar-inner {
  max-width: 1440px;
  margin: 0 auto;
  padding: 14px 18px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
}
h1, h2, h3, p { margin-top: 0; }
h1 { font-size: 22px; line-height: 1.2; margin-bottom: 4px; }
h2 { font-size: 18px; margin-bottom: 10px; }
h3 { font-size: 15px; margin-bottom: 8px; }
.meta, .hint {
  color: var(--muted);
  font-size: 13px;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}
.btn {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 6px;
  padding: 8px 10px;
  cursor: pointer;
}
.btn:hover { border-color: #9ca3af; }
.btn.primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.btn.warn {
  color: var(--warn);
}
.layout {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 16px;
  align-items: start;
}
.sidebar {
  position: sticky;
  top: 88px;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 12px;
  max-height: calc(100vh - 104px);
  overflow: auto;
}
.task-nav {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px;
}
.task-dot {
  min-width: 0;
  padding: 6px 0;
  border-radius: 5px;
  border: 1px solid var(--line);
  background: #fff;
  cursor: pointer;
  font-size: 12px;
}
.task-dot.done {
  background: var(--accent-soft);
  border-color: var(--accent);
}
.task-dot.active {
  outline: 2px solid var(--accent);
}
.panel {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 14px;
}
.prompt {
  line-height: 1.5;
}
.label-box {
  border-left: 4px solid var(--accent);
  padding: 10px 12px;
  background: var(--accent-soft);
  border-radius: 6px;
  margin-top: 12px;
}
.label-title {
  font-weight: 700;
  margin-bottom: 5px;
}
.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
}
.candidate-card, .record-card, .label-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 10px;
}
.candidate-card.selected, .label-card.ranked {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.candidate-card img, .record-card img, .target-image {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 6px;
  border: 1px solid var(--line);
}
.code {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 42px;
  height: 26px;
  border-radius: 5px;
  background: #111827;
  color: #fff;
  font-weight: 700;
  font-size: 12px;
  margin-bottom: 8px;
}
.candidate-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}
.score-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px;
  margin-top: 10px;
}
.score-row .btn {
  padding: 7px 0;
}
.score-row .btn.selected {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.axis2-layout {
  display: grid;
  grid-template-columns: minmax(260px, 390px) 1fr;
  gap: 14px;
  align-items: start;
}
.label-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
}
.label-card {
  text-align: left;
  cursor: pointer;
}
.rank-badge {
  display: inline-flex;
  min-width: 28px;
  height: 24px;
  align-items: center;
  justify-content: center;
  border-radius: 5px;
  background: var(--accent);
  color: #fff;
  font-size: 12px;
  font-weight: 700;
}
.candidate-label {
  font-weight: 700;
  margin-bottom: 4px;
}
.candidate-desc {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.35;
}
.rank-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.rank-chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 5px 8px;
  background: #fff;
  font-size: 12px;
}
.footer-nav {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.index-table {
  width: 100%;
  border-collapse: collapse;
}
.index-table th, .index-table td {
  border-bottom: 1px solid var(--line);
  padding: 9px 8px;
  text-align: left;
}
.index-table th {
  color: var(--muted);
  font-size: 13px;
}
details {
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 980px) {
  .layout, .axis2-layout, .topbar-inner {
    grid-template-columns: 1fr;
  }
  .sidebar {
    position: static;
    max-height: none;
  }
  .controls {
    justify-content: flex-start;
  }
}
"""


JS_TEXT = r"""
const data = window.HUMAN_EVAL_DATA;
const storageKey = "human-eval:" + data.metadata.page_id;
let responses = loadResponses();
let currentIndex = Number(new URLSearchParams(window.location.search).get("i") || "0");
if (!Number.isFinite(currentIndex) || currentIndex < 0 || currentIndex >= data.items.length) currentIndex = 0;

function loadResponses() {
  try {
    return JSON.parse(localStorage.getItem(storageKey) || "{}");
  } catch {
    return {};
  }
}

function saveResponses() {
  localStorage.setItem(storageKey, JSON.stringify(responses));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function itemResponse(item) {
  if (!responses[item.item_id]) responses[item.item_id] = {};
  return responses[item.item_id];
}

function isComplete(item) {
  const resp = responses[item.item_id] || {};
  if (data.metadata.task_type === "axis1") return Boolean(resp.selected_candidate);
  if (data.metadata.task_type === "axis2") return Array.isArray(resp.ranked_candidates) && resp.ranked_candidates.length === item.candidates.length;
  if (data.metadata.task_type.startsWith("supp")) {
    const scores = resp.record_scores || {};
    return item.records.every((row) => Number.isInteger(scores[row.code]));
  }
  return false;
}

function progressText() {
  const done = data.items.filter(isComplete).length;
  return `${done} / ${data.items.length} complete`;
}

function setCurrent(index) {
  currentIndex = Math.max(0, Math.min(data.items.length - 1, index));
  const url = new URL(window.location.href);
  url.searchParams.set("i", String(currentIndex));
  window.history.replaceState({}, "", url);
  render();
}

function labelBlock(item) {
  const title = item.label?.canonical_label || "(empty label)";
  const desc = item.label?.description || "";
  return `
    <div class="label-box">
      <div class="label-title">${escapeHtml(title)}</div>
      ${desc ? `<div>${escapeHtml(desc)}</div>` : ""}
    </div>
  `;
}

function detailsBlock(item) {
  return `
    <details>
      <summary>Session metadata</summary>
      <div>Repeat: <code>r${String(item.repeat_index).padStart(2, "0")}</code></div>
    </details>
  `;
}

function renderAxis1(item) {
  const resp = itemResponse(item);
  const cards = item.candidates.map((cand) => {
    const selected = resp.selected_candidate === cand.code;
    return `
      <button class="candidate-card ${selected ? "selected" : ""}" data-action="axis1-select" data-code="${escapeHtml(cand.code)}">
        <div class="candidate-top">
          <span class="code">${escapeHtml(cand.code)}</span>
          <span class="hint">${selected ? "selected" : "click to select"}</span>
        </div>
        <img src="${escapeHtml(cand.image)}" alt="${escapeHtml(cand.code)}">
      </button>
    `;
  }).join("");
  return `
    <section class="panel prompt">
      <h2>Axis 1: same-image token choice</h2>
      <p>Choose the single candidate whose cyan-cross-marked token best matches the feature label. Use token-local evidence, not the broad scene.</p>
      ${labelBlock(item)}
    </section>
    <section class="panel">
      <div class="image-grid">${cards}</div>
    </section>
    <section class="panel">${detailsBlock(item)}</section>
  `;
}

function renderAxis2(item) {
  const resp = itemResponse(item);
  const ranking = Array.isArray(resp.ranked_candidates) ? resp.ranked_candidates : [];
  const rankMap = new Map(ranking.map((code, idx) => [code, idx + 1]));
  const labels = item.candidates.map((cand) => {
    const rank = rankMap.get(cand.code);
    return `
      <button class="label-card ${rank ? "ranked" : ""}" data-action="axis2-toggle" data-code="${escapeHtml(cand.code)}">
        <div class="candidate-top">
          <span class="code">${escapeHtml(cand.code)}</span>
          ${rank ? `<span class="rank-badge">#${rank}</span>` : `<span class="hint">click to rank</span>`}
        </div>
        <div class="candidate-label">${escapeHtml(cand.label?.canonical_label || "(empty label)")}</div>
        ${cand.label?.description ? `<div class="candidate-desc">${escapeHtml(cand.label.description)}</div>` : ""}
      </button>
    `;
  }).join("");
  const chips = ranking.map((code, idx) => `<span class="rank-chip">${idx + 1}. ${escapeHtml(code)}</span>`).join("");
  return `
    <section class="panel prompt">
      <h2>Axis 2: label discrimination</h2>
      <p>Rank all candidate labels from best to worst for the cyan-cross-marked target token. The first ranked label is your answer.</p>
    </section>
    <section class="axis2-layout">
      <div class="panel">
        <h3>Target token</h3>
        <img class="target-image" src="${escapeHtml(item.image)}" alt="target token">
      </div>
      <div>
        <section class="panel">
          <div class="controls" style="justify-content:flex-start; margin-bottom:10px;">
            <button class="btn" data-action="axis2-undo">Undo last</button>
            <button class="btn" data-action="axis2-clear">Clear ranking</button>
            <button class="btn" data-action="axis2-append">Append remaining</button>
          </div>
          <div class="hint">Current ranking: ${chips || "none"}</div>
        </section>
        <section class="panel">
          <div class="label-grid">${labels}</div>
        </section>
      </div>
    </section>
    <section class="panel">${detailsBlock(item)}</section>
  `;
}

function renderSupp(item) {
  const resp = itemResponse(item);
  if (!resp.record_scores) resp.record_scores = {};
  const cards = item.records.map((record) => {
    const selected = resp.record_scores[record.code];
    const buttons = [0, 1, 2, 3, 4].map((score) => `
      <button class="btn ${selected === score ? "selected" : ""}" data-action="supp-score" data-code="${escapeHtml(record.code)}" data-score="${score}">${score}</button>
    `).join("");
    return `
      <div class="record-card">
        <div class="candidate-top">
          <span class="code">${escapeHtml(record.code)}</span>
          <span class="hint">score 0-4</span>
        </div>
        <img src="${escapeHtml(record.image)}" alt="${escapeHtml(record.code)}">
        <div class="score-row">${buttons}</div>
      </div>
    `;
  }).join("");
  return `
    <section class="panel prompt">
      <h2>Supplementary activation prediction</h2>
      <p>For each record, predict how strongly the feature should activate on the cyan-cross-marked token. Use integer scores from 0 to 4.</p>
      <p class="hint">0 means the explanation does not fit the marked token. 4 means the explanation strongly fits the marked token.</p>
      ${labelBlock(item)}
    </section>
    <section class="panel">
      <div class="image-grid">${cards}</div>
    </section>
    <section class="panel">${detailsBlock(item)}</section>
  `;
}

function renderTask() {
  const item = data.items[currentIndex];
  if (data.metadata.task_type === "axis1") return renderAxis1(item);
  if (data.metadata.task_type === "axis2") return renderAxis2(item);
  return renderSupp(item);
}

function renderSidebar() {
  const dots = data.items.map((item, idx) => `
    <button class="task-dot ${idx === currentIndex ? "active" : ""} ${isComplete(item) ? "done" : ""}" data-action="jump" data-index="${idx}">${idx + 1}</button>
  `).join("");
  return `
    <div class="sidebar">
      <h3>Progress</h3>
      <p class="meta">${progressText()}</p>
      <div class="task-nav">${dots}</div>
    </div>
  `;
}

function render() {
  document.getElementById("progress").textContent = progressText();
  document.getElementById("task-title").textContent = `${data.metadata.model_label} - ${data.metadata.task_label} - ${data.metadata.condition_label}`;
  document.getElementById("task-meta").textContent = `${data.items.length} items. Responses are stored only in this browser until exported.`;
  document.getElementById("sidebar-root").innerHTML = renderSidebar();
  document.getElementById("task-root").innerHTML = renderTask();
  document.getElementById("prev-btn").disabled = currentIndex === 0;
  document.getElementById("next-btn").disabled = currentIndex === data.items.length - 1;
}

function downloadJson() {
  const responseRows = data.items.map((item) => ({
    item_id: item.item_id,
    response: responses[item.item_id] || {},
  }));
  const payload = {
    metadata: data.metadata,
    exported_at: new Date().toISOString(),
    responses: responseRows,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${data.metadata.page_id}_responses.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

function importJson(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(String(reader.result || "{}"));
      const incoming = payload.responses || {};
      if (Array.isArray(incoming)) {
        for (const row of incoming) {
          if (row.item_id) responses[row.item_id] = row.response || {};
        }
      } else {
        Object.assign(responses, incoming);
      }
      saveResponses();
      render();
    } catch (err) {
      alert("Could not import JSON: " + err);
    }
  };
  reader.readAsText(file);
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  const item = data.items[currentIndex];
  const resp = itemResponse(item);

  if (action === "jump") setCurrent(Number(target.dataset.index));
  if (action === "axis1-select") {
    resp.selected_candidate = target.dataset.code;
    saveResponses();
    render();
  }
  if (action === "axis2-toggle") {
    const code = target.dataset.code;
    const ranking = Array.isArray(resp.ranked_candidates) ? resp.ranked_candidates.slice() : [];
    const idx = ranking.indexOf(code);
    if (idx >= 0) ranking.splice(idx, 1);
    else ranking.push(code);
    resp.ranked_candidates = ranking;
    resp.best_candidate = ranking[0] || "";
    saveResponses();
    render();
  }
  if (action === "axis2-undo") {
    const ranking = Array.isArray(resp.ranked_candidates) ? resp.ranked_candidates.slice() : [];
    ranking.pop();
    resp.ranked_candidates = ranking;
    resp.best_candidate = ranking[0] || "";
    saveResponses();
    render();
  }
  if (action === "axis2-clear") {
    resp.ranked_candidates = [];
    resp.best_candidate = "";
    saveResponses();
    render();
  }
  if (action === "axis2-append") {
    const ranking = Array.isArray(resp.ranked_candidates) ? resp.ranked_candidates.slice() : [];
    for (const cand of item.candidates) {
      if (!ranking.includes(cand.code)) ranking.push(cand.code);
    }
    resp.ranked_candidates = ranking;
    resp.best_candidate = ranking[0] || "";
    saveResponses();
    render();
  }
  if (action === "supp-score") {
    if (!resp.record_scores) resp.record_scores = {};
    resp.record_scores[target.dataset.code] = Number(target.dataset.score);
    saveResponses();
    render();
  }
});

document.getElementById("prev-btn").addEventListener("click", () => setCurrent(currentIndex - 1));
document.getElementById("next-btn").addEventListener("click", () => setCurrent(currentIndex + 1));
document.getElementById("export-btn").addEventListener("click", downloadJson);
document.getElementById("import-input").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) importJson(file);
});
document.getElementById("clear-current-btn").addEventListener("click", () => {
  const item = data.items[currentIndex];
  delete responses[item.item_id];
  saveResponses();
  render();
});
document.getElementById("clear-all-btn").addEventListener("click", () => {
  if (!confirm("Clear all local responses for this page?")) return;
  responses = {};
  saveResponses();
  render();
});

render();
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def derive_description(output: dict[str, Any]) -> str:
    for key in ("description", "support_summary", "detailed_description", "rationale", "notes"):
        value = str(output.get(key) or "").strip()
        if value:
            return value
    return ""


def load_label_lookup(workspace_root: Path, session_name: str) -> dict[str, dict[str, str]]:
    raw_path = workspace_root / "outputs" / "review_sessions" / session_name / "raw_predictions.json"
    payload = read_json(raw_path)
    lookup: dict[str, dict[str, str]] = {}
    for row in list(payload.get("features") or []):
        output = dict(row.get("output") or {})
        lookup[str(row["feature_key"])] = {
            "canonical_label": str(output.get("canonical_label") or "").strip(),
            "description": derive_description(output),
        }
    return lookup


def rel_asset(path: str, page_dir: Path) -> str:
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = (ROOT / asset_path).resolve()
    return os.path.relpath(str(asset_path), str(page_dir))


def condition_map_for_suite(session_prefix: str, seed: int) -> dict[str, str]:
    variants = ["erf_cyan_cross", "sae_only"]
    rng = random.Random(f"{seed}:{session_prefix}")
    rng.shuffle(variants)
    return {"condition_a": variants[0], "condition_b": variants[1]}


def axis_manifest_from_summary(summary_path: Path) -> dict[str, Any]:
    return read_json(summary_path.parent / "pilot_manifest.json")


def supp_manifest_from_summary(summary_path: Path) -> dict[str, Any]:
    return read_json(summary_path.parent / "pilot_manifest.json")


def item_base(item: dict[str, Any], repeat_index: int, task_type: str) -> dict[str, Any]:
    return {
        "item_id": f"r{repeat_index:02d}:{task_type}:{item['feature_key']}",
        "repeat_index": int(repeat_index),
        "feature_key": str(item["feature_key"]),
        "block_idx": int(item["block_idx"]),
        "feature_id": int(item["feature_id"]),
    }


def build_axis1_page_data(
    *,
    suite_manifest: dict[str, Any],
    condition_id: str,
    variant_id: str,
    page_dir: Path,
    model_slug: str,
    model_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace_root = Path(suite_manifest["workspace_root"])
    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for repeat in list(suite_manifest["repeats"]):
        repeat_index = int(repeat["repeat_index"])
        manifest = axis_manifest_from_summary(Path(repeat["axis_summary_json"]))
        labels = load_label_lookup(workspace_root, str(manifest["variant_sessions"][variant_id]))
        for raw_item in list(manifest["axis1_items"]):
            feature_key = str(raw_item["feature_key"])
            row = {
                **item_base(raw_item, repeat_index, "axis1"),
                "label": labels.get(feature_key, {"canonical_label": "", "description": ""}),
                "candidates": [
                    {
                        "code": str(img["candidate_code"]),
                        "image": rel_asset(str(img["image_path"]), page_dir),
                    }
                    for img in list(raw_item["input_images"])
                ],
            }
            items.append(row)
            key_items.append(
                {
                    "item_id": row["item_id"],
                    "feature_key": feature_key,
                    "block_idx": int(raw_item["block_idx"]),
                    "gold_code": str(raw_item["gold_code"]),
                    "positive_token_idx": int(raw_item["token_idx"]),
                }
            )
    data = {
        "metadata": {
            "page_id": f"{model_slug}_axis1_{condition_id}",
            "model_slug": model_slug,
            "model_label": model_label,
            "task_type": "axis1",
            "task_label": "Axis 1",
            "condition_id": condition_id,
            "condition_label": condition_id.replace("_", " ").title(),
            "n_items": len(items),
        },
        "items": items,
    }
    key = {
        "metadata": {**data["metadata"], "variant_id": variant_id},
        "items": key_items,
    }
    return data, key


def build_axis2_page_data(
    *,
    suite_manifest: dict[str, Any],
    condition_id: str,
    variant_id: str,
    page_dir: Path,
    model_slug: str,
    model_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace_root = Path(suite_manifest["workspace_root"])
    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for repeat in list(suite_manifest["repeats"]):
        repeat_index = int(repeat["repeat_index"])
        manifest = axis_manifest_from_summary(Path(repeat["axis_summary_json"]))
        labels = load_label_lookup(workspace_root, str(manifest["variant_sessions"][variant_id]))
        for raw_item in list(manifest["axis2_items"]):
            feature_key = str(raw_item["feature_key"])
            row = {
                **item_base(raw_item, repeat_index, "axis2"),
                "image": rel_asset(str(raw_item["input_images"][0]["image_path"]), page_dir),
                "candidates": [
                    {
                        "code": str(cand["candidate_code"]),
                        "label": labels.get(str(cand["feature_key"]), {"canonical_label": "", "description": ""}),
                    }
                    for cand in list(raw_item["candidates"])
                ],
            }
            items.append(row)
            key_items.append(
                {
                    "item_id": row["item_id"],
                    "feature_key": feature_key,
                    "block_idx": int(raw_item["block_idx"]),
                    "gold_code": str(raw_item["gold_code"]),
                    "candidate_count": len(raw_item["candidates"]),
                }
            )
    data = {
        "metadata": {
            "page_id": f"{model_slug}_axis2_{condition_id}",
            "model_slug": model_slug,
            "model_label": model_label,
            "task_type": "axis2",
            "task_label": "Axis 2",
            "condition_id": condition_id,
            "condition_label": condition_id.replace("_", " ").title(),
            "n_items": len(items),
        },
        "items": items,
    }
    key = {
        "metadata": {**data["metadata"], "variant_id": variant_id},
        "items": key_items,
    }
    return data, key


def build_supp_page_data(
    *,
    suite_manifest: dict[str, Any],
    condition_id: str,
    variant_id: str,
    split_name: str,
    page_dir: Path,
    model_slug: str,
    model_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace_root = Path(suite_manifest["workspace_root"])
    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for repeat in list(suite_manifest["repeats"]):
        repeat_index = int(repeat["repeat_index"])
        manifest = supp_manifest_from_summary(Path(repeat["supp_summary_json"]))
        labels = load_label_lookup(workspace_root, str(manifest["variant_sessions"][variant_id]))
        for raw_item in list(manifest[f"{split_name}_items"]):
            feature_key = str(raw_item["feature_key"])
            row = {
                **item_base(raw_item, repeat_index, split_name),
                "label": labels.get(feature_key, {"canonical_label": "", "description": ""}),
                "records": [
                    {
                        "code": str(record["record_code"]),
                        "image": rel_asset(str(record["original_with_token_box"]), page_dir),
                    }
                    for record in list(raw_item["records"])
                ],
            }
            items.append(row)
            key_items.append(
                {
                    "item_id": row["item_id"],
                    "feature_key": feature_key,
                    "block_idx": int(raw_item["block_idx"]),
                    "records": [
                        {
                            "code": str(record["record_code"]),
                            "role": str(record["role"]),
                            "binary_label": int(record["binary_label"]),
                            "normalized_activation": float(record["normalized_activation"]),
                        }
                        for record in list(raw_item["records"])
                    ],
                }
            )
    data = {
        "metadata": {
            "page_id": f"{model_slug}_{split_name}_{condition_id}",
            "model_slug": model_slug,
            "model_label": model_label,
            "task_type": split_name,
            "task_label": "Supplementary test" if split_name == "supp_test" else "Supplementary valid",
            "condition_id": condition_id,
            "condition_label": condition_id.replace("_", " ").title(),
            "n_items": len(items),
        },
        "items": items,
    }
    key = {
        "metadata": {**data["metadata"], "variant_id": variant_id},
        "items": key_items,
    }
    return data, key


def page_html(data: dict[str, Any], page_dir: Path, out_root: Path) -> str:
    css_rel = os.path.relpath(str(out_root / "assets" / "human_eval.css"), str(page_dir))
    js_rel = os.path.relpath(str(out_root / "assets" / "human_eval.js"), str(page_dir))
    data_json = json.dumps(data, ensure_ascii=True)
    title = f"{data['metadata']['model_label']} - {data['metadata']['task_label']} - {data['metadata']['condition_label']}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{html.escape(css_rel)}">
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <h1 id="task-title">{html.escape(title)}</h1>
        <div class="meta" id="task-meta"></div>
      </div>
      <div class="controls">
        <span class="meta" id="progress"></span>
        <button class="btn" id="prev-btn">Previous</button>
        <button class="btn" id="next-btn">Next</button>
        <button class="btn primary" id="export-btn">Export JSON</button>
        <label class="btn">Import JSON<input id="import-input" type="file" accept="application/json" hidden></label>
        <button class="btn warn" id="clear-current-btn">Clear Item</button>
        <button class="btn warn" id="clear-all-btn">Clear All</button>
      </div>
    </div>
  </header>
  <main class="shell layout">
    <aside id="sidebar-root"></aside>
    <div>
      <div id="task-root"></div>
      <div class="footer-nav panel">
        <button class="btn" onclick="document.getElementById('prev-btn').click()">Previous</button>
        <button class="btn primary" onclick="document.getElementById('next-btn').click()">Next</button>
      </div>
    </div>
  </main>
  <script>window.HUMAN_EVAL_DATA = {data_json};</script>
  <script src="{html.escape(js_rel)}"></script>
</body>
</html>
"""


def index_html(rows: list[dict[str, Any]]) -> str:
    row_html = "\n".join(
        f"""<tr>
  <td>{html.escape(row['model_label'])}</td>
  <td>{html.escape(row['task_label'])}</td>
  <td>{html.escape(row['condition_label'])}</td>
  <td>{int(row['n_items'])}</td>
  <td><a href="{html.escape(row['href'])}">open</a></td>
</tr>"""
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human baseline evaluation index</title>
  <link rel="stylesheet" href="assets/human_eval.css">
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Human baseline evaluation pages</h1>
      <p class="meta">These pages mirror the machine Axis 1, Axis 2, and supplementary test tasks. Participant pages hide gold answers and use blind Condition A/B labels. The condition-to-variant mapping and gold answers are stored separately under <code>answer_keys/</code>.</p>
      <p class="meta">Responses are saved in browser localStorage. Use Export JSON after completing a page.</p>
    </section>
    <section class="panel">
      <table class="index-table">
        <thead>
          <tr><th>Model</th><th>Task</th><th>Condition</th><th>Items</th><th>Page</th></tr>
        </thead>
        <tbody>
          {row_html}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_pages_for_suite(
    *,
    suite_info: dict[str, Any],
    out_root: Path,
    seed: int,
    supp_splits: list[str],
) -> list[dict[str, Any]]:
    suite_manifest_path = Path(suite_info["suite_manifest"])
    suite_manifest = read_json(suite_manifest_path)
    session_prefix = str(suite_manifest["session_prefix"])
    model_slug = str(suite_info["model_slug"])
    model_label = str(suite_info["model_label"])
    model_dir = out_root / model_slug
    answer_dir = out_root / "answer_keys"
    cond_map = condition_map_for_suite(session_prefix, seed)
    write_json(
        answer_dir / f"{model_slug}_condition_map.json",
        {
            "model_slug": model_slug,
            "model_label": model_label,
            "session_prefix": session_prefix,
            "condition_map": cond_map,
        },
    )

    index_rows: list[dict[str, Any]] = []

    for condition_id, variant_id in cond_map.items():
        for task_type, builder, task_label in [
            ("axis1", build_axis1_page_data, "Axis 1"),
            ("axis2", build_axis2_page_data, "Axis 2"),
        ]:
            page_path = model_dir / f"{task_type}_{condition_id}.html"
            data, key = builder(
                suite_manifest=suite_manifest,
                condition_id=condition_id,
                variant_id=variant_id,
                page_dir=page_path.parent,
                model_slug=model_slug,
                model_label=model_label,
            )
            write_text(page_path, page_html(data, page_path.parent, out_root))
            write_json(answer_dir / f"{model_slug}_{task_type}_{condition_id}_answer_key.json", key)
            index_rows.append(
                {
                    "model_label": model_label,
                    "task_label": task_label,
                    "condition_label": condition_id.replace("_", " ").title(),
                    "n_items": len(data["items"]),
                    "href": os.path.relpath(str(page_path), str(out_root)),
                }
            )

        for split_name in supp_splits:
            page_path = model_dir / f"{split_name}_{condition_id}.html"
            data, key = build_supp_page_data(
                suite_manifest=suite_manifest,
                condition_id=condition_id,
                variant_id=variant_id,
                split_name=split_name,
                page_dir=page_path.parent,
                model_slug=model_slug,
                model_label=model_label,
            )
            write_text(page_path, page_html(data, page_path.parent, out_root))
            write_json(answer_dir / f"{model_slug}_{split_name}_{condition_id}_answer_key.json", key)
            index_rows.append(
                {
                    "model_label": model_label,
                    "task_label": "Supplementary test" if split_name == "supp_test" else "Supplementary valid",
                    "condition_label": condition_id.replace("_", " ").title(),
                    "n_items": len(data["items"]),
                    "href": os.path.relpath(str(page_path), str(out_root)),
                }
            )

    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static human-baseline HTML pages for Axis 1, Axis 2, and supplementary tasks.")
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "outputs/autolabel_metric_runs/human_baseline_axis_supp_html_20260503"),
    )
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument(
        "--supp-split",
        action="append",
        default=None,
        choices=["supp_valid", "supp_test"],
        help="Supplementary split to expose as a human task. Repeat to include both.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    (out_root / "assets").mkdir(parents=True, exist_ok=True)
    write_text(out_root / "assets" / "human_eval.css", CSS_TEXT.strip() + "\n")
    write_text(out_root / "assets" / "human_eval.js", JS_TEXT.strip() + "\n")

    index_rows: list[dict[str, Any]] = []
    for suite_info in DEFAULT_SUITES:
        index_rows.extend(
            build_pages_for_suite(
                suite_info=suite_info,
                out_root=out_root,
                seed=int(args.seed),
                supp_splits=list(args.supp_split or ["supp_test"]),
            )
        )
    write_text(out_root / "index.html", index_html(index_rows))
    print(f"Wrote human baseline HTML index: {out_root / 'index.html'}")
    print(f"Wrote {len(index_rows)} participant pages plus answer keys.")


if __name__ == "__main__":
    main()
