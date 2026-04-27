from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence


FieldKind = Literal["text", "textarea", "number", "select", "checkbox", "choice"]


@dataclass(frozen=True)
class RoleField:
    key: str
    label: str
    kind: FieldKind = "text"
    default: Any = ""
    placeholder: str = ""
    help_text: str = ""
    options: tuple[str, ...] = ()
    rows: int = 3
    step: str = "any"


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    title: str
    instructions: str
    fields: tuple[RoleField, ...]


@dataclass(frozen=True)
class StudyItem:
    item_id: str
    title: str
    evidence_html: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudySessionSpec:
    session_id: str
    title: str
    roles: tuple[RoleSpec, ...]
    items: tuple[StudyItem, ...]
    storage_key: str | None = None
    intro_html: str = ""
    footer_html: str = ""
    autosave: bool = True
    show_import_export: bool = True
    initial_state: dict[str, Any] = field(default_factory=dict)
    export_filename: str | None = None


def text_field(
    key: str,
    label: str,
    *,
    default: str = "",
    placeholder: str = "",
    help_text: str = "",
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="text",
        default=default,
        placeholder=placeholder,
        help_text=help_text,
    )


def textarea_field(
    key: str,
    label: str,
    *,
    default: str = "",
    placeholder: str = "",
    help_text: str = "",
    rows: int = 4,
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="textarea",
        default=default,
        placeholder=placeholder,
        help_text=help_text,
        rows=rows,
    )


def number_field(
    key: str,
    label: str,
    *,
    default: float | int = 0,
    placeholder: str = "",
    help_text: str = "",
    step: str = "any",
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="number",
        default=default,
        placeholder=placeholder,
        help_text=help_text,
        step=step,
    )


def select_field(
    key: str,
    label: str,
    options: Sequence[str],
    *,
    default: str = "",
    help_text: str = "",
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="select",
        default=default,
        help_text=help_text,
        options=tuple(str(v) for v in options),
    )


def checkbox_field(
    key: str,
    label: str,
    *,
    default: bool = False,
    help_text: str = "",
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="checkbox",
        default=bool(default),
        help_text=help_text,
    )


def choice_field(
    key: str,
    label: str,
    options: Sequence[str],
    *,
    default: str = "",
    help_text: str = "",
) -> RoleField:
    return RoleField(
        key=key,
        label=label,
        kind="choice",
        default=default,
        help_text=help_text,
        options=tuple(str(v) for v in options),
    )


def default_axis1_team_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            role_id="planner",
            title="Planner",
            instructions=(
                "Inspect the evidence and write the minimal hypothesis that could explain the feature. "
                "Prefer a short concept phrase and note uncertainty if needed."
            ),
            fields=(
                textarea_field("hypothesis", "Hypothesis", rows=4, placeholder="Short concept phrase or short description"),
                textarea_field("evidence_notes", "Evidence notes", rows=3, placeholder="What evidence supports this hypothesis?"),
                select_field("status", "Status", ("draft", "confident", "uncertain", "reject"), default="draft"),
            ),
        ),
        RoleSpec(
            role_id="generator",
            title="Generator",
            instructions=(
                "Turn the planner's hypothesis into a canonical label and a short general description. "
                "Avoid sample-specific locations or numeric thresholds."
            ),
            fields=(
                text_field("canonical_label", "Canonical label", placeholder="2 to 8 words"),
                textarea_field("description", "Description", rows=4, placeholder="Short general description"),
                number_field("confidence", "Confidence", default=0.5, step="0.05"),
            ),
        ),
        RoleSpec(
            role_id="evaluator",
            title="Evaluator",
            instructions=(
                "Judge whether the generator output is acceptable for downstream Axis 1 and Axis 2 tasks. "
                "If not, explain the failure mode."
            ),
            fields=(
                select_field("decision", "Decision", ("accept", "revise", "reject"), default="revise"),
                number_field("score", "Score", default=0, step="1"),
                textarea_field("feedback", "Feedback", rows=4, placeholder="What should be changed?"),
            ),
        ),
    )


def _json_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _field_input_markup(session_id: str, item_id: str, role_id: str, field: RoleField) -> str:
    dom_id = f"{session_id}__{item_id}__{role_id}__{field.key}"
    data_attrs = (
        f'data-session-id="{html.escape(session_id)}" '
        f'data-item-id="{html.escape(item_id)}" '
        f'data-role-id="{html.escape(role_id)}" '
        f'data-field-key="{html.escape(field.key)}" '
        f'data-binding="true"'
    )
    help_html = f'<div class="field-help">{html.escape(field.help_text)}</div>' if field.help_text else ""
    if field.kind == "textarea":
        return (
            f'<label class="field" for="{dom_id}">'
            f'<div class="field-label">{html.escape(field.label)}</div>'
            f'<textarea id="{dom_id}" rows="{int(field.rows)}" placeholder="{html.escape(field.placeholder)}" {data_attrs}></textarea>'
            f"{help_html}"
            "</label>"
        )
    if field.kind == "number":
        return (
            f'<label class="field" for="{dom_id}">'
            f'<div class="field-label">{html.escape(field.label)}</div>'
            f'<input id="{dom_id}" type="number" step="{html.escape(str(field.step))}" placeholder="{html.escape(field.placeholder)}" {data_attrs}>'
            f"{help_html}"
            "</label>"
        )
    if field.kind == "select":
        options = []
        for option in field.options:
            options.append(f'<option value="{html.escape(option)}">{html.escape(option)}</option>')
        return (
            f'<label class="field" for="{dom_id}">'
            f'<div class="field-label">{html.escape(field.label)}</div>'
            f'<select id="{dom_id}" {data_attrs}>'
            f'{"".join(options)}'
            "</select>"
            f"{help_html}"
            "</label>"
        )
    if field.kind == "choice":
        hidden_value = html.escape(str(field.default)) if field.default is not None else ""
        buttons = []
        for option in field.options:
            buttons.append(
                f'<button type="button" class="choice-btn" '
                f'data-choice-button="true" '
                f'data-target-id="{dom_id}" '
                f'data-choice-value="{html.escape(option)}">{html.escape(option)}</button>'
            )
        return (
            f'<label class="field" for="{dom_id}">'
            f'<div class="field-label">{html.escape(field.label)}</div>'
            f'<input id="{dom_id}" type="hidden" value="{hidden_value}" {data_attrs}>'
            f'<div class="choice-group" data-choice-group="true" data-target-id="{dom_id}">'
            f'{"".join(buttons)}'
            "</div>"
            f"{help_html}"
            "</label>"
        )
    if field.kind == "checkbox":
        checked = " checked" if bool(field.default) else ""
        return (
            f'<label class="field field-inline" for="{dom_id}">'
            f'<input id="{dom_id}" type="checkbox"{checked} {data_attrs}>'
            f'<span class="field-label">{html.escape(field.label)}</span>'
            f"{help_html}"
            "</label>"
        )
    return (
        f'<label class="field" for="{dom_id}">'
        f'<div class="field-label">{html.escape(field.label)}</div>'
        f'<input id="{dom_id}" type="text" placeholder="{html.escape(field.placeholder)}" {data_attrs}>'
        f"{help_html}"
        "</label>"
    )


def _default_item_state(session: StudySessionSpec) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for item in session.items:
        role_state: dict[str, Any] = {}
        for role in session.roles:
            field_state: dict[str, Any] = {}
            for field in role.fields:
                field_state[field.key] = field.default
            role_state[role.role_id] = field_state
        items[item.item_id] = role_state
    return {"items": items}


def session_manifest(session: StudySessionSpec) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "title": session.title,
        "storage_key": session.storage_key or f"autolabel.study.{session.session_id}.v1",
        "export_filename": session.export_filename or f"{session.session_id}.json",
        "autosave": bool(session.autosave),
        "show_import_export": bool(session.show_import_export),
        "roles": [
            {
                "role_id": role.role_id,
                "title": role.title,
                "instructions": role.instructions,
                "fields": [_to_serializable(asdict(field)) for field in role.fields],
            }
            for role in session.roles
        ],
        "items": [
            {
                "item_id": item.item_id,
                "title": item.title,
                "metadata": _to_serializable(item.metadata),
            }
            for item in session.items
        ],
    }


def build_study_page_html(session: StudySessionSpec) -> str:
    storage_key = session.storage_key or f"autolabel.study.{session.session_id}.v1"
    manifest = session_manifest(session)
    initial_state = _to_serializable(session.initial_state) or _default_item_state(session)
    payload = {
        "manifest": manifest,
        "initial_state": initial_state,
        "storage_key": storage_key,
    }
    role_html = []
    for role in session.roles:
        field_html = "".join(
            _field_input_markup(session.session_id, "{item_id}", role.role_id, field) for field in role.fields
        )
        role_html.append(
            f"""
            <section class="role-card" data-role-id="{html.escape(role.role_id)}">
              <div class="role-head">
                <h3>{html.escape(role.title)}</h3>
                <p>{html.escape(role.instructions)}</p>
              </div>
              <div class="role-fields" data-role-field-template="{html.escape(role.role_id)}">
                {field_html}
              </div>
            </section>
            """
        )

    item_html = []
    for item in session.items:
        per_role = []
        for role in session.roles:
            fields = "".join(_field_input_markup(session.session_id, item.item_id, role.role_id, field) for field in role.fields)
            per_role.append(
                f"""
                <section class="role-card" data-role-id="{html.escape(role.role_id)}">
                  <div class="role-head">
                    <h3>{html.escape(role.title)}</h3>
                    <p>{html.escape(role.instructions)}</p>
                  </div>
                  <div class="role-fields">
                    {fields}
                  </div>
                </section>
                """
            )
        metadata_bits = " ".join(
            f"<span class='chip'>{html.escape(str(key))}: {html.escape(str(value))}</span>"
            for key, value in item.metadata.items()
        )
        item_html.append(
            f"""
            <article class="study-item" data-item-id="{html.escape(item.item_id)}">
              <header class="item-head">
                <h2>{html.escape(item.title)}</h2>
                <div class="item-meta">{metadata_bits}</div>
              </header>
              <div class="evidence">{item.evidence_html}</div>
              <div class="role-grid">
                {''.join(per_role)}
              </div>
            </article>
            """
        )

    import_export_block = ""
    if session.show_import_export:
        import_export_block = """
        <section class="storage-panel">
          <div class="storage-row">
            <button type="button" id="btn-load">Load Saved</button>
            <button type="button" id="btn-reset">Reset</button>
            <button type="button" id="btn-export">Export JSON</button>
            <button type="button" id="btn-copy">Copy JSON</button>
            <label class="file-picker">
              <input type="file" id="file-import" accept="application/json">
              <span>Import file</span>
            </label>
          </div>
          <textarea id="json-import" spellcheck="false" placeholder="Paste JSON here to import"></textarea>
          <div class="storage-row">
            <button type="button" id="btn-import">Import JSON</button>
            <span id="storage-status" class="status">Idle</span>
          </div>
        </section>
        """

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(session.title)}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffdf9;
      --ink: #1c1c1c;
      --muted: #66615a;
      --line: #d8d1c6;
      --accent: #bb5a1f;
      --accent-soft: #f5e3d5;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f7f1e9 0%, #f2ece2 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.05);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: -0.02em;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .study-item {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.04);
    }}
    .item-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: baseline;
      justify-content: space-between;
    }}
    .item-head h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .item-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      border: 1px solid #efc7ae;
    }}
    .evidence {{
      margin: 16px 0 18px;
    }}
    .role-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .role-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 14px;
    }}
    .role-head h3 {{
      margin: 0 0 6px;
      font-size: 16px;
    }}
    .role-head p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .field {{
      display: block;
      margin-bottom: 12px;
    }}
    .field-inline {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .field-label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 5px;
    }}
    .field input[type="text"],
    .field input[type="number"],
    .field input[type="hidden"],
    .field select,
    .field textarea,
    #json-import {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    .field textarea,
    #json-import {{
      min-height: 96px;
      resize: vertical;
    }}
    .choice-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .choice-btn {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
      min-width: 44px;
      text-align: center;
    }}
    .choice-btn.is-active {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(187, 90, 31, 0.15);
    }}
    .field-help {{
      margin-top: 5px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .storage-panel {{
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      margin-bottom: 20px;
      backdrop-filter: blur(4px);
    }}
    .storage-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .storage-row button,
    .file-picker span {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }}
    .file-picker input {{
      display: none;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
    }}
    .intro, .footer {{
      color: var(--muted);
      line-height: 1.55;
    }}
    @media (max-width: 1080px) {{
      .role-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(session.title)}</h1>
      <p class="intro">{session.intro_html}</p>
    </section>
    {import_export_block}
    {"".join(item_html)}
    <section class="footer">{session.footer_html}</section>
  </div>
  <script>
    window.__STUDY_SPEC__ = {_json_blob(payload)};
  </script>
  <script>
    (() => {{
      const SPEC = window.__STUDY_SPEC__;
      const STORAGE_KEY = SPEC.storage_key;
      const DEFAULT_STATE = SPEC.initial_state || {{items: {{}}}};
      const STATUS = document.getElementById("storage-status");
      const IMPORT_BOX = document.getElementById("json-import");
      const IMPORT_FILE = document.getElementById("file-import");

      let state = structuredClone(DEFAULT_STATE);

      function setStatus(text) {{
        if (STATUS) {{
          STATUS.textContent = text;
        }}
      }}

      function readControlValue(el) {{
        if (el.type === "checkbox") {{
          return !!el.checked;
        }}
        if (el.type === "number") {{
          const value = el.value.trim();
          return value === "" ? null : Number(value);
        }}
        return el.value;
      }}

      function writeControlValue(el, value) {{
        if (el.type === "checkbox") {{
          el.checked = !!value;
          return;
        }}
        if (value === undefined || value === null) {{
          el.value = "";
          return;
        }}
        el.value = String(value);
      }}

      function ensurePath(itemId, roleId) {{
        if (!state.items) state.items = {{}};
        if (!state.items[itemId]) state.items[itemId] = {{}};
        if (!state.items[itemId][roleId]) state.items[itemId][roleId] = {{}};
        return state.items[itemId][roleId];
      }}

      function syncDomFromState() {{
        document.querySelectorAll("[data-binding='true']").forEach((el) => {{
          const itemId = el.dataset.itemId;
          const roleId = el.dataset.roleId;
          const fieldKey = el.dataset.fieldKey;
          const item = state.items?.[itemId]?.[roleId] || {{}};
          writeControlValue(el, item[fieldKey]);
        }});
        refreshChoiceGroups();
      }}

      function refreshChoiceGroups() {{
        document.querySelectorAll("[data-choice-group='true']").forEach((group) => {{
          const targetId = group.dataset.targetId;
          const input = targetId ? document.getElementById(targetId) : null;
          const current = input ? String(input.value ?? "") : "";
          group.querySelectorAll("[data-choice-button='true']").forEach((button) => {{
            button.classList.toggle("is-active", String(button.dataset.choiceValue ?? "") === current);
          }});
        }});
      }}

      function persist() {{
        try {{
          localStorage.setItem(STORAGE_KEY, JSON.stringify({{
            saved_at: new Date().toISOString(),
            state,
          }}));
          setStatus("Saved locally");
        }} catch (err) {{
          setStatus("Save failed: " + err);
        }}
      }}

      function loadSaved() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) {{
            setStatus("No saved state");
            return false;
          }}
          const parsed = JSON.parse(raw);
          state = parsed.state || parsed;
          syncDomFromState();
          setStatus("Loaded from localStorage");
          return true;
        }} catch (err) {{
          setStatus("Load failed: " + err);
          return false;
        }}
      }}

      function exportJSON() {{
        return JSON.stringify({{
          session_id: SPEC.manifest.session_id,
          storage_key: STORAGE_KEY,
          exported_at: new Date().toISOString(),
          state,
        }}, null, 2);
      }}

      function download(filename, text) {{
        const blob = new Blob([text], {{type: "application/json"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }}

      function importJSON(text) {{
        const parsed = JSON.parse(text);
        const next = parsed.state || parsed;
        state = next;
        syncDomFromState();
        persist();
        setStatus("Imported JSON");
      }}

      document.addEventListener("input", (ev) => {{
        const el = ev.target;
        if (!el.matches("[data-binding='true']")) return;
        const itemId = el.dataset.itemId;
        const roleId = el.dataset.roleId;
        const fieldKey = el.dataset.fieldKey;
        const slot = ensurePath(itemId, roleId);
        slot[fieldKey] = readControlValue(el);
        if (SPEC.autosave) persist();
        else setStatus("Edited");
      }});

      document.addEventListener("change", (ev) => {{
        const el = ev.target;
        if (!el.matches("[data-binding='true']")) return;
        const itemId = el.dataset.itemId;
        const roleId = el.dataset.roleId;
        const fieldKey = el.dataset.fieldKey;
        const slot = ensurePath(itemId, roleId);
        slot[fieldKey] = readControlValue(el);
        if (SPEC.autosave) persist();
        else setStatus("Edited");
      }});

      document.addEventListener("click", (ev) => {{
        const el = ev.target;
        if (!(el instanceof HTMLElement)) return;
        if (!el.matches("[data-choice-button='true']")) return;
        const targetId = el.dataset.targetId;
        const value = el.dataset.choiceValue ?? "";
        const input = targetId ? document.getElementById(targetId) : null;
        if (!input) return;
        input.value = value;
        const itemId = input.dataset.itemId;
        const roleId = input.dataset.roleId;
        const fieldKey = input.dataset.fieldKey;
        const slot = ensurePath(itemId, roleId);
        slot[fieldKey] = value;
        refreshChoiceGroups();
        if (SPEC.autosave) persist();
        else setStatus("Edited");
      }});

      const btnLoad = document.getElementById("btn-load");
      const btnReset = document.getElementById("btn-reset");
      const btnExport = document.getElementById("btn-export");
      const btnCopy = document.getElementById("btn-copy");
      const btnImport = document.getElementById("btn-import");

      if (btnLoad) btnLoad.addEventListener("click", () => loadSaved());
      if (btnReset) btnReset.addEventListener("click", () => {{
        state = structuredClone(DEFAULT_STATE);
        syncDomFromState();
        persist();
        setStatus("Reset to defaults");
      }});
      if (btnExport) btnExport.addEventListener("click", () => {{
        download(SPEC.manifest.export_filename || `${{SPEC.manifest.session_id}}.json`, exportJSON());
      }});
      if (btnCopy) btnCopy.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(exportJSON());
          setStatus("Copied JSON");
        }} catch (err) {{
          setStatus("Copy failed: " + err);
        }}
      }});
      if (btnImport) btnImport.addEventListener("click", () => {{
        try {{
          if (!IMPORT_BOX) return;
          importJSON(IMPORT_BOX.value.trim());
        }} catch (err) {{
          setStatus("Import failed: " + err);
        }}
      }});
      if (IMPORT_FILE) IMPORT_FILE.addEventListener("change", async () => {{
        const file = IMPORT_FILE.files && IMPORT_FILE.files[0];
        if (!file) return;
        try {{
          const text = await file.text();
          importJSON(text);
        }} catch (err) {{
          setStatus("File import failed: " + err);
        }}
      }});

      syncDomFromState();
      if (!loadSaved()) {{
        persist();
      }}
    }})();
  </script>
</body>
</html>
"""
    return html_doc


def write_study_page(path: Path, session: StudySessionSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_study_page_html(session))


def axis1_team_session(
    *,
    session_id: str,
    title: str,
    items: Sequence[StudyItem],
    roles: Sequence[RoleSpec] | None = None,
    storage_key: str | None = None,
    intro_html: str = "",
    footer_html: str = "",
    initial_state: dict[str, Any] | None = None,
    export_filename: str | None = None,
) -> StudySessionSpec:
    return StudySessionSpec(
        session_id=session_id,
        title=title,
        roles=tuple(roles or default_axis1_team_roles()),
        items=tuple(items),
        storage_key=storage_key,
        intro_html=intro_html,
        footer_html=footer_html,
        initial_state=initial_state or {},
        export_filename=export_filename,
    )
