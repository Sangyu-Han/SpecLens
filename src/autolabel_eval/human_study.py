from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import EvalConfig
from .feature_bank import load_feature_bank
from .legacy import LegacyRuntime
from .prompts import axis1_c_conditioned_prompt
from .rendering import (
    save_cosine_overlay_image,
    save_erf_heatmap_image,
    save_feature_actmap_overlay,
    save_original_with_token_box,
    save_support_outline_crop_image,
    save_support_mask_image,
)
from .study_protocol import _collect_label_examples
from .utils import write_json


def _session_dir(config: EvalConfig, session_name: str) -> Path:
    return config.human_study_root / session_name


def _select_features_for_session(
    config: EvalConfig,
    feature_bank: dict[str, Any],
    *,
    features_per_block: int,
    seed: int,
    feature_specs: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if feature_specs:
        feature_lookup = {
            (int(feature["block_idx"]), int(feature["feature_id"])): feature
            for block in feature_bank["blocks"].values()
            for feature in block["features"]
        }
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for block_idx, feature_id in feature_specs:
            key = (int(block_idx), int(feature_id))
            feature = feature_lookup.get(key)
            if feature is None:
                missing.append(f"{block_idx}:{feature_id}")
                continue
            selected.append(feature)
        if missing:
            raise ValueError(f"Unknown feature specs for human session: {', '.join(missing)}")
        return selected
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for block_idx in config.blocks:
        features = list(feature_bank["blocks"][str(block_idx)]["features"])
        rng.shuffle(features)
        selected.extend(features[: int(features_per_block)])
    return selected


def _item_relpath(base_dir: Path, path: Path) -> str:
    return str(path.relative_to(base_dir))


def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f5f5; color: #111; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .feature {{ background: white; border: 1px solid #ddd; padding: 16px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }}
    .sample {{ background: #fafafa; border: 1px solid #e5e5e5; padding: 10px; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    .panel {{ background: #fafafa; border: 1px solid #e5e5e5; padding: 10px; }}
    img {{ width: 100%; max-width: 320px; border: 1px solid #ccc; display: block; }}
    .meta {{ font-size: 12px; color: #555; margin-top: 4px; }}
    .hint {{ font-size: 13px; color: #333; margin-bottom: 16px; }}
    code {{ background: #eee; padding: 1px 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def build_human_session(
    config: EvalConfig,
    *,
    session_name: str = "pilot_v1",
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    examples_per_feature: int | None = None,
) -> dict[str, Any]:
    feature_bank = load_feature_bank(config)
    features_per_block = int(features_per_block or config.human_session_default_features_per_block)
    seed = int(seed if seed is not None else config.human_session_default_seed)
    session_dir = _session_dir(config, session_name)
    session_dir.mkdir(parents=True, exist_ok=True)
    selected_features = _select_features_for_session(
        config,
        feature_bank,
        features_per_block=features_per_block,
        seed=seed,
        feature_specs=feature_specs,
    )
    runtime = LegacyRuntime(config)
    feature_entries: list[dict[str, Any]] = []
    label_example_target = int(examples_per_feature or config.study_label_examples_per_feature)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    try:
        total = len(selected_features)
        for idx, feature in enumerate(selected_features, start=1):
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            feature_dir = session_dir / "labeling_assets" / feature_key.replace("/", "__")
            train_entries: list[dict[str, Any]] = []
            label_examples = _collect_label_examples(
                config=config,
                runtime=runtime,
                feature=feature,
                target_count=label_example_target,
                frame_cache=frame_cache,
                sid_to_path_cache=sid_to_path_cache,
            )
            for rank, row in enumerate(label_examples):
                image_path = str(row["image_path"])
                token_idx = int(row["target_patch_idx"])
                original_path = feature_dir / f"train_{rank:02d}_original_token.png"
                actmap_path = feature_dir / f"train_{rank:02d}_feature_actmap.png"
                erf_path = feature_dir / f"train_{rank:02d}_feature_erf_heatmap.png"
                erf_support_path = feature_dir / f"train_{rank:02d}_feature_erf_support.png"
                erf_zoom_path = feature_dir / f"train_{rank:02d}_feature_erf_zoom.png"
                erf_json_path = feature_dir / f"train_{rank:02d}_feature_erf.json"

                save_original_with_token_box(image_path, original_path, token_idx)
                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
                erf = runtime.cautious_feature_erf(image_path, block_idx, token_idx, feature_id)
                save_erf_heatmap_image(image_path, erf["prob_scores"], erf_path, token_idx=token_idx)
                save_support_mask_image(image_path, erf["support_indices"], erf_support_path, token_idx=token_idx)
                zoom_meta = save_support_outline_crop_image(
                    image_path,
                    erf["support_indices"],
                    erf_zoom_path,
                    score_map=erf["prob_scores"],
                )
                write_json(erf_json_path, erf)
                train_entries.append(
                    {
                        "rank": rank,
                        "sample_id": int(row["sample_id"]),
                        "token_idx": token_idx,
                        "label_example_source": str(row.get("label_example_source", "train")),
                        "image_path": image_path,
                        "original_with_token_box": _item_relpath(session_dir, original_path),
                        "feature_actmap": _item_relpath(session_dir, actmap_path),
                        "feature_erf": _item_relpath(session_dir, erf_support_path),
                        "feature_erf_heatmap": _item_relpath(session_dir, erf_path),
                        "feature_erf_zoom": _item_relpath(session_dir, erf_zoom_path),
                        "feature_erf_zoom_meta": zoom_meta,
                        "feature_erf_support": _item_relpath(session_dir, erf_support_path),
                        "feature_erf_json": _item_relpath(session_dir, erf_json_path),
                        "feature_erf_support_threshold": float(erf["support_threshold"]),
                    }
                )
            feature_entries.append(
                {
                    "feature_key": feature_key,
                    "block_idx": block_idx,
                    "feature_id": feature_id,
                    "train_examples": train_entries,
                    "holdout_rows": feature["holdout"],
                }
            )
            print(
                f"[human-session {idx:03d}/{total:03d}] block={block_idx} feature={feature_id}",
                flush=True,
            )
    finally:
        runtime.close()

    human_labels_json = session_dir / "human_labels.json"
    if not human_labels_json.exists():
        write_json(
            human_labels_json,
            {
                entry["feature_key"]: {
                    "predicted_label": "",
                    "notes": "",
                }
                for entry in feature_entries
            },
        )

    manifest = {
        "config": config.to_dict(),
        "session_name": session_name,
        "features_per_block": features_per_block,
        "seed": seed,
        "label_examples_per_feature": label_example_target,
        "selected_features": feature_entries,
        "human_labels_json": str(human_labels_json),
    }
    write_json(session_dir / "session_manifest.json", manifest)

    sections: list[str] = [
        "<h1>Human Labeling Session</h1>",
        "<p class='hint'>"
        f"For each feature, inspect the {label_example_target} labeling examples. "
        "Each sample shows: original image with target token box, a feature SAE activation localization panel where cyan indicates activation overlay only, a feature-conditioned ERF zoom crop with black outside-support fill, and a feature-conditioned ERF quantitative heatmap. "
        "Write your guessed label into <code>human_labels.json</code>. "
        "Do not edit <code>session_manifest.json</code>."
        "</p>",
    ]
    for entry in feature_entries:
        train_html = []
        for sample in entry["train_examples"]:
            zoom_html = (
                f'<img src="{sample["feature_erf_zoom"]}" alt="feature-erf-zoom">'
                '<div class="meta">Feature-conditioned ERF zoom crop with black outside-support fill</div>'
            )
            heatmap_html = (
                f'<img src="{sample["feature_erf_heatmap"]}" alt="feature-erf-heatmap">'
                '<div class="meta">Feature-conditioned ERF quantitative heatmap</div>'
            )
            train_html.append(
                f"""
                <div class="sample">
                  <div><strong>Sample {sample['rank'] + 1}</strong></div>
                  <div class="meta">sample_id={sample['sample_id']} token={sample['token_idx']}</div>
                  <img src="{sample['original_with_token_box']}" alt="original">
                  <div class="meta">Original + target token</div>
                  <img src="{sample['feature_actmap']}" alt="actmap">
                  <div class="meta">Feature SAE activation localization panel</div>
                  {zoom_html}
                  {heatmap_html}
                </div>
                """
            )
        sections.append(
            f"""
            <div class="feature">
              <h2>{entry['feature_key']}</h2>
              <div class="meta">block={entry['block_idx']} feature_id={entry['feature_id']}</div>
              <div class="grid">
                {''.join(train_html)}
              </div>
            </div>
            """
        )
    (session_dir / "labeling.html").write_text(_html_page("Human Labeling Session", "\n".join(sections)))
    return manifest


def build_human_quiz(
    config: EvalConfig,
    *,
    session_name: str = "pilot_v1",
    seed: int | None = None,
) -> dict[str, Any]:
    session_dir = _session_dir(config, session_name)
    manifest = json.loads((session_dir / "session_manifest.json").read_text())
    human_labels = json.loads((session_dir / "human_labels.json").read_text())
    seed = int(seed if seed is not None else manifest.get("seed", config.human_session_default_seed))
    rng = random.Random(seed + 997)
    runtime = LegacyRuntime(config)
    questions: list[dict[str, Any]] = []
    skipped_unlabeled_features: list[str] = []
    try:
        total = sum(len(entry["holdout_rows"]) for entry in manifest["selected_features"])
        q_idx = 0
        for entry in manifest["selected_features"]:
            feature_key = str(entry["feature_key"])
            predicted_label = str(human_labels.get(feature_key, {}).get("predicted_label", "")).strip()
            if not predicted_label:
                skipped_unlabeled_features.append(feature_key)
                continue
            block_idx = int(entry["block_idx"])
            feature_id = int(entry["feature_id"])
            for holdout_rank, row in enumerate(entry["holdout_rows"]):
                q_idx += 1
                image_path = str(row["image_path"])
                sample_id = int(row["sample_id"])
                positive_idx = int(row["target_patch_idx"])
                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                candidate_indices = [idx for idx in range(config.n_patches) if idx != positive_idx]
                candidate_indices = sorted(candidate_indices, key=lambda idx: (float(actmap[idx]), idx))
                negative_idx = int(candidate_indices[0])

                pos_uid = f"block_{block_idx}/sample_{sample_id}/tok_{positive_idx}"
                neg_uid = f"block_{block_idx}/sample_{sample_id}/tok_{negative_idx}"
                pos_dir = session_dir / "quiz_assets" / feature_key.replace("/", "__") / f"q{holdout_rank+1}_positive"
                neg_dir = session_dir / "quiz_assets" / feature_key.replace("/", "__") / f"q{holdout_rank+1}_negative"

                pos_original = pos_dir / "original.png"
                pos_erf = pos_dir / "erf.png"
                pos_erf_support = pos_dir / "erf_support.png"
                pos_erf_zoom = pos_dir / "erf_zoom.png"
                pos_erf_json = pos_dir / "erf.json"
                pos_cos = pos_dir / "cosine.png"
                save_original_with_token_box(image_path, pos_original, positive_idx)
                pos_erf_payload = runtime.cautious_token_erf(image_path, block_idx, positive_idx)
                save_erf_heatmap_image(image_path, pos_erf_payload["prob_scores"], pos_erf, token_idx=positive_idx)
                save_support_mask_image(image_path, pos_erf_payload["support_indices"], pos_erf_support, token_idx=positive_idx)
                pos_zoom_meta = save_support_outline_crop_image(
                    image_path,
                    pos_erf_payload["support_indices"],
                    pos_erf_zoom,
                    score_map=pos_erf_payload["prob_scores"],
                )
                write_json(pos_erf_json, pos_erf_payload)
                pos_cos_map = runtime.token_cosine_map(image_path, block_idx, positive_idx)
                save_cosine_overlay_image(image_path, pos_cos_map, pos_cos, token_idx=positive_idx)

                neg_original = neg_dir / "original.png"
                neg_erf = neg_dir / "erf.png"
                neg_erf_support = neg_dir / "erf_support.png"
                neg_erf_zoom = neg_dir / "erf_zoom.png"
                neg_erf_json = neg_dir / "erf.json"
                neg_cos = neg_dir / "cosine.png"
                save_original_with_token_box(image_path, neg_original, negative_idx)
                neg_erf_payload = runtime.cautious_token_erf(image_path, block_idx, negative_idx)
                save_erf_heatmap_image(image_path, neg_erf_payload["prob_scores"], neg_erf, token_idx=negative_idx)
                save_support_mask_image(image_path, neg_erf_payload["support_indices"], neg_erf_support, token_idx=negative_idx)
                neg_zoom_meta = save_support_outline_crop_image(
                    image_path,
                    neg_erf_payload["support_indices"],
                    neg_erf_zoom,
                    score_map=neg_erf_payload["prob_scores"],
                )
                write_json(neg_erf_json, neg_erf_payload)
                neg_cos_map = runtime.token_cosine_map(image_path, block_idx, negative_idx)
                save_cosine_overlay_image(image_path, neg_cos_map, neg_cos, token_idx=negative_idx)

                positive_option = "A" if rng.random() < 0.5 else "B"
                option_map = {
                    positive_option: {
                        "token_uid": pos_uid,
                        "token_idx": positive_idx,
                        "original": _item_relpath(session_dir, pos_original),
                        "erf": _item_relpath(session_dir, pos_erf_support),
                        "erf_heatmap": _item_relpath(session_dir, pos_erf),
                        "erf_zoom": _item_relpath(session_dir, pos_erf_zoom),
                        "erf_zoom_meta": pos_zoom_meta,
                        "cosine": _item_relpath(session_dir, pos_cos),
                    },
                    "B" if positive_option == "A" else "A": {
                        "token_uid": neg_uid,
                        "token_idx": negative_idx,
                        "original": _item_relpath(session_dir, neg_original),
                        "erf": _item_relpath(session_dir, neg_erf_support),
                        "erf_heatmap": _item_relpath(session_dir, neg_erf),
                        "erf_zoom": _item_relpath(session_dir, neg_erf_zoom),
                        "erf_zoom_meta": neg_zoom_meta,
                        "cosine": _item_relpath(session_dir, neg_cos),
                    },
                }
                questions.append(
                    {
                        "question_id": f"{feature_key}::holdout_{holdout_rank+1}",
                        "feature_key": feature_key,
                        "block_idx": block_idx,
                        "feature_id": feature_id,
                        "sample_id": sample_id,
                        "predicted_label": predicted_label,
                        "options": option_map,
                        "answer": positive_option,
                    }
                )
                print(
                    f"[human-quiz {q_idx:03d}/{total:03d}] block={block_idx} feature={feature_id} sample={sample_id}",
                    flush=True,
                )
    finally:
        runtime.close()

    if not questions:
        raise ValueError(
            f"No labeled features found in {session_dir / 'human_labels.json'}. "
            "Fill at least one predicted_label before building the quiz."
        )

    quiz_answers_json = session_dir / "quiz_answers.json"
    existing_answers = json.loads(quiz_answers_json.read_text()) if quiz_answers_json.exists() else {}
    write_json(
        quiz_answers_json,
        {
            question["question_id"]: {
                "selected_option": str(existing_answers.get(question["question_id"], {}).get("selected_option", "")),
            }
            for question in questions
        },
    )

    answer_key = {question["question_id"]: question["answer"] for question in questions}
    write_json(session_dir / "quiz_answer_key.json", answer_key)
    quiz_manifest = {
        "config": config.to_dict(),
        "session_name": session_name,
        "questions": questions,
        "quiz_answers_json": str(quiz_answers_json),
        "n_labeled_features": len({str(question["feature_key"]) for question in questions}),
        "skipped_unlabeled_features": skipped_unlabeled_features,
    }
    write_json(session_dir / "quiz_manifest.json", quiz_manifest)

    sections: list[str] = [
        "<h1>Human Axis 1 Quiz</h1>",
        "<p class='hint'>"
        "For each question, one option matches the feature label and one option is a wrong token from the same image. "
        "Use only the boxed token, binary ERF support map, and cosine map. "
        "Record your choices in <code>quiz_answers.json</code> using <code>A</code> or <code>B</code>."
        "</p>",
    ]
    for question in questions:
        option_html = []
        for option_name in ("A", "B"):
            option = question["options"][option_name]
            zoom_html = (
                f'<img src="{option["erf_zoom"]}" alt="erf-zoom">'
                '<div class="meta">ERF zoom crop with black outside-support fill</div>'
            )
            option_html.append(
                f"""
                <div class="panel">
                  <h3>Option {option_name}</h3>
                  <img src="{option['original']}" alt="original">
                  <div class="meta">Original + token box (token={option['token_idx']})</div>
                  <img src="{option['erf']}" alt="erf">
                  <div class="meta">Binary token ERF support</div>
                  {zoom_html}
                  <img src="{option['cosine']}" alt="cos">
                  <div class="meta">Token neighbor cosine map</div>
                </div>
                """
            )
        sections.append(
            f"""
            <div class="feature">
              <h2>{question['question_id']}</h2>
              <div><strong>Feature label:</strong> {question['predicted_label']}</div>
              <div class="meta">block={question['block_idx']} feature_id={question['feature_id']} sample_id={question['sample_id']}</div>
              <div class="pair">
                {''.join(option_html)}
              </div>
            </div>
            """
        )
    (session_dir / "quiz.html").write_text(_html_page("Human Axis 1 Quiz", "\n".join(sections)))
    return quiz_manifest


def score_human_quiz(
    config: EvalConfig,
    *,
    session_name: str = "pilot_v1",
) -> dict[str, Any]:
    session_dir = _session_dir(config, session_name)
    quiz_manifest = json.loads((session_dir / "quiz_manifest.json").read_text())
    answer_key = json.loads((session_dir / "quiz_answer_key.json").read_text())
    answers = json.loads((session_dir / "quiz_answers.json").read_text())

    total = 0
    correct = 0
    by_block: dict[str, dict[str, int]] = {}
    for question in quiz_manifest["questions"]:
        qid = str(question["question_id"])
        gold = str(answer_key[qid]).strip().upper()
        pred = str(answers.get(qid, {}).get("selected_option", "")).strip().upper()
        if pred not in {"A", "B"}:
            continue
        block = str(question["block_idx"])
        by_block.setdefault(block, {"total": 0, "correct": 0})
        by_block[block]["total"] += 1
        total += 1
        if pred == gold:
            by_block[block]["correct"] += 1
            correct += 1

    summary = {
        "session_name": session_name,
        "answered": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "by_block": {
            block: {
                **stats,
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for block, stats in by_block.items()
        },
    }
    write_json(session_dir / "quiz_score_summary.json", summary)
    return summary
