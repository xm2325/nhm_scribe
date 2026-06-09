from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from herbarium_scribe.config import load_config
from herbarium_scribe.download import safe_filename
from herbarium_scribe.evaluate import evidence_proxy, field_exact_match, field_token_f1
from herbarium_scribe.llm_backends import call_llm_with_metadata
from herbarium_scribe.metadata import clean_str
from herbarium_scribe.paths import ensure_dirs
from herbarium_scribe.qwen_vision import parse_qwen_vision_output, primary_label_vision_messages
from herbarium_scribe.schema import EXTRACTION_FIELDS, flatten_record, validate_record


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def query_models(base_url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = [
            clean_str(item.get("id"))
            for item in body.get("data", [])
            if isinstance(item, dict) and clean_str(item.get("id"))
        ]
        return {"reachable": True, "models": models, "error": ""}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "models": [], "error": f"http_error:{exc.code}"}
    except Exception as exc:
        return {"reachable": False, "models": [], "error": f"{type(exc).__name__}:{exc}"}


def field_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for field, group in detail.groupby("field", sort=False):
        evaluable = group[group["evaluable"]]
        filled = evaluable[evaluable["prediction"].astype(str).str.strip().ne("")]
        rows.append({
            "field": field,
            "evaluable_field_units": len(evaluable),
            "coverage": len(filled) / len(evaluable) if len(evaluable) else float("nan"),
            "exact_match": evaluable["exact_match"].mean() if len(evaluable) else float("nan"),
            "token_f1": evaluable["token_f1"].mean() if len(evaluable) else float("nan"),
            "transcription_gold_evidence_proxy": evaluable["qwen_transcription_gold_proxy"].mean()
            if len(evaluable)
            else float("nan"),
            "tesseract_gold_evidence_proxy": evaluable["tesseract_gold_proxy"].mean()
            if len(evaluable)
            else float("nan"),
            "unsupported_prediction_rate": filled["unsupported_prediction"].mean()
            if len(filled)
            else float("nan"),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


def write_report(
    path: Path,
    *,
    eval_count: int,
    primary_count: int,
    parsed_count: int,
    model_probe: dict[str, Any],
    preflight: dict[str, Any],
    requested_model: str,
    actual_models: list[str],
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    outputs: list[dict[str, Any]],
) -> None:
    evaluable = detail[detail["evaluable"]] if len(detail) else detail
    filled = evaluable[evaluable["prediction"].astype(str).str.strip().ne("")] if len(evaluable) else evaluable
    overall = pd.DataFrame([{
        "records": eval_count,
        "records_with_primary_label": primary_count,
        "nonempty_raw_outputs": sum(bool(clean_str(row.get("raw_output"))) for row in outputs),
        "parsed_outputs": parsed_count,
        "coverage": len(filled) / len(evaluable) if len(evaluable) else float("nan"),
        "exact_match": evaluable["exact_match"].mean() if len(evaluable) else float("nan"),
        "token_f1": evaluable["token_f1"].mean() if len(evaluable) else float("nan"),
        "qwen_transcription_gold_proxy": evaluable["qwen_transcription_gold_proxy"].mean()
        if len(evaluable)
        else float("nan"),
        "tesseract_gold_proxy": evaluable["tesseract_gold_proxy"].mean()
        if len(evaluable)
        else float("nan"),
        "unsupported_prediction_rate": filled["unsupported_prediction"].mean()
        if len(filled)
        else float("nan"),
    }])
    lines = [
        "# Hespi v11 Qwen Primary Label Vision Report\n",
        "## Configuration\n",
        f"- EVAL records: `{eval_count}`\n",
        f"- Records with an automatically detected primary label: `{primary_count}`\n",
        f"- Requested Qwen model: `{requested_model}`\n",
        f"- Actual model(s): `{', '.join(actual_models)}`\n",
        f"- Model endpoint reachable: `{model_probe.get('reachable')}`\n",
        f"- Models endpoint error: `{model_probe.get('error', '')}`\n",
        f"- Models advertised by endpoint: `{', '.join(model_probe.get('models', []))}`\n",
        f"- Text-chat preflight actual model: `{preflight.get('actual_model', '')}`\n",
        f"- Text-chat preflight error: `{preflight.get('error_message', '')}`\n",
        "\n## Overall result\n",
        markdown_table(overall),
        "\n\n## Field result\n",
        markdown_table(summary),
        "\n\n## Interpretation\n",
        "- `qwen_transcription_gold_proxy` and `tesseract_gold_proxy` only test whether gold field values "
        "appear in the respective transcription after normalization. They are not OCR CER/WER.\n",
        "- Qwen extraction metrics evaluate fields returned directly from the primary-label image. "
        "Values absent from the primary label should remain empty rather than be inferred from the full sheet.\n",
        "- A strong visual model can still hallucinate fluent text. Review raw transcription, uncertain spans, "
        "and exact evidence before treating a field as supported.\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def build_streamlit_bundle(
    report_dir: Path,
    outputs: list[dict[str, Any]],
    crop_rows: list[dict[str, Any]],
    files: list[Path],
) -> Path:
    bundle = report_dir / "streamlit_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    crop_dir = bundle / "primary_labels"
    crop_dir.mkdir(parents=True)
    copied_rows = []
    for row in crop_rows:
        source = Path(row["crop_path"])
        occurrence_id = row["occurrenceID"]
        index = int(row["image_index"])
        target = crop_dir / f"{safe_filename(occurrence_id)}_{index:02d}.jpg"
        shutil.copy2(source, target)
        copied_rows.append({**row, "bundle_crop_path": str(target.relative_to(bundle))})
    pd.DataFrame(copied_rows).to_csv(bundle / "primary_label_manifest.csv", index=False)
    for source in files:
        if source.exists():
            shutil.copy2(source, bundle / source.name)
    with (bundle / "qwen_vision_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    archive = report_dir / "hespi_v11_qwen_streamlit_bundle.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source in sorted(bundle.rglob("*")):
            if source.is_file():
                zf.write(source, Path("hespi_v11_qwen_streamlit_bundle") / source.relative_to(bundle))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hespi_v11_eval10_qwen_vision.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    eval_set = read_csv(paths["processed"] / "eval_set.csv")
    layout = read_csv(paths["processed"] / "layout_boxes.csv")
    ocr = read_csv(paths["processed"] / "ocr_by_region.csv")
    eval_ids = set(eval_set["occurrenceID"])
    primary = layout[
        layout["occurrenceID"].isin(eval_ids)
        & layout["evidence_source"].astype(str).eq("primary_label")
    ].copy()
    if primary.empty:
        raise SystemExit("No primary-label crops were produced.")
    primary = primary.sort_values(["occurrenceID", "region_id"])
    primary["image_index"] = primary.groupby("occurrenceID").cumcount() + 1

    vision_cfg = cfg.get("vision_llm", {})
    base_url = os.environ.get("QWEN_BASE_URL") or clean_str(vision_cfg.get("base_url"))
    requested_model = os.environ.get("QWEN_MODEL") or clean_str(vision_cfg.get("model_name"))
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    if not api_key:
        raise SystemExit("QWEN_API_KEY is not set.")
    model_probe = query_models(base_url, api_key)
    model_probe_path = paths["processed"] / "qwen_model_probe.json"
    model_probe_path.write_text(
        json.dumps(model_probe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    call_cfg = {"llm": dict(vision_cfg)}
    call_cfg["llm"]["backend"] = "qwen_api"
    preflight_meta = call_llm_with_metadata(
        [{
            "role": "user",
            "content": (
                "Return exactly this JSON object to verify text chat access: "
                '{"status":"ok","capability":"text_chat"}'
            ),
        }],
        call_cfg,
    )
    preflight = {
        "requested_model": preflight_meta.get("requested_model", requested_model),
        "actual_model": preflight_meta.get("actual_model", ""),
        "content": clean_str(preflight_meta.get("content")),
        "finish_reason": preflight_meta.get("finish_reason", ""),
        "usage": preflight_meta.get("usage", {}),
        "error_message": preflight_meta.get("error_message", ""),
        "response_body": preflight_meta.get("response", {}),
    }
    preflight_path = paths["processed"] / "qwen_text_chat_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    provider_blocker = clean_str(preflight["error_message"])

    ocr_primary = ocr[
        ocr["occurrenceID"].isin(eval_ids)
        & ocr.get("evidence_source", pd.Series("", index=ocr.index)).astype(str).eq("primary_label")
    ].groupby("occurrenceID")["ocr_text"].apply("\n\n".join).to_dict()
    ocr_by_region = {
        clean_str(row.get("region_id")): clean_str(row.get("ocr_text"))
        for _, row in ocr.iterrows()
        if clean_str(row.get("region_id"))
    }

    outputs: list[dict[str, Any]] = []
    predictions = []
    crop_rows = []
    gold_by_id = eval_set.set_index("occurrenceID")
    detail_rows = []
    for occurrence_id in eval_set["occurrenceID"]:
        crop_group = primary[primary["occurrenceID"] == occurrence_id]
        crop_paths = [clean_str(value) for value in crop_group["crop_path"] if clean_str(value)]
        for _, crop in crop_group.iterrows():
            crop_rows.append({
                "occurrenceID": occurrence_id,
                "catalogNumber": clean_str(gold_by_id.loc[occurrence_id].get("catalogNumber")),
                "image_index": int(crop["image_index"]),
                "region_id": clean_str(crop.get("region_id")),
                "crop_path": clean_str(crop["crop_path"]),
                "bbox": clean_str(crop.get("bbox")),
                "layout_confidence": clean_str(crop.get("layout_confidence")),
                "tesseract_text": ocr_by_region.get(clean_str(crop.get("region_id")), ""),
            })
        if not crop_paths:
            outputs.append({
                "occurrenceID": occurrence_id,
                "status": "no_primary_label",
                "requested_model": requested_model,
                "raw_output": "",
                "parsed_json": None,
            })
            continue
        if provider_blocker:
            outputs.append({
                "occurrenceID": occurrence_id,
                "status": "skipped_after_preflight_error",
                "requested_model": requested_model,
                "raw_output": "",
                "parsed_json": None,
                "error_message": provider_blocker,
            })
            continue
        messages, image_meta = primary_label_vision_messages(
            crop_paths,
            max_dimension=int(vision_cfg.get("max_image_dimension", 2200)),
        )
        meta = call_llm_with_metadata(messages, call_cfg)
        raw = clean_str(meta.get("content"))
        parsed = parse_qwen_vision_output(raw)
        message_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "system": messages[0]["content"],
                    "instruction": messages[1]["content"][0]["text"],
                    "images": image_meta,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        item = {
            "occurrenceID": occurrence_id,
            "status": "parsed" if parsed else ("empty_raw_output" if not raw else "parse_failure"),
            "backend": "qwen_api",
            "requested_model": meta.get("requested_model", requested_model),
            "actual_model": meta.get("actual_model", ""),
            "base_url": meta.get("base_url", base_url),
            "image_metadata": image_meta,
            "prompt_text": messages[1]["content"][0]["text"],
            "prompt_sha256": message_fingerprint,
            "raw_output": raw,
            "raw_output_length": len(raw),
            "parsed_json": parsed,
            "finish_reason": meta.get("finish_reason", ""),
            "reasoning_content": meta.get("reasoning_content", ""),
            "usage": meta.get("usage", {}),
            "error_message": meta.get("error_message", ""),
            "response_message_keys": meta.get("message_keys", []),
            "response_body": meta.get("response", {}),
        }
        outputs.append(item)
        if "HTTP 401" in clean_str(item["error_message"]) or "HTTP 403" in clean_str(item["error_message"]):
            provider_blocker = clean_str(item["error_message"])
        if not parsed:
            continue
        record = validate_record(parsed["fields"])
        flat = flatten_record(record)
        flat.update({
            "occurrenceID": occurrence_id,
            "method": "qwen_primary_label_vision",
            "full_transcription": parsed["full_transcription"],
            "parse_failure": False,
        })
        predictions.append(flat)
        gold = gold_by_id.loc[occurrence_id]
        tesseract_text = clean_str(ocr_primary.get(occurrence_id, ""))
        qwen_text = clean_str(parsed.get("full_transcription"))
        for field in EXTRACTION_FIELDS:
            gold_value = clean_str(gold.get(field))
            prediction = clean_str(record[field]["value"])
            evidence_span = clean_str(record[field]["evidence_span"])
            evaluable = bool(gold_value)
            prediction_support = evidence_proxy(prediction, qwen_text) if prediction else None
            evidence_support = evidence_proxy(evidence_span, qwen_text) if evidence_span else None
            unsupported_prediction = bool(
                prediction
                and (prediction_support is None or prediction_support < 1.0)
                and (evidence_support is None or evidence_support < 1.0)
            )
            detail_rows.append({
                "occurrenceID": occurrence_id,
                "field": field,
                "gold": gold_value,
                "prediction": prediction,
                "confidence": record[field]["confidence"],
                "evidence_span": evidence_span,
                "prediction_transcription_support": prediction_support,
                "evidence_transcription_support": evidence_support,
                "unsupported_prediction": unsupported_prediction,
                "evaluable": evaluable,
                "coverage": int(bool(prediction)) if evaluable else float("nan"),
                "exact_match": field_exact_match(field, prediction, gold_value)
                if evaluable
                else float("nan"),
                "token_f1": field_token_f1(field, prediction, gold_value)
                if evaluable
                else float("nan"),
                "qwen_transcription_gold_proxy": evidence_proxy(gold_value, qwen_text)
                if evaluable
                else float("nan"),
                "tesseract_gold_proxy": evidence_proxy(gold_value, tesseract_text)
                if evaluable
                else float("nan"),
            })

    output_jsonl = paths["llm"] / "qwen_primary_label_vision_outputs.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    prediction_df = pd.DataFrame(predictions)
    detail = pd.DataFrame(detail_rows)
    summary = field_metrics(detail)
    prediction_path = paths["processed"] / "qwen_primary_label_predictions.csv"
    detail_path = paths["processed"] / "qwen_primary_label_evaluation_detail.csv"
    summary_path = paths["processed"] / "qwen_primary_label_evaluation_summary.csv"
    manifest_path = paths["processed"] / "qwen_primary_label_manifest.csv"
    prediction_df.to_csv(prediction_path, index=False)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(crop_rows).to_csv(manifest_path, index=False)
    actual_models = sorted({
        clean_str(row.get("actual_model"))
        for row in outputs
        if clean_str(row.get("actual_model"))
    })
    report_path = paths["reports"] / clean_str(
        cfg.get("outputs", {}).get("report_name", "hespi_v11_qwen_vision_report.md")
    )
    write_report(
        report_path,
        eval_count=len(eval_set),
        primary_count=primary["occurrenceID"].nunique(),
        parsed_count=len(prediction_df),
        model_probe=model_probe,
        preflight=preflight,
        requested_model=requested_model,
        actual_models=actual_models,
        detail=detail,
        summary=summary,
        outputs=outputs,
    )
    archive = build_streamlit_bundle(
        paths["reports"],
        outputs,
        crop_rows,
        [
            prediction_path,
            detail_path,
            summary_path,
            manifest_path,
            model_probe_path,
            preflight_path,
            report_path,
        ],
    )
    print({
        "eval_records": len(eval_set),
        "primary_label_records": primary["occurrenceID"].nunique(),
        "parsed_outputs": len(prediction_df),
        "requested_model": requested_model,
        "actual_models": actual_models,
        "report": str(report_path),
        "streamlit_bundle": str(archive),
    })


if __name__ == "__main__":
    main()
