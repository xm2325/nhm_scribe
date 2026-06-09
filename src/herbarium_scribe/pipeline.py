from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .paths import ensure_dirs, repo_root, resolve_path
from .metadata import load_metadata, save_metadata_copy, clean_str
from .sampling import make_demo_eval_split, save_split_outputs, stratified_random_sample
from .download import run_download
from .layout import detect_layout
from .ocr import run_ocr
from .extract_rules import extract_rule_based
from .schema import EXTRACTION_FIELDS, flatten_record, validate_record
from .llm_backends import call_llm_with_metadata
from .rag import build_rag_corpus, retrieve_context, format_context_for_prompt
from .reconcile import reconcile_dataframe
from .evaluate import evaluate_predictions
from .graph_export import export_graph
from .report import write_report


def load_runtime(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = load_config(config_path)
    paths = ensure_dirs(cfg)
    return cfg, paths


def _output_prefix(cfg: dict[str, Any]) -> str:
    return clean_str(cfg.get("outputs", {}).get("prefix", ""))


def _prefixed_processed_path(paths: dict[str, Path], cfg: dict[str, Any], suffix: str) -> Path | None:
    prefix = _output_prefix(cfg)
    return paths["processed"] / f"{prefix}_{suffix}" if prefix else None


def _write_prefixed_csv(df: pd.DataFrame, paths: dict[str, Path], cfg: dict[str, Any], suffix: str) -> None:
    out = _prefixed_processed_path(paths, cfg, suffix)
    if out is not None:
        df.to_csv(out, index=False)


def stage_metadata(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    frozen_dir_value = clean_str(cfg.get("sampling", {}).get("frozen_split_dir", ""))
    if frozen_dir_value:
        frozen_dir = resolve_path(frozen_dir_value, repo_root())
        demo = pd.read_csv(frozen_dir / "demo_set.csv", dtype=str).fillna("")
        eval_df = pd.read_csv(frozen_dir / "eval_set.csv", dtype=str).fillna("")
        summary = pd.read_csv(frozen_dir / "split_summary.csv", dtype=str).fillna("")
        original_demo_size = len(demo)
        original_eval_size = len(eval_df)
        sampling_cfg = cfg.get("sampling", {})
        seed = int(cfg.get("project", {}).get("random_state", 42))
        by = sampling_cfg.get("stratify_by", "institutionCode")
        frozen_demo_size = int(sampling_cfg.get("frozen_demo_size", len(demo)))
        frozen_eval_size = int(sampling_cfg.get("frozen_eval_size", len(eval_df)))
        if frozen_demo_size < len(demo):
            demo = stratified_random_sample(demo, frozen_demo_size, by, seed + 201).reset_index(drop=True)
        if frozen_eval_size < len(eval_df):
            eval_df = stratified_random_sample(eval_df, frozen_eval_size, by, seed + 202).reset_index(drop=True)
        if frozen_demo_size < original_demo_size or frozen_eval_size < original_eval_size:
            rows = []
            for split_name, frame in [("DEMO_SET", demo), ("EVAL_SET", eval_df)]:
                counts = frame[by].value_counts(dropna=False).to_dict() if by in frame.columns else {"all": len(frame)}
                for stratum, count in counts.items():
                    rows.append({
                        "split": split_name,
                        "stratify_by": by,
                        "stratum": stratum,
                        "n": int(count),
                        "split_mode": "frozen_stratified_subset",
                    })
            summary = pd.DataFrame(rows)
        overlap = set(demo["occurrenceID"]) & set(eval_df["occurrenceID"])
        if overlap:
            raise ValueError(f"Frozen DEMO_SET and EVAL_SET overlap: {sorted(overlap)[:5]}")
        save_metadata_copy(pd.concat([demo, eval_df], ignore_index=True), paths)
    else:
        df = load_metadata(cfg)
        save_metadata_copy(df, paths)
        demo, eval_df, summary = make_demo_eval_split(df, cfg)
    save_split_outputs(demo, eval_df, summary, paths["processed"])
    _write_prefixed_csv(demo, paths, cfg, "demo_set.csv")
    _write_prefixed_csv(eval_df, paths, cfg, "eval_set.csv")
    _write_prefixed_csv(summary, paths, cfg, "split_summary.csv")
    print("DEMO_SET summary")
    print(demo[["occurrenceID", "institutionCode", "catalogNumber"]].to_string(index=False))
    print("EVAL_SET summary")
    print(eval_df[["occurrenceID", "institutionCode", "catalogNumber"]].to_string(index=False))
    print("Split by institution")
    print(summary.to_string(index=False))
    return demo, eval_df, summary


def _read_eval(paths: dict[str, Path]) -> pd.DataFrame:
    p = paths["processed"] / "eval_set.csv"
    if not p.exists():
        raise FileNotFoundError("Run metadata stage first: eval_set.csv not found")
    return pd.read_csv(p, dtype=str).fillna("")


def _read_demo(paths: dict[str, Path]) -> pd.DataFrame:
    p = paths["processed"] / "demo_set.csv"
    if not p.exists():
        raise FileNotFoundError("Run metadata stage first: demo_set.csv not found")
    return pd.read_csv(p, dtype=str).fillna("")


def stage_download(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    eval_df = _read_eval(paths)
    demo = _read_demo(paths)
    records = pd.concat([demo, eval_df], ignore_index=True).drop_duplicates("occurrenceID")
    manifest = run_download(records, cfg, paths)
    _write_prefixed_csv(manifest, paths, cfg, "image_manifest.csv")
    return manifest


def stage_layout(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    records = pd.concat([_read_demo(paths), _read_eval(paths)], ignore_index=True).drop_duplicates("occurrenceID")
    manifest_path = paths["processed"] / "image_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("") if manifest_path.exists() else run_download(records, cfg, paths)
    layout = detect_layout(records, manifest, cfg, paths)
    _write_prefixed_csv(layout, paths, cfg, "layout_boxes.csv")
    return layout


def stage_ocr(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    layout_path = paths["processed"] / "layout_boxes.csv"
    if not layout_path.exists():
        layout_df = stage_layout(config_path)
    else:
        layout_df = pd.read_csv(layout_path, dtype=str).fillna("")
    ocr = run_ocr(layout_df, cfg, paths)
    _write_prefixed_csv(ocr, paths, cfg, "ocr_by_region.csv")
    combined_path = paths["processed"] / "ocr_combined.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path, dtype=str).fillna("")
        _write_prefixed_csv(combined, paths, cfg, "ocr_combined.csv")
    return ocr


def require_htr_evidence(
    ocr_df: pd.DataFrame,
    eval_occurrence_ids: set[str],
    config: dict[str, Any],
) -> None:
    htr_cfg = config.get("ocr", {}).get("handwriting_recognition", {})
    if not htr_cfg.get("enabled", False) or not htr_cfg.get("require_nonempty", False):
        return
    htr = ocr_df[
        ocr_df.get("ocr_engine", pd.Series(index=ocr_df.index, dtype=str))
        .astype(str)
        .str.startswith("hespi_trocr_")
    ].copy()
    if eval_occurrence_ids:
        htr = htr[htr["occurrenceID"].astype(str).isin(eval_occurrence_ids)]
    nonempty = htr.get("ocr_text", pd.Series(index=htr.index, dtype=str)).astype(str).str.strip().ne("")
    if len(htr) and nonempty.any():
        return
    statuses = sorted(
        {
            clean_str(value)
            for value in htr.get("ocr_status", pd.Series(dtype=str)).tolist()
            if clean_str(value)
        }
    )
    detail = "; ".join(statuses[:5]) if statuses else "no HTR rows"
    raise RuntimeError(
        "HTR evidence gate failed before LLM extraction: "
        f"{len(htr)} EVAL HTR attempts, 0 non-empty outputs; {detail}"
    )


FIELD_ALIASES = {
    "catalog_number": "catalogNumber",
    "catalog_no": "catalogNumber",
    "barcode": "catalogNumber",
    "scientific_name": "scientificName",
    "taxon": "scientificName",
    "recorded_by": "recordedBy",
    "collector": "recordedBy",
    "event_date": "eventDate",
    "collection_date": "eventDate",
    "state_province": "stateProvince",
    "state": "stateProvince",
    "province": "stateProvince",
    "decimal_latitude": "decimalLatitude",
    "latitude": "decimalLatitude",
    "decimal_longitude": "decimalLongitude",
    "longitude": "decimalLongitude",
    "type_status": "typeStatus",
}


def _extraction_system_prompt() -> str:
    return (
        "Extract herbarium label fields as JSON. Return one JSON object with keys "
        "catalogNumber, scientificName, recordedBy, eventDate, country, stateProvince, "
        "decimalLatitude, decimalLongitude, and typeStatus. Each field must be an object "
        'like {"value":"verbatim or normalized value","confidence":0.0,"evidence_span":"exact OCR evidence"}. '
        "Use an empty value and empty evidence_span when the OCR does not support a field. "
        "Every non-empty field must include an evidence_span copied exactly from the supplied OCR. "
        "Country, stateProvince, eventDate, recordedBy, and coordinates must remain empty unless "
        "their source text is explicit in OCR. Barcode decoder values are stronger catalogNumber "
        "evidence than numeric job, collection, or image-processing numbers. Catalog-number OCR "
        "ensemble values are ranked hypotheses from repeated readings of the same crop, not "
        "independent confirmations. When several OCR hypotheses or decoded "
        "barcodes disagree, use surrounding OCR and institutional identifier structure; leave "
        "catalogNumber empty if the specimen-level identifier remains ambiguous. TrOCR "
        "handwriting results are supplementary hypotheses for their labelled field crops; "
        "prefer them only when the text is coherent and supported by the same crop or nearby "
        "OCR context. Evidence tagged FIELD=recorded_by may only support recordedBy. Evidence "
        "tagged FIELD=event_date may only support eventDate. Never use either of those HTR "
        "channels to alter catalogNumber, scientificName, country, stateProvince, coordinates, "
        "or typeStatus. Never repair uncertain OCR by guessing. Return JSON only."
    )


def _strip_json_fence(text: str) -> str:
    text = clean_str(text)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _normalise_field_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    if "value" in item:
        return item
    for key in ["text", "answer", "extracted_value", "verbatim", "prediction"]:
        if key in item:
            return {
                "value": item.get(key, ""),
                "confidence": item.get("confidence", 0.5),
                "evidence_span": item.get("evidence_span", item.get(key, "")),
            }
    return item


def _normalise_llm_record(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    for key in ["record", "fields", "extraction", "extracted_record", "data", "result"]:
        if isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    normalised: dict[str, Any] = {}
    for key, value in obj.items():
        canonical = FIELD_ALIASES.get(str(key), FIELD_ALIASES.get(str(key).lower(), str(key)))
        normalised[canonical] = _normalise_field_item(value)
    if not any(field in normalised for field in EXTRACTION_FIELDS):
        return None
    return normalised


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = clean_str(text)
    if not text:
        return None
    text = _strip_json_fence(text.replace("```json", "```"))
    try:
        return _normalise_llm_record(json.loads(text))
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return _normalise_llm_record(json.loads(text[start:end + 1]))
            except Exception:
                return None
    return None


def _llm_static_metadata(cfg: dict[str, Any], backend: str) -> dict[str, Any]:
    lcfg = cfg.get("llm", {})
    if backend in {"deepseek", "deepseek_api"}:
        return {
            "backend": backend,
            "requested_model": os.environ.get("DEEPSEEK_MODEL") or lcfg.get("model_name") or lcfg.get("model") or "deepseek-v4-pro",
            "actual_model": "",
            "content": "",
            "error_message": "",
            "endpoint_reachable": False,
            "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY_SELF")),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL") or lcfg.get("base_url") or "https://api.deepseek.com",
            "min_interval_seconds": float(os.environ.get("DEEPSEEK_MIN_INTERVAL_SECONDS") or lcfg.get("min_interval_seconds") or 0),
            "thinking": lcfg.get("thinking", ""),
            "response_format": lcfg.get("response_format", ""),
        }
    return {
        "backend": backend,
        "requested_model": lcfg.get("model_name") or lcfg.get("model") or "",
        "actual_model": "",
        "content": "",
        "error_message": "",
        "endpoint_reachable": False,
        "api_key_present": False,
        "base_url": lcfg.get("base_url", ""),
        "min_interval_seconds": float(lcfg.get("min_interval_seconds", 0) or 0),
        "thinking": lcfg.get("thinking", ""),
        "response_format": lcfg.get("response_format", ""),
    }


def _normalised_evidence_length(text: str) -> int:
    return len(re.sub(r"[^A-Za-z0-9]+", "", clean_str(text)))


def _llm_evidence_gate_reason(
    occurrence_id: str,
    prompt_text: str,
    image_manifest: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> str:
    gate = cfg.get("llm", {}).get("evidence_gate", {})
    if not bool(gate.get("enabled", False)):
        return ""
    image_row = image_manifest.get(occurrence_id, {})
    if bool(gate.get("require_image", True)):
        image_path = clean_str(image_row.get("image_path", ""))
        if not image_path or not Path(image_path).exists():
            return "image_unavailable"
        if str(image_row.get("paired_eligible", "true")).lower() in {"false", "0", "no"}:
            return "image_not_paired_eligible"
    if not clean_str(prompt_text):
        return "empty_ocr_evidence"
    minimum = int(gate.get("min_alphanumeric_characters", 1))
    if _normalised_evidence_length(prompt_text) < minimum:
        return "insufficient_ocr_evidence"
    return ""


def stage_extract(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    backend = clean_str(cfg.get("llm", {}).get("backend", "none")).lower() or "none"
    method_name = clean_str(cfg.get("method_name", "")) or ("rule_ocr" if backend == "none" else f"{backend}_rag")
    rag_cfg = cfg.get("rag", {})
    rag_enabled = bool(rag_cfg.get("enabled", rag_cfg.get("top_k", 3) != 0))
    top_k = int(rag_cfg.get("top_k", 3 if rag_enabled else 0))
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    ocr_combined = ocr_df.groupby("occurrenceID")["ocr_text"].apply("\n".join).to_dict()
    prompt_column = "ocr_prompt_text" if "ocr_prompt_text" in ocr_df.columns else "ocr_text"
    ocr_prompt_combined = ocr_df.groupby("occurrenceID")[prompt_column].apply("\n\n".join).to_dict()
    manifest_path = paths["processed"] / "image_manifest.csv"
    manifest_df = pd.read_csv(manifest_path, dtype=str).fillna("") if manifest_path.exists() else pd.DataFrame()
    image_manifest = {
        clean_str(row.get("occurrenceID")): row.to_dict()
        for _, row in manifest_df.iterrows()
    }
    demo_df = _read_demo(paths)
    corpus = build_rag_corpus(demo_df if rag_enabled and rag_cfg.get("use_demo_examples", True) else None)
    rows = []
    llm_diag_rows = []
    llm_jsonl = paths["llm"] / "raw_llm_outputs.jsonl"
    rag_jsonl = paths["llm"] / "rag_contexts.jsonl"
    diag_json = paths["processed"] / "llm_diagnostics.json"
    diag_csv = paths["processed"] / "llm_diagnostics.csv"
    if llm_jsonl.exists():
        llm_jsonl.unlink()
    if rag_jsonl.exists():
        rag_jsonl.unlink()
    if diag_json.exists():
        diag_json.unlink()
    if diag_csv.exists():
        diag_csv.unlink()
    for _, gold in eval_df.iterrows():
        occ = clean_str(gold.get("occurrenceID"))
        text = ocr_combined.get(occ, "")
        prompt_text = ocr_prompt_combined.get(occ, text)
        rule_rec = extract_rule_based(text, gold.to_dict())
        flat = flatten_record(rule_rec)
        flat.update({"occurrenceID": occ, "method": "rule_ocr", "parse_failure": False})
        rows.append(flat)
        if backend != "none":
            gate_reason = _llm_evidence_gate_reason(occ, prompt_text, image_manifest, cfg)
            retrieved = retrieve_context(prompt_text, corpus, top_k=top_k) if rag_enabled and top_k > 0 else []
            ctx = format_context_for_prompt(retrieved)
            prompt = f"Context:\n{ctx}\n\nOCR text:\n{prompt_text}" if retrieved else f"OCR text:\n{prompt_text}"
            messages = [
                {
                    "role": "system",
                    "content": _extraction_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ]
            messages_json = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            prompt_sha256 = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()
            with rag_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"occurrenceID": occ, "method": method_name, "backend": backend, "retrieved_context": retrieved}, ensure_ascii=False) + "\n")
            llm_call_attempted = not bool(gate_reason)
            meta = call_llm_with_metadata(messages, cfg) if llm_call_attempted else _llm_static_metadata(cfg, backend)
            raw = clean_str(meta.get("content", ""))
            reasoning_content = clean_str(meta.get("reasoning_content", ""))
            response_body = meta.get("response", {})
            message_keys = meta.get("message_keys", [])
            finish_reason = clean_str(meta.get("finish_reason", ""))
            error_message = clean_str(meta.get("error_message", ""))
            with llm_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "occurrenceID": occ,
                    "method": method_name,
                    "backend": backend,
                    "requested_model": meta.get("requested_model", ""),
                    "actual_model_if_available": meta.get("actual_model", ""),
                    "min_interval_seconds": meta.get("min_interval_seconds", 0.0),
                    "thinking": meta.get("thinking", {}),
                    "response_format": meta.get("response_format", {}),
                    "prompt": prompt,
                    "messages": messages,
                    "prompt_sha256": prompt_sha256,
                    "retrieved_context": retrieved,
                    "raw_output": raw,
                    "raw_output_length": len(raw),
                    "reasoning_content": reasoning_content,
                    "reasoning_content_length": len(reasoning_content),
                    "response_finish_reason": finish_reason,
                    "response_message_keys": message_keys,
                    "response_usage": meta.get("usage", {}),
                    "response_body": response_body,
                    "parsed_json": None,
                    "parse_failure": None,
                    "error_message": error_message,
                    "llm_call_attempted": llm_call_attempted,
                    "skip_reason": gate_reason,
                }, ensure_ascii=False) + "\n")
            obj = _parse_llm_json(raw)
            not_evaluated = bool(gate_reason) or not bool(raw)
            not_evaluated_reason = gate_reason or (error_message if error_message else ("empty_raw_output" if not raw else ""))
            if obj is None:
                obj = {}
                parse_failure = bool(raw) and not not_evaluated
            else:
                parse_failure = False
            llm_rec = validate_record(obj)
            llm_flat = flatten_record(llm_rec)
            llm_flat.update({
                "occurrenceID": occ,
                "method": method_name,
                "parse_failure": parse_failure,
                "not_evaluated": not_evaluated,
                "not_evaluated_reason": not_evaluated_reason if not_evaluated else "",
            })
            rows.append(llm_flat)
            # Rewrite the last JSONL line with parse details for easier artifact inspection.
            raw_lines = llm_jsonl.read_text(encoding="utf-8").splitlines()
            last = json.loads(raw_lines[-1])
            last["parsed_json"] = obj if obj else None
            last["parse_failure"] = parse_failure
            last["not_evaluated"] = not_evaluated
            last["not_evaluated_reason"] = not_evaluated_reason if not_evaluated else ""
            raw_lines[-1] = json.dumps(last, ensure_ascii=False)
            llm_jsonl.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            llm_diag_rows.append({
                "occurrenceID": occ,
                "backend": backend,
                "method": method_name,
                "api_key_present": bool(meta.get("api_key_present", False)),
                "base_url": meta.get("base_url", ""),
                "requested_model": meta.get("requested_model", ""),
                "actual_model_if_available": meta.get("actual_model", ""),
                "endpoint_reachable": bool(meta.get("endpoint_reachable", False)),
                "min_interval_seconds": meta.get("min_interval_seconds", 0.0),
                "thinking": json.dumps(meta.get("thinking", {}), ensure_ascii=False),
                "response_format": json.dumps(meta.get("response_format", {}), ensure_ascii=False),
                "prompt_sha256": prompt_sha256,
                "response_finish_reason": finish_reason,
                "response_message_keys": json.dumps(message_keys, ensure_ascii=False),
                "response_usage": json.dumps(meta.get("usage", {}), ensure_ascii=False),
                "reasoning_content_length": len(reasoning_content),
                "reasoning_content_nonempty": bool(reasoning_content),
                "raw_output_length": len(raw),
                "raw_output_nonempty": bool(raw),
                "rag_context_count": len(retrieved),
                "parse_failure": parse_failure,
                "not_evaluated": not_evaluated,
                "not_evaluated_reason": not_evaluated_reason if not_evaluated else "",
                "error_message": error_message,
                "llm_call_attempted": llm_call_attempted,
                "skip_reason": gate_reason,
                "status": "skipped" if gate_reason else ("not_evaluated" if not_evaluated else ("parsed" if raw and not parse_failure else ("parse_failure" if raw else "empty_raw_output"))),
            })
    out = pd.DataFrame(rows)
    out = out.merge(eval_df[["occurrenceID", "institutionCode"]], on="occurrenceID", how="left")
    out.to_json(paths["processed"] / "extractions.jsonl", orient="records", lines=True, force_ascii=False)
    out.to_csv(paths["processed"] / "extractions_flat.csv", index=False)
    prediction_name = clean_str(cfg.get("outputs", {}).get("prediction_name", ""))
    if prediction_name:
        target_method = method_name if backend != "none" else "rule_ocr"
        out[out["method"] == target_method].to_csv(paths["processed"] / f"{_output_prefix(cfg)}_predictions_{prediction_name}.csv", index=False)
    else:
        _write_prefixed_csv(out, paths, cfg, "predictions.csv")
    if llm_diag_rows:
        diag_df = pd.DataFrame(llm_diag_rows)
        diag_df.to_csv(diag_csv, index=False)
        summary = {
            "backend": backend,
            "method": method_name,
            "records": int(len(diag_df)),
            "llm_calls_attempted": int(diag_df["llm_call_attempted"].sum()),
            "llm_calls_skipped": int((~diag_df["llm_call_attempted"]).sum()),
            "api_key_present": bool(diag_df["api_key_present"].any()),
            "base_url": clean_str(diag_df["base_url"].iloc[0]) if len(diag_df) else "",
            "requested_model": clean_str(diag_df["requested_model"].iloc[0]) if len(diag_df) else "",
            "actual_models": sorted(set(diag_df["actual_model_if_available"].dropna().astype(str)) - {""}),
            "endpoint_reachable": bool(diag_df["endpoint_reachable"].any()),
            "raw_output_nonempty": int(diag_df["raw_output_nonempty"].sum()),
            "empty_raw_outputs": int((~diag_df["raw_output_nonempty"]).sum()),
            "api_empty_responses": int(((~diag_df["raw_output_nonempty"]) & diag_df["endpoint_reachable"] & ~diag_df["not_evaluated_reason"].astype(str).str.contains("API_KEY|authentication_error|rate_limit|provider_error", case=False, na=False)).sum()),
            "api_empty_response_rate": float(((~diag_df["raw_output_nonempty"]) & diag_df["endpoint_reachable"] & ~diag_df["not_evaluated_reason"].astype(str).str.contains("API_KEY|authentication_error|rate_limit|provider_error", case=False, na=False)).mean()),
            "parsed_records": int((diag_df["raw_output_nonempty"] & ~diag_df["parse_failure"] & ~diag_df["not_evaluated"]).sum()),
            "parse_failures": int(diag_df["parse_failure"].sum()),
            "not_evaluated": int(diag_df["not_evaluated"].sum()),
            "rag_context_rows": int(len(diag_df)),
            "rag_contexts_path": str(rag_jsonl),
            "raw_outputs_path": str(llm_jsonl),
        }
        diag_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    llm_outputs_name = clean_str(cfg.get("outputs", {}).get("llm_outputs_name", ""))
    if llm_outputs_name and llm_jsonl.exists():
        (paths["llm"] / llm_outputs_name).write_text(llm_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    rag_contexts_name = clean_str(cfg.get("outputs", {}).get("rag_contexts_name", ""))
    if rag_contexts_name and rag_jsonl.exists():
        (paths["llm"] / rag_contexts_name).write_text(rag_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    llm_diagnostics_name = clean_str(cfg.get("outputs", {}).get("llm_diagnostics_name", ""))
    if llm_diagnostics_name and diag_csv.exists():
        (paths["llm"] / llm_diagnostics_name).write_text(diag_csv.read_text(encoding="utf-8"), encoding="utf-8")
    llm_diagnostics_json_name = clean_str(cfg.get("outputs", {}).get("llm_diagnostics_json_name", ""))
    if llm_diagnostics_json_name and diag_json.exists():
        (paths["llm"] / llm_diagnostics_json_name).write_text(diag_json.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def stage_reconcile(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    p = paths["processed"] / "extractions_flat.csv"
    pred = pd.read_csv(p, dtype=str).fillna("") if p.exists() else stage_extract(config_path)
    out = reconcile_dataframe(pred, paths, cfg)
    prediction_name = clean_str(cfg.get("outputs", {}).get("prediction_name", ""))
    if prediction_name:
        method_name = clean_str(cfg.get("method_name", ""))
        selected = out[out["method"].astype(str).eq(method_name)] if method_name else out
        selected.to_csv(
            paths["processed"]
            / f"{_output_prefix(cfg)}_predictions_{prediction_name}_reconciled.csv",
            index=False,
        )
    return out


def stage_evaluate(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    combine_files = cfg.get("outputs", {}).get("combine_prediction_files", [])
    if combine_files:
        frames = []
        for name in combine_files:
            p = paths["processed"] / name
            if p.exists():
                frames.append(pd.read_csv(p, dtype=str).fillna(""))
        pred = pd.concat(frames, ignore_index=True) if frames else stage_reconcile(config_path)
    else:
        pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
        pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    fields = cfg.get("evaluation", {}).get("fields", [])
    review_config = cfg.get("evaluation", {}).get("review", {})
    detail, summary, strat = evaluate_predictions(pred, eval_df, ocr_df, fields, paths, review_config)
    _write_prefixed_csv(detail, paths, cfg, "evaluation_detail.csv")
    _write_prefixed_csv(summary, paths, cfg, "evaluation_summary.csv")
    _write_prefixed_csv(strat, paths, cfg, "evaluation_by_ocr_tertile.csv")
    return detail, summary, strat


def stage_graph(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
    pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    graph_src = pred.copy()
    if "method" in pred.columns:
        preferred = clean_str(cfg.get("graph", {}).get("preferred_method", ""))
        if preferred and (pred["method"] == preferred).any():
            graph_src = pred[pred["method"] == preferred].copy()
        elif cfg.get("llm", {}).get("backend", "none") != "none":
            llm_rows = pred[pred["method"].astype(str).str.endswith("_rag")].copy()
            graph_src = llm_rows if len(llm_rows) else pred[pred["method"] == "rule_ocr"].copy()
        else:
            graph_src = pred[pred["method"] == "rule_ocr"].copy()
    return export_graph(graph_src, paths)


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    cfg, paths = load_runtime(config_path)
    demo, eval_df, split_summary = stage_metadata(config_path)
    stage_download(config_path)
    stage_layout(config_path)
    ocr_df = stage_ocr(config_path)
    require_htr_evidence(
        ocr_df,
        set(eval_df["occurrenceID"].astype(str)),
        cfg,
    )
    stage_extract(config_path)
    pred = stage_reconcile(config_path)
    detail, summary, strat = stage_evaluate(config_path)
    nodes, edges = stage_graph(config_path)
    report_path = write_report(cfg, split_summary, summary, strat, paths)
    return {
        "demo_n": len(demo),
        "eval_n": len(eval_df),
        "ocr_rows": len(ocr_df),
        "prediction_rows": len(pred),
        "eval_rows": len(detail),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "report_path": str(report_path),
    }
