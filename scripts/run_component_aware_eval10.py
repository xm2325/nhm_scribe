from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from PIL import Image

from herbarium_scribe.component_aware import (
    build_evidence_packets,
    deterministic_reconciliation,
    direct_evidence_packet,
    empty_reconciled_field,
    evidence_packet_text,
    flatten_reconciled_record,
    read_sheet_components,
    reconcile_with_optional_llm,
)
from herbarium_scribe.config import load_config
from herbarium_scribe.download import safe_filename
from herbarium_scribe.evaluate import field_exact_match, field_token_f1, truthy_flag
from herbarium_scribe.hespi_layout import normalise_bbox
from herbarium_scribe.metadata import clean_str
from herbarium_scribe.pipeline import load_runtime, stage_download, stage_layout, stage_metadata
from herbarium_scribe.rag import (
    assert_no_rag_leakage,
    clip_image_embeddings,
    retrieve_hybrid_references,
    tfidf_embeddings,
)
from herbarium_scribe.schema import EXTRACTION_FIELDS

BRANCHES = [
    "baseline_full_sheet",
    "baseline_primary_label",
    "component_aware_no_rag",
    "component_aware_with_rag",
]
TEXT_BEARING_COMPONENTS = {
    "primary_specimen_label",
    "annotation_label",
    "database_label",
    "barcode",
    "number",
    "type_label",
    "stamp",
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_runtime_checkpoints(
    processed: Path,
    retrieval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    reconciled_jsonl: list[dict[str, Any]],
    llm_diagnostics: list[dict[str, Any]],
) -> None:
    pd.DataFrame(retrieval_rows).to_csv(
        processed / "rag_retrieval_results.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        processed / "reconciled_predictions_flat.csv",
        index=False,
    )
    write_jsonl(processed / "reconciled_predictions.jsonl", reconciled_jsonl)
    pd.DataFrame(llm_diagnostics).to_csv(
        processed / "llm_diagnostics.csv",
        index=False,
    )


def normalized_mean_embedding(vectors: list[np.ndarray]) -> np.ndarray | None:
    valid = [
        np.asarray(vector, dtype=np.float32)
        for vector in vectors
        if vector is not None and np.asarray(vector).size
    ]
    if not valid:
        return None
    mean = np.mean(np.stack(valid), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def component_crop_paths(
    components: pd.DataFrame,
    occurrence_id: str,
    *,
    limit: int,
) -> list[str]:
    rows = components[
        components["occurrenceID"].astype(str).eq(occurrence_id)
        & components["component_type"].astype(str).isin(TEXT_BEARING_COMPONENTS)
    ].copy()
    rows["_detector_confidence"] = pd.to_numeric(
        rows.get("detector_confidence", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    rows = rows.sort_values("_detector_confidence", ascending=False)
    paths = []
    for value in rows.get("crop_path", pd.Series(dtype=str)):
        path = clean_str(value)
        if path and Path(path).is_file() and path not in paths:
            paths.append(path)
        if len(paths) >= max(0, limit):
            break
    return paths


def add_whole_sheet_components(
    components: pd.DataFrame,
    eval_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    image_by_occurrence = manifest.set_index("occurrenceID").to_dict(orient="index")
    rows = components.to_dict(orient="records")
    for _, record in eval_df.iterrows():
        occurrence_id = clean_str(record.get("occurrenceID"))
        image_path = clean_str(image_by_occurrence.get(occurrence_id, {}).get("image_path"))
        if not image_path or not Path(image_path).exists():
            continue
        with Image.open(image_path) as image:
            bbox = [0, 0, image.width, image.height]
            rows.append({
                "occurrenceID": occurrence_id,
                "catalogNumber": clean_str(record.get("catalogNumber")),
                "region_id": f"{occurrence_id}::whole_sheet",
                "component_id": f"{occurrence_id}::whole_sheet",
                "component_type": "whole_sheet",
                "detector_model": "deterministic_full_sheet",
                "detector_confidence": 1.0,
                "confidence": 1.0,
                "bbox_xyxy": json.dumps(bbox),
                "bbox_normalized_xyxy": json.dumps(normalise_bbox(bbox, image.width, image.height)),
                "coordinate_space": "full_sheet_pixels",
                "bbox": json.dumps(bbox),
                "source_image_path": image_path,
                "image_path": image_path,
                "crop_path": image_path,
                "selected_for_field_detection": False,
                "annotation_path": "",
                "detection_source": "deterministic",
            })
    return pd.DataFrame(rows)


def packet_subset(packet: dict[str, Any], component_types: set[str]) -> dict[str, Any]:
    return {
        **packet,
        "components": [
            component
            for component in packet.get("components", [])
            if clean_str(component.get("component_type")).lower() in component_types
        ],
    }


def build_reference_manifest(
    demo: pd.DataFrame,
    manifest: pd.DataFrame,
    eval_ids: set[str],
    output_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    refs = demo.merge(
        manifest[["occurrenceID", "image_path"]],
        on="occurrenceID",
        how="left",
    )
    refs = refs[~refs["occurrenceID"].isin(eval_ids)].copy()
    assert_no_rag_leakage(eval_ids, set(refs["occurrenceID"]))
    rows = []
    references = []
    for _, row in refs.iterrows():
        occurrence_id = clean_str(row.get("occurrenceID"))
        text = " | ".join(
            f"{field}: {clean_str(row.get(field))}"
            for field in ["scientificName", "recordedBy", "eventDate", "country", "stateProvince"]
            if clean_str(row.get(field))
        )
        item = {
            "reference_occurrenceID": occurrence_id,
            "institutionCode": clean_str(row.get("institutionCode")),
            "image_path": clean_str(row.get("image_path")),
            "image_url": clean_str(row.get("image_url")),
            "gold_metadata_json": json.dumps(
                {field: clean_str(row.get(field)) for field in EXTRACTION_FIELDS},
                ensure_ascii=False,
            ),
            "gold_transcription": "",
            "full_sheet_visual_embedding": "",
            "component_visual_embeddings": "",
            "text_embedding": "runtime_tfidf",
            "text": text,
            "gold_scientificName": clean_str(row.get("scientificName")),
            "gold_recordedBy": clean_str(row.get("recordedBy")),
            "gold_eventDate": clean_str(row.get("eventDate")),
        }
        rows.append(item)
        references.append(item)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame, references


def evaluate_predictions(
    predictions: pd.DataFrame,
    eval_df: pd.DataFrame,
    components: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold = eval_df.set_index("occurrenceID")
    detail_rows = []
    for _, prediction in predictions.iterrows():
        occurrence_id = clean_str(prediction.get("occurrenceID"))
        if occurrence_id not in gold.index:
            continue
        for field in EXTRACTION_FIELDS:
            gold_value = clean_str(gold.loc[occurrence_id].get(field))
            if not gold_value:
                continue
            predicted = clean_str(prediction.get(field))
            detail_rows.append({
                "occurrenceID": occurrence_id,
                "branch": clean_str(prediction.get("branch")),
                "field": field,
                "prediction": predicted,
                "gold": gold_value,
                "coverage": float(bool(predicted)),
                "exact_match": field_exact_match(field, predicted, gold_value) if predicted else 0,
                "token_f1": field_token_f1(field, predicted, gold_value) if predicted else 0.0,
                "review_required": truthy_flag(prediction.get(f"{field}_review_required", False)),
            })
    detail = pd.DataFrame(detail_rows)
    field_summary = (
        detail.groupby(["branch", "field"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            review_required_rate=("review_required", "mean"),
            n=("occurrenceID", "count"),
        )
        if len(detail)
        else pd.DataFrame()
    )
    component_counts = components.groupby("occurrenceID").size()
    identifier_counts = components[
        components["component_type"].isin(["barcode", "database_label", "number"])
    ].groupby("occurrenceID").size()
    fallback_by_occurrence = diagnostics.set_index("occurrenceID").get(
        "fallback_used",
        pd.Series(False, index=diagnostics["occurrenceID"]),
    )
    branch_rows = []
    for branch in BRANCHES:
        branch_detail = detail[detail["branch"] == branch]
        branch_predictions = predictions[predictions["branch"] == branch]
        row = {
            "branch": branch,
            "records": int(branch_predictions["occurrenceID"].nunique()),
            "coverage": branch_detail["coverage"].mean() if len(branch_detail) else float("nan"),
            "exact_match": branch_detail["exact_match"].mean() if len(branch_detail) else float("nan"),
            "token_f1": branch_detail["token_f1"].mean() if len(branch_detail) else float("nan"),
            "component_count": float(component_counts.reindex(eval_df["occurrenceID"]).fillna(0).mean()),
            "identifier_component_recall_proxy": float(
                identifier_counts.reindex(eval_df["occurrenceID"]).fillna(0).gt(0).mean()
            ),
            "fallback_rate": float(
                fallback_by_occurrence.reindex(eval_df["occurrenceID"]).fillna(False).map(truthy_flag).mean()
            ),
            "review_required_rate": float(
                branch_detail["review_required"].mean() if len(branch_detail) else 0
            ),
            "not_evaluated_rate": float(
                branch_predictions["status"].eq("not_evaluated").mean()
                if len(branch_predictions) else 0
            ),
            "parse_failure_rate": float(
                branch_predictions["status"].eq("parse_failure").mean()
                if len(branch_predictions) else 0
            ),
        }
        for field in ["catalogNumber", "scientificName", "recordedBy", "eventDate"]:
            subset = branch_detail[branch_detail["field"] == field]
            row[f"{field}_exact_match"] = subset["exact_match"].mean() if len(subset) else float("nan")
        branch_rows.append(row)
    return detail, field_summary, pd.DataFrame(branch_rows)


def build_review_queue(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, prediction in predictions.iterrows():
        for field in EXTRACTION_FIELDS:
            if truthy_flag(prediction.get(f"{field}_review_required", False)):
                rows.append({
                    "occurrenceID": prediction["occurrenceID"],
                    "branch": prediction["branch"],
                    "field": field,
                    "value": prediction.get(field, ""),
                    "evidence_source": prediction.get(f"{field}_evidence_source", ""),
                    "alternative_candidates": prediction.get(f"{field}_alternative_candidates", "[]"),
                    "reason": "conflicting_or_unsupported_evidence",
                })
    return pd.DataFrame(rows)


def build_report(
    path: Path,
    eval_df: pd.DataFrame,
    branch_comparison: pd.DataFrame,
    field_comparison: pd.DataFrame,
    manifest: pd.DataFrame,
    retrieval: pd.DataFrame,
    llm_diagnostics: list[dict[str, Any]],
) -> None:
    image_success = int(manifest["image_path"].astype(str).ne("").sum())
    requested_models = sorted({
        clean_str(item.get("requested_model"))
        for item in llm_diagnostics
        if clean_str(item.get("requested_model"))
    })
    actual_models = sorted({
        clean_str(item.get("actual_model"))
        for item in llm_diagnostics
        if clean_str(item.get("actual_model"))
    })
    parsed_calls = sum(item.get("llm_status") == "parsed" for item in llm_diagnostics)
    not_evaluated_calls = sum(
        item.get("llm_status") != "parsed" for item in llm_diagnostics
    )
    lines = [
        "# Component-Aware Herbarium Eval10 Report",
        "",
        f"- Frozen EVAL records: `{len(eval_df)}`",
        f"- Images available: `{image_success}`",
        f"- LLM calls represented: `{len(llm_diagnostics)}`",
        f"- LLM parsed calls: `{parsed_calls}`",
        f"- LLM not evaluated / failed calls: `{not_evaluated_calls}`",
        f"- Requested model names: `{', '.join(requested_models) or 'not available'}`",
        f"- Actual model names returned: `{', '.join(actual_models) or 'not available'}`",
        f"- Retrieval rows: `{len(retrieval)}`",
        "- RAG references are support context only and exclude every EVAL occurrenceID.",
        "- Identifier-component recall is a detection proxy, not ground-truth object-detection recall.",
        "- Model-reported confidence is not a calibrated probability.",
        "",
        "## Four-branch comparison",
        "",
        branch_comparison.to_markdown(index=False),
        "",
        "## Field-level comparison",
        "",
        field_comparison.to_markdown(index=False) if len(field_comparison) else "No evaluable fields.",
        "",
        "## Limitations",
        "",
        "- The public reference corpus is currently the small human-curated DEMO split; expand it before drawing RAG conclusions.",
        "- Visual retrieval is disabled when CLIP model dependencies or downloads are unavailable.",
        "- CER and WER are omitted because this frozen set does not include verified full-label transcriptions.",
        "- Ten records are a diagnostic pilot, not a production performance estimate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_bundle(
    report_dir: Path,
    processed: Path,
    components: pd.DataFrame,
    report_path: Path,
) -> Path:
    bundle = report_dir / "review_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "processed").mkdir(parents=True)
    (bundle / "assets" / "overviews").mkdir(parents=True)
    (bundle / "assets" / "crops").mkdir(parents=True)
    for path in processed.glob("*"):
        if path.is_file() and path.name != "sheet_components.csv":
            shutil.copy2(path, bundle / "processed" / path.name)
    shutil.copy2(report_path, bundle / report_path.name)
    bundle_components = components.copy()
    overview_members: list[str] = []
    crop_members: list[str] = []
    for _, component in bundle_components.iterrows():
        identifier = safe_filename(
            clean_str(component.get("catalogNumber"))
            or clean_str(component.get("occurrenceID"))
        )
        annotation_value = clean_str(component.get("annotation_path"))
        annotation = Path(annotation_value) if annotation_value else None
        if annotation is not None and annotation.is_file():
            overview_name = f"{identifier}_{annotation.name}"
            destination = bundle / "assets" / "overviews" / overview_name
            if not destination.exists():
                shutil.copy2(annotation, destination)
            overview_members.append(
                f"component_aware_eval10/assets/overviews/{overview_name}"
            )
        else:
            overview_members.append("")

        crop_value = clean_str(component.get("crop_path"))
        crop = Path(crop_value) if crop_value else None
        if crop is not None and crop.is_file():
            region = safe_filename(clean_str(component.get("region_id")) or crop.stem)
            crop_name = f"{identifier}_{region}{crop.suffix.lower() or '.jpg'}"
            destination = bundle / "assets" / "crops" / crop_name
            if not destination.exists():
                shutil.copy2(crop, destination)
            crop_members.append(f"component_aware_eval10/assets/crops/{crop_name}")
        else:
            crop_members.append("")
    bundle_components["review_overview_member"] = overview_members
    bundle_components["review_crop_member"] = crop_members
    bundle_components.to_csv(bundle / "processed" / "sheet_components.csv", index=False)
    archive = report_dir / "review_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source in bundle.rglob("*"):
            if source.is_file():
                zf.write(source, Path("component_aware_eval10") / source.relative_to(bundle))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/component_aware_eval10.yaml")
    args = parser.parse_args()
    cfg, paths = load_runtime(args.config)
    processed = paths["processed"]

    print("[component-aware] preparing frozen metadata and images", flush=True)
    demo, eval_df, _ = stage_metadata(args.config)
    manifest = stage_download(args.config)
    print("[component-aware] running full-sheet component detection", flush=True)
    stage_layout(args.config)
    raw_components = pd.read_csv(
        processed / "hespi_sheet_components.csv",
        dtype=str,
    ).fillna("")
    eval_ids = set(eval_df["occurrenceID"])
    components = raw_components[raw_components["occurrenceID"].isin(eval_ids)].copy()
    components = add_whole_sheet_components(components, eval_df, manifest)
    components.to_csv(processed / "sheet_components.csv", index=False)

    readings = read_sheet_components(components, cfg)
    readings.to_csv(processed / "component_readings.csv", index=False)
    packets = build_evidence_packets(components, readings)
    write_jsonl(processed / "evidence_packets.jsonl", packets)
    print(
        f"[component-aware] detected {len(components)} components and saved "
        f"{len(readings)} readings",
        flush=True,
    )

    reference_manifest, references = build_reference_manifest(
        demo,
        manifest,
        eval_ids,
        processed / "rag_reference_manifest.csv",
    )
    visual_enabled = bool(cfg.get("rag", {}).get("visual_enabled", True))
    visual_model = clean_str(
        cfg.get("rag", {}).get("visual_model", "openai/clip-vit-base-patch32")
    ) or "openai/clip-vit-base-patch32"
    reference_sheet_embeddings, visual_status = clip_image_embeddings(
        reference_manifest.get("image_path", pd.Series(dtype=str)).astype(str).tolist(),
        model_name=visual_model,
        enabled=visual_enabled and len(reference_manifest) > 0,
    )
    max_component_images = int(cfg.get("rag", {}).get("max_component_images", 8))
    reference_component_embeddings: list[list[np.ndarray]] = []
    component_visual_statuses: list[str] = []
    for occurrence_id in reference_manifest.get(
        "reference_occurrenceID",
        pd.Series(dtype=str),
    ).astype(str):
        crop_paths = component_crop_paths(
            raw_components,
            occurrence_id,
            limit=max_component_images,
        )
        vectors, status = clip_image_embeddings(
            crop_paths,
            model_name=visual_model,
            enabled=visual_enabled and bool(crop_paths),
        )
        reference_component_embeddings.append(
            [] if vectors is None else [vector for vector in vectors]
        )
        component_visual_statuses.append(
            "no_component_crops" if not crop_paths else status
        )
    reference_embeddings = None
    if len(reference_manifest):
        aggregated = []
        for index in range(len(reference_manifest)):
            vectors = []
            if reference_sheet_embeddings is not None:
                vectors.append(reference_sheet_embeddings[index])
            vectors.extend(reference_component_embeddings[index])
            aggregated.append(normalized_mean_embedding(vectors))
        if aggregated and all(vector is not None for vector in aggregated):
            reference_embeddings = np.stack(aggregated).astype(np.float32)
    text_embeddings = tfidf_embeddings(
        reference_manifest.get("text", pd.Series(dtype=str)).astype(str).tolist()
    )
    if len(reference_manifest):
        if reference_sheet_embeddings is not None:
            reference_manifest["full_sheet_visual_embedding"] = [
                json.dumps(vector.astype(float).tolist())
                for vector in reference_sheet_embeddings
            ]
        else:
            reference_manifest["full_sheet_visual_embedding"] = ""
        reference_manifest["component_visual_embeddings"] = [
            json.dumps(
                [vector.astype(float).tolist() for vector in vectors],
            )
            for vectors in reference_component_embeddings
        ]
        reference_manifest["component_visual_embedding_status"] = (
            component_visual_statuses
        )
        reference_manifest["text_embedding"] = [
            json.dumps(vector.astype(float).tolist()) for vector in text_embeddings
        ]
        reference_manifest.to_csv(processed / "rag_reference_manifest.csv", index=False)

    packet_by_occurrence = {packet["occurrenceID"]: packet for packet in packets}
    image_by_occurrence = manifest.set_index("occurrenceID").to_dict(orient="index")
    retrieval_rows = []
    predictions = []
    reconciled_jsonl = []
    llm_diagnostics: list[dict[str, Any]] = []
    for record_index, (_, gold_row) in enumerate(eval_df.iterrows(), start=1):
        occurrence_id = clean_str(gold_row.get("occurrenceID"))
        print(
            f"[component-aware] reconciling record {record_index}/{len(eval_df)}: "
            f"{occurrence_id}",
            flush=True,
        )
        packet = packet_by_occurrence.get(occurrence_id, {
            "occurrenceID": occurrence_id,
            "catalogNumber_gold": "",
            "gold_withheld_until_evaluation": True,
            "components": [],
        })
        direct_packet = direct_evidence_packet(packet)
        whole_packet = packet_subset(packet, {"whole_sheet"})
        primary_packet = packet_subset(packet, {"primary_specimen_label"})
        baseline_packets = {
            "baseline_full_sheet": whole_packet,
            "baseline_primary_label": primary_packet,
        }
        for branch, branch_packet in baseline_packets.items():
            record = deterministic_reconciliation(branch_packet)
            predictions.append(flatten_reconciled_record(
                occurrence_id,
                branch,
                record,
                status="deterministic_ocr",
            ))
            reconciled_jsonl.append({
                "occurrenceID": occurrence_id,
                "branch": branch,
                "record": record,
                "status": "deterministic_ocr",
            })

        image_path = clean_str(image_by_occurrence.get(occurrence_id, {}).get("image_path"))
        query_embedding = None
        if image_path:
            query_paths = [image_path, *component_crop_paths(
                components,
                occurrence_id,
                limit=max_component_images,
            )]
            query_vectors, _ = clip_image_embeddings(
                query_paths,
                model_name=visual_model,
                enabled=visual_enabled,
            )
            if query_vectors is not None:
                query_embedding = normalized_mean_embedding(
                    [vector for vector in query_vectors]
                )
        retrieved = retrieve_hybrid_references(
            query_text=evidence_packet_text(direct_packet),
            query_visual_embedding=query_embedding,
            query_institution_code=clean_str(gold_row.get("institutionCode")),
            references=references,
            reference_visual_embeddings=reference_embeddings,
            top_k=int(cfg.get("rag", {}).get("top_k", 3)),
        )
        for item in retrieved:
            retrieval_rows.append({
                "occurrenceID": occurrence_id,
                **{key: value for key, value in item.items() if key != "text"},
                "visual_index_status": visual_status,
            })

        for branch, context in [
            ("component_aware_no_rag", []),
            ("component_aware_with_rag", retrieved),
        ]:
            if cfg.get("llm", {}).get("backend") == "none":
                record = deterministic_reconciliation(direct_packet)
                meta = {
                    "llm_status": "not_evaluated",
                    "not_evaluated_reason": "missing_api_credentials_or_disabled",
                    "raw_output": "",
                }
            else:
                record, meta = reconcile_with_optional_llm(direct_packet, cfg, context)
            predictions.append(flatten_reconciled_record(
                occurrence_id,
                branch,
                record,
                status=clean_str(meta.get("llm_status")),
            ))
            reconciled_jsonl.append({
                "occurrenceID": occurrence_id,
                "branch": branch,
                "record": record,
                "status": clean_str(meta.get("llm_status")),
                "not_evaluated_reason": clean_str(meta.get("not_evaluated_reason")),
                "backend": clean_str(
                    meta.get("backend", cfg.get("llm", {}).get("backend"))
                ),
                "requested_model": clean_str(meta.get("requested_model")),
                "actual_model": clean_str(meta.get("actual_model")),
                "retrieved_reference_occurrenceIDs": [
                    clean_str(item.get("reference_occurrenceID"))
                    for item in context
                ],
                "raw_output": clean_str(meta.get("raw_output")),
                "finish_reason": clean_str(meta.get("finish_reason")),
                "usage": meta.get("usage", {}),
                "error_message": clean_str(meta.get("error_message")),
            })
            llm_diagnostics.append({
                "occurrenceID": occurrence_id,
                "branch": branch,
                "backend": clean_str(meta.get("backend", cfg.get("llm", {}).get("backend"))),
                "requested_model": clean_str(meta.get("requested_model")),
                "actual_model": clean_str(meta.get("actual_model")),
                "llm_status": clean_str(meta.get("llm_status")),
                "not_evaluated_reason": clean_str(meta.get("not_evaluated_reason")),
                "raw_output_length": len(clean_str(meta.get("raw_output"))),
                "finish_reason": clean_str(meta.get("finish_reason")),
                "endpoint_reachable": meta.get("endpoint_reachable", False),
                "api_key_present": meta.get("api_key_present", False),
                "error_message": clean_str(meta.get("error_message")),
                "visual_index_status": visual_status,
            })
        write_runtime_checkpoints(
            processed,
            retrieval_rows,
            predictions,
            reconciled_jsonl,
            llm_diagnostics,
        )

    retrieval = pd.DataFrame(retrieval_rows)
    predictions_frame = pd.DataFrame(predictions)
    write_runtime_checkpoints(
        processed,
        retrieval_rows,
        predictions,
        reconciled_jsonl,
        llm_diagnostics,
    )

    diagnostics = pd.read_csv(processed / "hespi_layout_diagnostics.csv", dtype=str).fillna("")
    detail, field_comparison, branch_comparison = evaluate_predictions(
        predictions_frame,
        eval_df,
        components,
        diagnostics[diagnostics["occurrenceID"].isin(eval_ids)],
    )
    detail.to_csv(processed / "evaluation_detail.csv", index=False)
    field_comparison.to_csv(processed / "field_comparison.csv", index=False)
    branch_comparison.to_csv(processed / "branch_comparison.csv", index=False)
    build_review_queue(predictions_frame).to_csv(processed / "review_queue.csv", index=False)

    report_path = paths["reports"] / "component_aware_eval10_report.md"
    build_report(
        report_path,
        eval_df,
        branch_comparison,
        field_comparison,
        manifest[manifest["occurrenceID"].isin(eval_ids)],
        retrieval,
        llm_diagnostics,
    )
    archive = build_review_bundle(paths["reports"], processed, components, report_path)
    print(branch_comparison.to_string(index=False))
    print(archive)


if __name__ == "__main__":
    main()
