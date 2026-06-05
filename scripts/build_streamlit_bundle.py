from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


KEEP_PREFIXES = (
    "data/processed/real_eval_100_",
    "data/interim/llm/deepseek_v4_pro_eval100_",
)
KEEP_FILES = {
    "reports/real_eval_100_deepseek_v4_pro_report.md",
}
DROP_JSONL_KEYS = {
    "reasoning_content",
    "response_body",
}


def should_keep(name: str) -> bool:
    return name in KEEP_FILES or any(name.startswith(prefix) for prefix in KEEP_PREFIXES)


def sanitise_jsonl(data: bytes) -> bytes:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for key in DROP_JSONL_KEYS:
            item.pop(key, None)
        rows.append(json.dumps(item, ensure_ascii=False))
    return ("\n".join(rows) + "\n").encode("utf-8")


def build_bundle(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.is_dir() or not should_keep(info.filename):
                continue
            data = src.read(info.filename)
            if info.filename.endswith("_outputs.jsonl"):
                data = sanitise_jsonl(data)
            dst.writestr(info.filename, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small Streamlit demo bundle from an eval100 artifact zip.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_bundle(args.source, args.output)


if __name__ == "__main__":
    main()
