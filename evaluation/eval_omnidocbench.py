"""Project-parser bridge for offline OmniDocBench page evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_settings
from evaluation.eval_qasper import result_metadata
from evaluation.pdf_benchmark import (
    evidence_coverage,
    normalize_text,
    ordered_similarity,
    save_json,
    token_f1,
    token_recall,
)
from retrieval.pdf_parse import provider_from_config


def _component_text(item: dict[str, Any]) -> str:
    category = str(item.get("category_type") or "")
    keys = (
        ("html", "latex", "text")
        if category == "table"
        else ("latex", "text")
        if "equation" in category or "formula" in category
        else ("text", "latex", "html")
    )
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ordered_components(sample: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        item
        for item in sample.get("layout_dets", [])
        if isinstance(item, dict) and _component_text(item)
    ]
    def order_value(item: dict[str, Any]) -> int | float | None:
        value = item.get("order")
        return value if isinstance(value, (int, float)) else item.get("reading_order")

    if items and all(isinstance(order_value(item), (int, float)) for item in items):
        return sorted(
            items,
            key=lambda item: float(order_value(item) or 0),
        )
    return items


def _resolve_image(images_root: Path, image_path: str, by_name: dict[str, Path]) -> Path:
    relative = images_root / Path(image_path)
    if relative.is_file():
        return relative
    name = Path(image_path).name.casefold()
    if name in by_name:
        return by_name[name]
    raise FileNotFoundError(f"找不到 OmniDocBench 页面图像：{image_path}")


def _image_to_pdf(image: Path, output: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("OmniDocBench 页面适配需要 PyMuPDF") from exc
    pixmap = fitz.Pixmap(str(image))
    document = fitz.open()
    page = document.new_page(width=pixmap.width, height=pixmap.height)
    page.insert_image(page.rect, filename=str(image))
    document.save(str(output))
    document.close()


def _prediction_path(predictions: Path, image_path: str) -> Path | None:
    relative = Path(image_path).with_suffix(".md")
    candidates = [
        predictions / relative,
        predictions / relative.name,
        predictions / f"{Path(image_path).stem}.txt",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_omnidocbench(
    annotations: Path,
    images_root: Path,
    *,
    predictions_dir: Path | None,
    export_predictions: Path | None,
    limit: int,
    language: str | None,
    data_source: str | None,
    force_ocr: bool,
) -> dict[str, Any]:
    """Evaluate project/externally produced page Markdown against Omni annotations."""

    settings = load_settings()
    samples = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("OmniDocBench annotations 必须是 JSON list")
    filtered = []
    language_aliases = {
        "en": "english",
        "chinese": "simplified_chinese",
        "zh": "simplified_chinese",
    }
    requested_language = language_aliases.get(language or "", language)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        attributes = sample.get("page_info", {}).get("page_attribute", {})
        if requested_language and attributes.get("language") != requested_language:
            continue
        if data_source and attributes.get("data_source") != data_source:
            continue
        filtered.append(sample)
        if limit and len(filtered) >= limit:
            break

    if not filtered:
        available_languages = sorted(
            {
                str(item.get("page_info", {}).get("page_attribute", {}).get("language"))
                for item in samples
                if isinstance(item, dict)
            }
        )
        available_sources = sorted(
            {
                str(item.get("page_info", {}).get("page_attribute", {}).get("data_source"))
                for item in samples
                if isinstance(item, dict)
            }
        )
        raise ValueError(
            "OmniDocBench 过滤后没有样本；"
            f"language={language!r}, data_source={data_source!r}, "
            f"available_languages={available_languages}, "
            f"available_sources={available_sources}"
        )

    if export_predictions:
        export_predictions.mkdir(parents=True, exist_ok=True)
    image_index = {
        path.name.casefold(): path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for path in images_root.rglob(pattern)
    }
    provider = provider_from_config(
        "tesseract" if force_ocr else settings.pdf_parse.provider,
        auto_ocr=force_ocr or settings.pdf_parse.auto_ocr,
        timeout_seconds=settings.pdf_parse.timeout_seconds,
    )
    page_rows: list[dict[str, Any]] = []
    missing = 0
    empty = 0
    with TemporaryDirectory() as directory:
        temp_root = Path(directory)
        for index, sample in enumerate(filtered):
            page_info = sample.get("page_info", {})
            image_path = str(page_info.get("image_path") or "")
            name = Path(image_path).stem or f"page-{index}"
            prediction_file = (
                _prediction_path(predictions_dir, image_path)
                if predictions_dir
                else None
            )
            error: str | None
            try:
                if prediction_file is not None:
                    prediction = prediction_file.read_text(encoding="utf-8")
                elif predictions_dir is not None:
                    raise FileNotFoundError(f"缺少 prediction：{image_path}")
                else:
                    image = _resolve_image(images_root, image_path, image_index)
                    pdf = temp_root / f"{index}.pdf"
                    _image_to_pdf(image, pdf)
                    parsed = provider.parse(pdf)
                    prediction = parsed.pages[0].indexed_text if parsed.pages else ""
            except (OSError, RuntimeError, FileNotFoundError) as exc:
                prediction = ""
                missing += 1
                error = str(exc)
            else:
                error = None
            if not normalize_text(prediction):
                empty += 1
            if export_predictions:
                (export_predictions / f"{name}.md").write_text(
                    prediction,
                    encoding="utf-8",
                )

            components = _ordered_components(sample)
            gold = "\n\n".join(_component_text(item) for item in components)
            text_gold = "\n\n".join(
                _component_text(item)
                for item in components
                if item.get("text")
                and item.get("category_type") not in {"table", "equation_isolated"}
            )
            tables = [
                _component_text(item)
                for item in components
                if item.get("category_type") == "table"
            ]
            formulas = [
                _component_text(item)
                for item in components
                if "equation" in str(item.get("category_type") or "")
            ]
            page_rows.append(
                {
                    "image": image_path,
                    "text_token_f1": token_f1(text_gold, prediction),
                    "text_token_recall": token_recall(text_gold, prediction),
                    "ordered_token_similarity": ordered_similarity(gold, prediction),
                    "table_content_coverage": _mean(
                        [evidence_coverage(item, prediction) for item in tables]
                    ),
                    "formula_content_coverage": _mean(
                        [evidence_coverage(item, prediction) for item in formulas]
                    ),
                    "table_count": len(tables),
                    "formula_count": len(formulas),
                    "error": error,
                }
            )

    table_rows = [row for row in page_rows if row["table_count"]]
    formula_rows = [row for row in page_rows if row["formula_count"]]
    return {
        "dataset": "OmniDocBench",
        "metadata": result_metadata(settings, annotations),
        "filters": {"language": language, "data_source": data_source},
        "parser": provider.name,
        "force_ocr": force_ocr,
        "pages": len(page_rows),
        "missing_predictions": missing,
        "empty_prediction_rate": empty / len(page_rows) if page_rows else 0.0,
        "metrics": {
            "text_token_f1": _mean([row["text_token_f1"] for row in page_rows]),
            "text_token_recall": _mean(
                [row["text_token_recall"] for row in page_rows]
            ),
            "ordered_token_similarity": _mean(
                [row["ordered_token_similarity"] for row in page_rows]
            ),
            "table_content_coverage": _mean(
                [row["table_content_coverage"] for row in table_rows]
            ),
            "formula_content_coverage": _mean(
                [row["formula_content_coverage"] for row in formula_rows]
            ),
        },
        "metric_scope": (
            "项目桥接指标，不等同于 OmniDocBench 官方 TEDS/CDM/COCODet；"
            "使用 --export-predictions 后可继续运行官方 evaluator"
        ),
        "per_page": page_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniDocBench 项目 parser 离线评测")
    parser.add_argument(
        "annotations",
        type=Path,
        help="OmniDocBench JSON annotation 文件",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="包含 page images 的目录",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="可选：已有逐页 .md/.txt；省略时运行项目 parser",
    )
    parser.add_argument("--export-predictions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--language", default=None)
    parser.add_argument("--data-source", default=None)
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="页面图像转单页 PDF 后强制使用本地 Tesseract；不调用 LLM API",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/omnidocbench_parse.json"),
    )
    args = parser.parse_args()
    report = evaluate_omnidocbench(
        args.annotations,
        args.images_root,
        predictions_dir=args.predictions_dir,
        export_predictions=args.export_predictions,
        limit=args.limit,
        language=args.language,
        data_source=args.data_source,
        force_ocr=args.ocr,
    )
    save_json(report, args.output)
    print(f"saved OmniDocBench report to {args.output}")


if __name__ == "__main__":
    main()
