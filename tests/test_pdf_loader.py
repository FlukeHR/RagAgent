from __future__ import annotations

import json
from pathlib import Path

import dataclasses

from indexing.build_index import build_collection
from retrieval.retriever import Retriever
from tools.pdf_page_tool import PDFPageTool
from tools.pdf_region_tool import PDFRegionTool
from tools.image_search_tool import ImageSearchTool
from retrieval.chunker import PaperChunker
from retrieval.loader import PaperLoader
from retrieval.pdf_parse import ParsedPDF, ParsedPage, PDFParseProvider


def _write_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "A Study of Page Aware Retrieval")
    page1.insert_text((72, 120), "Abstract")
    page1.insert_text((72, 150), "This abstract is on page one.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "1 Introduction")
    page2.insert_text((72, 120), "This introduction is on page two.")
    doc.save(path)
    doc.close()


def test_pdf_sections_and_chunks_keep_page_ranges(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf)

    doc = PaperLoader(str(tmp_path)).load_file(pdf)
    assert doc is not None
    assert doc.pages is not None
    assert [p.page_number for p in doc.pages] == [1, 2]

    intro = next(section for section in doc.sections if "Introduction" in section.title)
    assert intro.page_start == 2
    assert intro.page_end == 2

    chunks = PaperChunker(chunk_size=120, chunk_overlap=20).split([doc])
    intro_chunk = next(chunk for chunk in chunks if "introduction" in chunk.content.lower())
    assert intro_chunk.page_start == 2
    assert intro_chunk.page_end == 2
    assert intro_chunk.element_type == "text"

    page_chunks = PaperChunker(chunk_size=120, chunk_overlap=20).split_pages([doc])
    assert [chunk.element_type for chunk in page_chunks] == ["page", "page"]
    assert [chunk.page_start for chunk in page_chunks] == [1, 2]


def test_scanned_like_page_detection_and_render(tmp_path):
    import fitz

    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    pages = PaperLoader._read_pdf_pages(pdf)
    assert len(pages) == 1
    assert pages[0].is_scanned_like is True

    image = PaperLoader.render_pdf_page(pdf, page_number=1, max_side=400)
    assert image.page_number == 1
    assert image.mime_type in {"image/jpeg", "image/png"}
    assert image.data
    assert max(image.width, image.height) <= 400


def test_read_pdf_page_tool_returns_page_metadata_and_optional_image(tmp_path, settings):
    collection_dir = tmp_path / "demo"
    collection_dir.mkdir()
    pdf = collection_dir / "paper.pdf"
    _write_pdf(pdf)

    local_settings = dataclasses.replace(
        settings,
        project=dataclasses.replace(settings.project, data_root=str(tmp_path)),
    )
    tool = PDFPageTool(local_settings, "demo")

    out = tool.run("paper", 2, include_image=True, max_side=300, max_image_base64_chars=64)
    assert "page=2/2" in out.text
    assert "image_base64" in out.text
    assert out.sources[0]["page_start"] == 2
    assert out.sources[0]["page_end"] == 2
    assert out.sources[0]["element_type"] == "page_image"
    assert out.sources[0]["image_base64"]


def test_scanned_pdf_ocr_sidecar_is_indexed_and_retrievable(tmp_path, settings):
    import fitz

    data_root = tmp_path / "papers"
    index_root = tmp_path / "indexes"
    collection_dir = data_root / "scan"
    collection_dir.mkdir(parents=True)

    pdf = collection_dir / "scanpaper.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    pdf.with_suffix(".ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": "Scanned-only finding: quartz catalyst improves retrieval.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    local_settings = dataclasses.replace(
        settings,
        project=dataclasses.replace(settings.project, data_root=str(data_root)),
        index=dataclasses.replace(settings.index, index_root=str(index_root), top_k_recall=5, top_n_rerank=3),
        embedding=dataclasses.replace(settings.embedding, use_sentence_transformers=False),
        rerank=dataclasses.replace(settings.rerank, use_cross_encoder=False),
    )

    build_collection(local_settings, "scan", incremental=False)
    retriever = Retriever(local_settings, str(index_root / "scan"))
    results = retriever.search("quartz catalyst retrieval")

    assert results
    ocr_hits = [r.chunk for r in results if r.chunk.modality == "ocr" and r.chunk.element_type == "ocr"]
    assert ocr_hits
    assert ocr_hits[0].page_start == 1


def test_pdf_element_sidecars_become_contextual_chunks(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf)
    pdf.with_suffix(".tables.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "id": "t1",
                        "page": 2,
                        "bbox": [70, 100, 250, 180],
                        "caption": "Table 1: Retrieval accuracy",
                        "cells": [
                            {"row": 0, "col": 0, "text": "Dataset"},
                            {"row": 0, "col": 1, "text": "Accuracy"},
                            {"row": 1, "col": 0, "text": "QASPER"},
                            {"row": 1, "col": 1, "text": "91.2%"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    doc = PaperLoader(str(tmp_path)).load_file(pdf)
    assert doc is not None
    chunks = PaperChunker(chunk_size=120, chunk_overlap=20).split_elements([doc])

    table = chunks[0]
    assert table.element_type == "table"
    assert table.bbox == (70.0, 100.0, 250.0, 180.0)
    assert table.page_start == 2
    assert "QASPER" in table.content
    assert table.chunk_context


class MockOCRProvider(PDFParseProvider):
    name = "mock"

    def parse(self, file: Path) -> ParsedPDF:
        return ParsedPDF(
            pages=[
                ParsedPage(
                    page_number=1,
                    text="",
                    is_scanned_like=True,
                    ocr_text="Mock OCR runtime found emerald retrieval.",
                )
            ],
            trace=[{"provider": "mock", "generated": True}],
        )


def test_mock_pdf_parse_provider_generates_ocr_chunks(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf)
    doc = PaperLoader(str(tmp_path), pdf_provider=MockOCRProvider()).load_file(pdf)
    assert doc is not None
    assert doc.pages is not None
    assert doc.pages[0].ocr_text == "Mock OCR runtime found emerald retrieval."

    page_chunks = PaperChunker(chunk_size=120, chunk_overlap=20).split_pages([doc])
    assert any(c.modality == "ocr" and "emerald retrieval" in c.content for c in page_chunks)


def test_read_pdf_region_tool_returns_bbox_and_image(tmp_path, settings):
    collection_dir = tmp_path / "demo"
    collection_dir.mkdir()
    pdf = collection_dir / "paper.pdf"
    _write_pdf(pdf)
    local_settings = dataclasses.replace(
        settings,
        project=dataclasses.replace(settings.project, data_root=str(tmp_path)),
    )

    out = PDFRegionTool(local_settings, "demo").run(
        "paper",
        2,
        [60, 60, 260, 160],
        include_image=True,
        max_side=300,
        max_image_base64_chars=32,
    )

    assert "page=2" in out.text
    assert out.sources[0]["bbox"] == (60.0, 60.0, 260.0, 160.0)
    assert out.sources[0]["image_base64"]


def test_image_search_tool_matches_query_page(tmp_path, settings):
    collection_dir = tmp_path / "demo"
    collection_dir.mkdir()
    pdf = collection_dir / "paper.pdf"
    _write_pdf(pdf)
    local_settings = dataclasses.replace(
        settings,
        project=dataclasses.replace(settings.project, data_root=str(tmp_path)),
        image_search=dataclasses.replace(settings.image_search, max_pages=4, max_side=180),
    )

    out = ImageSearchTool(local_settings, "demo").run(paper_id="paper", page_number=2, top_k=1)

    assert out.sources
    assert out.sources[0]["paper_id"] == "paper"
    assert out.sources[0]["page_start"] == 2
    assert out.sources[0]["element_type"] == "page_image"
