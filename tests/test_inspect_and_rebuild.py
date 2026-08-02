from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from config.settings import load_settings
from indexing.build_index import build_index
from retrieval.index import Embedder
from retrieval.mineru import MinerUAdapter, MinerUError, atomic_json
from tools.inspect_paper_tool import InspectPaperTool


class _FailingProvider:
    name = "mineru"

    def parse(self, file: Path) -> object:
        del file
        raise MinerUError("service unavailable")


class _FakeEmbedder(Embedder):
    signature = {"fingerprint": "fake", "backend": "fake"}

    def __init__(self) -> None:
        pass

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class InspectAndRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.papers = self.root / "papers"
        self.indexes = self.root / "indexes"
        self.papers.mkdir()
        self.indexes.mkdir()
        self.pdf = self.papers / "demo.pdf"
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "Demo")
            document.save(self.pdf)
        base = load_settings()
        self.settings = replace(
            base,
            project=replace(base.project, data_root=str(self.papers)),
            index=replace(base.index, index_root=str(self.indexes)),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_inspect_reads_canonical_section_and_region(self) -> None:
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "demo_content_list.json").write_text(
            json.dumps(
                [
                    {
                        "type": "text",
                        "text": "Methods",
                        "text_level": 1,
                        "page_idx": 0,
                        "bbox": [10, 10, 900, 80],
                    },
                    {
                        "type": "text",
                        "text": "Bounded method evidence.",
                        "page_idx": 0,
                        "bbox": [10, 100, 900, 300],
                    },
                ]
            ),
            encoding="utf-8",
        )
        adapter = MinerUAdapter(self.settings.mineru)
        payload = adapter.canonicalize(
            self.pdf,
            raw,
            {"status": "completed", "version": "3.4.4"},
        )
        atomic_json(self.pdf.with_suffix(".mineru.json"), payload)
        tool = InspectPaperTool(self.settings)

        section = tool.run("demo", {"kind": "section", "section": "Methods"})
        region = tool.run(
            "demo", {"kind": "region", "page_number": 1, "bbox": [0, 90, 1000, 400]}
        )
        self.assertIn("Bounded method evidence", section.text)
        self.assertIn("Bounded method evidence", region.text)
        self.assertEqual(region.sources[0].bbox_space, "normalized_1000")

    def test_failed_full_rebuild_does_not_publish_manifest(self) -> None:
        manifest = self.indexes / "manifest.json"
        manifest.write_text('{"generation":"old"}', encoding="utf-8")
        before = manifest.read_bytes()
        with patch(
            "indexing.build_index.provider_from_config",
            return_value=_FailingProvider(),
        ):
            with self.assertRaises(RuntimeError):
                build_index(
                    self.settings,
                    incremental=False,
                    embedder=_FakeEmbedder(),
                )
        self.assertEqual(manifest.read_bytes(), before)

    def test_embedder_lazy_load_is_safe_under_parallel_tool_calls(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                del model_name
                started.set()
                release.wait(timeout=1)

            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 384

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
        embedder = Embedder("fake-model")
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(lambda: embedder.backend)
                self.assertTrue(started.wait(timeout=1))
                second = executor.submit(lambda: embedder.backend)
                time.sleep(0.02)
                release.set()
                self.assertEqual(
                    [first.result(), second.result()],
                    ["sentence_transformers", "sentence_transformers"],
                )


if __name__ == "__main__":
    unittest.main()
