from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from retrieval.analyzer import QueryAnalyzer
from retrieval.chunker import Chunk, PaperChunker
from retrieval.models import PaperDocument, PaperPage, PaperSection
from retrieval.vector_store import IndexCompatibilityError, VectorStore


class RetrievalTests(unittest.TestCase):
    def test_cjk_analyzer(self) -> None:
        tokens = QueryAnalyzer(2).tokens("检索增强 generation RAG")
        self.assertIn("检索", tokens)
        self.assertIn("索增", tokens)
        self.assertIn("generation", tokens)
        self.assertIn("rag", tokens)

    def test_chunk_builder_deduplicates_and_page_bbox_is_none(self) -> None:
        document = PaperDocument(
            paper_id="p",
            title="Paper",
            source="p.pdf",
            sections=[PaperSection("Body", "same content")],
            pages=[
                PaperPage(
                    page_number=1,
                    text="same content",
                    ocr_text="same content",
                    vlm_summary="visual summary",
                )
            ],
        )
        chunks = PaperChunker(100, 10).build([document])
        self.assertEqual(
            len([chunk for chunk in chunks if chunk.content == "same content"]),
            1,
        )
        visual = next(chunk for chunk in chunks if chunk.content == "visual summary")
        self.assertIsNone(visual.bbox)
        self.assertEqual(visual.granularity, "page")
        self.assertTrue(visual.parent_id)
        self.assertTrue(visual.content_hash)

    def test_vector_snapshot_fingerprint_and_hot_reload(self) -> None:
        with TemporaryDirectory() as directory:
            store = VectorStore(directory)
            chunks = [
                Chunk("1", "p", "P", "Body", "alpha", "p.pdf"),
                Chunk("2", "p", "P", "Body", "beta", "p.pdf"),
            ]
            signature = {"fingerprint": "same"}
            store.build(
                chunks,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                embedding_signature=signature,
            )
            reader = VectorStore(directory)
            reader.load(signature)
            first_generation = reader.generation
            with self.assertRaises(IndexCompatibilityError):
                VectorStore(directory).load({"fingerprint": "different"})
            store.build(
                chunks,
                np.asarray([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32),
                embedding_signature=signature,
            )
            self.assertTrue(reader.reload_if_changed(signature))
            self.assertNotEqual(first_generation, reader.generation)
            self.assertEqual(
                len(list(Path(directory).glob("vectors-*.npy"))),
                1,
            )

    def test_vector_filtering(self) -> None:
        with TemporaryDirectory() as directory:
            chunks = [
                Chunk("1", "a", "A", "Body", "alpha", "a.pdf"),
                Chunk("2", "b", "B", "Body", "beta", "b.pdf"),
            ]
            store = VectorStore(directory)
            store.build(
                chunks,
                np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
                embedding_signature={"fingerprint": "x"},
            )
            result = store.search(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                top_k=2,
                allowed_indices=[1],
            )
            self.assertEqual([item[0].paper_id for item in result], ["b"])


if __name__ == "__main__":
    unittest.main()
