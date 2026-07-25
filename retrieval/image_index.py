from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from retrieval.pdf_service import PDFService


def image_signature(data: bytes, bins: int = 64) -> np.ndarray:
    """Deterministic pixel histogram, independent from PNG/JPEG byte encoding."""

    if not data:
        return np.zeros(bins, dtype=np.float32)
    try:
        import fitz

        pixmap = fitz.Pixmap(data)
        samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    except Exception:
        samples = np.frombuffer(data, dtype=np.uint8)
    if not samples.size:
        return np.zeros(bins, dtype=np.float32)
    histogram, _ = np.histogram(samples, bins=bins, range=(0, 256))
    vector = histogram.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


class PageImageIndex:
    """Precomputed bounded page-image signatures."""

    def __init__(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        self.vectors_path = root / "image_vectors.npy"
        self.metadata_path = root / "image_metadata.json"
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []

    def build(
        self,
        pdfs: list[Path],
        *,
        max_pages: int,
        max_side: int,
    ) -> int:
        service = PDFService()
        vectors: list[np.ndarray] = []
        metadata: list[dict] = []
        for pdf in sorted(pdfs):
            for page_number in range(1, service.page_count(pdf) + 1):
                if len(metadata) >= max_pages:
                    break
                image = service.render_page(pdf, page_number, max_side=max_side)
                vectors.append(image_signature(image.data))
                metadata.append(
                    {
                        "paper_id": pdf.stem,
                        "source": str(pdf),
                        "page_number": page_number,
                        "mime_type": image.mime_type,
                    }
                )
            if len(metadata) >= max_pages:
                break
        values = np.vstack(vectors) if vectors else np.empty((0, 64), dtype=np.float32)
        vector_tmp = self.vectors_path.with_suffix(".npy.tmp")
        metadata_tmp = self.metadata_path.with_suffix(".json.tmp")
        try:
            with vector_tmp.open("wb") as handle:
                np.save(handle, values)
                handle.flush()
                os.fsync(handle.fileno())
            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(vector_tmp, self.vectors_path)
            os.replace(metadata_tmp, self.metadata_path)
        except Exception:
            vector_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
            raise
        self.vectors = values
        self.metadata = metadata
        return len(metadata)

    def load(self) -> None:
        if not self.vectors_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Page image index is missing; rebuild the paper index")
        self.vectors = np.load(self.vectors_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if len(self.vectors) != len(self.metadata):
            raise ValueError("Page image index metadata/vector length mismatch")

    def search(self, query: bytes, top_k: int) -> list[tuple[dict, float]]:
        if self.vectors is None:
            self.load()
        if not len(self.metadata):
            return []
        scores = self.vectors @ image_signature(query)
        indices = np.argsort(-scores)[:top_k]
        return [(self.metadata[int(index)], float(scores[index])) for index in indices]
