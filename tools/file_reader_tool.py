from __future__ import annotations

from pathlib import Path


class FileReaderTool:
    def run(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        file = Path(path)
        text = file.read_text(encoding="utf-8")
        if start_line is None and end_line is None:
            return text

        lines = text.splitlines()
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        return "\n".join(lines[s:e])
