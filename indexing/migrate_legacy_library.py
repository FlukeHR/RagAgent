from __future__ import annotations

import argparse
import shutil

import fitz

from config.settings import BASE_DIR, load_settings
from indexing.build_index import build_index
from services.app_store import AppStore
from services.user_scope import scoped_settings, user_paths


def main() -> None:
    """Copy legacy PDFs into one explicit user library without deleting originals."""

    parser = argparse.ArgumentParser(description="Copy the legacy paper library to a user")
    parser.add_argument("--owner", required=True, help="32-character local user UUID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    store = AppStore(settings)
    user = store.get_user(args.owner)
    if user is None:
        raise SystemExit("owner does not exist in data/app.sqlite3")
    legacy = (BASE_DIR / settings.project.data_root).resolve()
    paths = user_paths(settings, user.user_id, create=args.apply)
    pdfs = sorted(legacy.glob("*.pdf")) if legacy.exists() else []
    if not pdfs:
        print("No legacy PDFs found.")
        return
    created_paper_ids: list[str] = []
    for pdf in pdfs:
        target = paths.papers / pdf.name
        sidecar = pdf.with_suffix(".mineru.json")
        print(f"{pdf.name} -> {target}")
        if args.dry_run:
            continue
        if target.exists():
            raise SystemExit(f"target already exists: {target.name}")
        shutil.copy2(pdf, target)
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_suffix(".mineru.json"))
        with fitz.open(target) as document:
            page_count = int(document.page_count)
        paper = store.create_paper(
            user.user_id,
            pdf.stem,
            pdf.stem,
            pdf.name,
            "upload",
            status="ready" if sidecar.exists() else "queued",
            page_count=page_count,
        )
        created_paper_ids.append(str(paper["paper_id"]))
    if args.apply:
        build_index(scoped_settings(settings, user.user_id), incremental=False)
        for paper_id in created_paper_ids:
            store.update_paper(user.user_id, paper_id, status="ready", error=None)
        print(f"Copied and indexed {len(pdfs)} PDFs for {user.username}.")


if __name__ == "__main__":
    main()
