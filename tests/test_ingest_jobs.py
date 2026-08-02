from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config.settings import load_settings
from services.ingest_jobs import IngestJobManager, IngestJobStore
from services.library_service import IngestReport


class IngestJobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        database = Path(self.temporary.name) / "jobs.sqlite3"
        settings = load_settings()
        self.settings = replace(
            settings,
            mineru=replace(settings.mineru, job_db_path=str(database)),
        )
        self.store = IngestJobStore(self.settings)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_confirmation_is_subset_scoped_and_single_use(self) -> None:
        proposal = self.store.create_proposal(
            "retrieval", ["2601.00001", "2601.00002"]
        )
        job = self.store.confirm(proposal.proposal_id, ["2601.00001"])
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.arxiv_ids, ("2601.00001",))
        with self.assertRaises(ValueError):
            self.store.confirm(proposal.proposal_id, ["2601.00002"])

    def test_confirmation_rejects_id_outside_proposal(self) -> None:
        proposal = self.store.create_proposal("retrieval", ["2601.00001"])
        with self.assertRaises(ValueError):
            self.store.confirm(proposal.proposal_id, ["2601.99999"])

    def test_incomplete_job_is_failed_and_retryable(self) -> None:
        proposal = self.store.create_proposal("retrieval", ["2601.00001"])
        job = self.store.confirm(proposal.proposal_id, ["2601.00001"])
        self.store.update_job(job.job_id, "parsing")
        self.store.fail_incomplete_jobs()
        failed = self.store.get_job(job.job_id)
        self.assertIsNotNone(failed)
        assert failed is not None
        self.assertEqual(failed.status, "failed")
        retried = self.store.retry(failed.job_id)
        self.assertEqual(retried.status, "queued")

    def test_expired_proposal_is_rejected(self) -> None:
        expired_settings = replace(
            self.settings,
            mineru=replace(self.settings.mineru, proposal_ttl_seconds=0),
        )
        store = IngestJobStore(expired_settings)
        proposal = store.create_proposal("retrieval", ["2601.00001"])
        with self.assertRaisesRegex(ValueError, "expired"):
            store.confirm(proposal.proposal_id, ["2601.00001"])

    def test_worker_calls_library_service_directly(self) -> None:
        manager = IngestJobManager(self.settings)
        try:
            proposal = manager.store.create_proposal("retrieval", ["2601.00001"])
            job = manager.store.confirm(proposal.proposal_id, ["2601.00001"])
            report = IngestReport(downloaded=["2601.00001"], indexed_chunks=3)
            with patch(
                "services.library_service.PaperLibraryService.ingest_arxiv",
                return_value=report,
            ) as ingest:
                manager._run(job.job_id)
            finished = manager.store.get_job(job.job_id)
            self.assertIsNotNone(finished)
            assert finished is not None
            self.assertEqual(finished.status, "succeeded")
            self.assertIsNotNone(finished.result)
            assert finished.result is not None
            self.assertEqual(finished.result["indexed_chunks"], 3)
            ingest.assert_called_once()
        finally:
            manager.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
