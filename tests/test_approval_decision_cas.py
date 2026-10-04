"""Approval decisions are resolved once in the store transaction."""

import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals.approval_store import ApprovalStore, DecisionConflict
from core import database
from core.database import get_db_connection


def test_decision_cas_and_duplicate(tmp_path):
    original = database.DB_PATH
    database.DB_PATH = tmp_path / "approval.db"
    try:
        store = ApprovalStore()
        request = store.create_request(operation="test.operation")
        first, created = store.resolve_decision(approval_id=request["id"], decision="deny",
                                                approver={"type": "remote_client"})
        assert created
        duplicate, created = store.resolve_decision(approval_id=request["id"], decision="deny",
                                                    approver={"type": "remote_client"})
        assert not created and duplicate["id"] == first["id"]
        with pytest.raises(DecisionConflict):
            store.resolve_decision(approval_id=request["id"], decision="approve_once")
        with get_db_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM approval_decisions WHERE approval_id = ?",
                                 (request["id"],)).fetchone()[0]
        assert count == 1
        assert store.get_request(request["id"])["status"] == "denied"
    finally:
        database.DB_PATH = original
