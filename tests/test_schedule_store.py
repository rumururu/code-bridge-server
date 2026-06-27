"""Unit tests for the task_schedules DAO + scheduler tick logic."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from agent import agent_store, schedule_store  # noqa: E402
from agent.scheduler import TaskScheduler  # noqa: E402
from core import database  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path):
    """Isolate the SQLite database per test."""
    original = database.DB_PATH
    database.DB_PATH = tmp_path / "test_schedule.db"
    # Reset singletons that may have cached the old DB
    agent_store._agent_store = None
    schedule_store._store = None
    database.init_db()
    yield
    database.DB_PATH = original
    agent_store._agent_store = None
    schedule_store._store = None


def _make_task(title: str = "Scheduled job"):
    return agent_store.get_agent_store().create_task(
        title=title,
        kind="general",
        source="test",
    )


# ---------------------------------------------------------------------------
# Expression validation
# ---------------------------------------------------------------------------


class TestValidateExpression:
    def test_interval_normalises_kind_seconds(self):
        result = schedule_store._validate_expression(
            {"kind": "interval", "seconds": 3600}
        )
        assert result == {"kind": "interval", "seconds": 3600}

    def test_interval_rejects_under_60_seconds(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression(
                {"kind": "interval", "seconds": 30}
            )

    def test_interval_rejects_non_int(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression(
                {"kind": "interval", "seconds": "3600"}
            )

    def test_daily_at_zero_pads_time(self):
        result = schedule_store._validate_expression(
            {"kind": "daily_at", "time": "9:5"}
        )
        assert result == {"kind": "daily_at", "time": "09:05"}

    def test_daily_at_rejects_bad_time(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression(
                {"kind": "daily_at", "time": "25:00"}
            )

    def test_daily_at_rejects_missing_time(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression({"kind": "daily_at"})

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression({"kind": "cron"})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            schedule_store._validate_expression("interval")


# ---------------------------------------------------------------------------
# compute_next_fire
# ---------------------------------------------------------------------------


class TestComputeNextFire:
    def test_interval_adds_seconds(self):
        base = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = schedule_store.compute_next_fire(
            {"kind": "interval", "seconds": 3600}, after=base
        )
        assert result == base + timedelta(seconds=3600)

    def test_daily_at_picks_next_occurrence_today(self):
        # 06:00 UTC, target 09:00 local — local time depends on host,
        # but the result must be in the future relative to base.
        base = datetime(2026, 5, 30, 6, 0, 0, tzinfo=timezone.utc)
        result = schedule_store.compute_next_fire(
            {"kind": "daily_at", "time": "09:00"}, after=base
        )
        assert result > base

    def test_daily_at_rolls_to_tomorrow_when_past(self):
        local_now = datetime.now().astimezone()
        # Time that has already passed today
        past_hour = (local_now - timedelta(hours=2)).strftime("%H:%M")
        result = schedule_store.compute_next_fire(
            {"kind": "daily_at", "time": past_hour}
        )
        # Should be roughly 22 hours away (within a day)
        delta = result - datetime.now(timezone.utc)
        assert delta > timedelta(hours=20)
        assert delta < timedelta(days=2)


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------


class TestScheduleStoreCrud:
    def test_create_persists_all_fields(self, tmp_db):
        task = _make_task()
        store = schedule_store.get_schedule_store()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 3600},
            name="Hourly job",
            provider_id="claude",
            model="claude-sonnet",
            cwd="/tmp/work",
            prompt="Run the thing",
            capabilities=["cap1", "cap2"],
            enabled=True,
            skip_if_active=True,
        )
        assert created["id"]
        assert created["task_id"] == task["id"]
        assert created["name"] == "Hourly job"
        assert created["expression"] == {"kind": "interval", "seconds": 3600}
        assert created["provider_id"] == "claude"
        assert created["capabilities"] == ["cap1", "cap2"]
        assert created["enabled"] is True
        assert created["next_run_at"]
        assert created["fire_count"] == 0
        assert created["skip_count"] == 0

    def test_get_returns_none_for_unknown(self, tmp_db):
        assert schedule_store.get_schedule_store().get("missing") is None

    def test_list_for_task_isolates_per_task(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task_a = _make_task("Task A")
        task_b = _make_task("Task B")
        store.create(
            task_id=task_a["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        store.create(
            task_id=task_b["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        store.create(
            task_id=task_a["id"],
            expression={"kind": "daily_at", "time": "09:00"},
        )

        a_list = store.list_for_task(task_a["id"])
        b_list = store.list_for_task(task_b["id"])
        assert len(a_list) == 2
        assert len(b_list) == 1

    def test_list_all_enabled_only_filters(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        on = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            enabled=True,
        )
        off = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            enabled=False,
        )
        all_list = store.list_all()
        assert {s["id"] for s in all_list} == {on["id"], off["id"]}
        enabled_list = store.list_all(enabled_only=True)
        assert {s["id"] for s in enabled_list} == {on["id"]}

    def test_update_enabled_toggles_state(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            enabled=True,
        )
        updated = store.update(created["id"], {"enabled": False})
        assert updated is not None
        assert updated["enabled"] is False

    def test_update_expression_recomputes_next_run_at(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        original_next = created["next_run_at"]
        updated = store.update(
            created["id"],
            {"expression": {"kind": "interval", "seconds": 7200}},
        )
        assert updated["expression"] == {"kind": "interval", "seconds": 7200}
        assert updated["next_run_at"] != original_next

    def test_update_rejects_unknown_field(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        with pytest.raises(ValueError):
            store.update(created["id"], {"forbidden_field": "x"})

    def test_delete_removes_row(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        assert store.delete(created["id"]) is True
        assert store.get(created["id"]) is None
        assert store.delete(created["id"]) is False

    def test_record_fire_increments_counter_and_advances(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        store.record_fire(
            created["id"],
            run_id="run_xyz",
            status="fired",
        )
        after = store.get(created["id"])
        assert after["fire_count"] == 1
        assert after["last_run_id"] == "run_xyz"
        assert after["last_status"] == "fired"
        assert after["next_run_at"] != created["next_run_at"]

    def test_record_fire_skipped_uses_skip_counter(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        created = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        store.record_fire(
            created["id"],
            run_id=None,
            status="skipped",
            error="previous run still active",
        )
        after = store.get(created["id"])
        assert after["fire_count"] == 0
        assert after["skip_count"] == 1
        assert after["last_status"] == "skipped"
        assert after["last_error"] == "previous run still active"


# ---------------------------------------------------------------------------
# list_due
# ---------------------------------------------------------------------------


class TestListDue:
    def test_lists_only_due_enabled_schedules(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        # Past next_run_at + enabled => due
        due = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        store.update(
            due["id"],
            {
                "expression": {"kind": "interval", "seconds": 60},
            },
        )
        # Push next_run_at to the past
        from core.database import get_db_connection

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", due["id"]),
            )
            conn.commit()

        # Disabled schedule with past next_run_at should NOT be due
        disabled = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            enabled=False,
        )
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", disabled["id"]),
            )
            conn.commit()

        # Schedule with future next_run_at is NOT due
        future = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 86400},
        )

        result = store.list_due()
        ids = {s["id"] for s in result}
        assert due["id"] in ids
        assert disabled["id"] not in ids
        assert future["id"] not in ids


# ---------------------------------------------------------------------------
# Scheduler tick orchestration
# ---------------------------------------------------------------------------


class TestSchedulerTick:
    def test_tick_skips_when_no_due(self, tmp_db):
        scheduler = TaskScheduler(tick_seconds=10)
        fired = asyncio.run(scheduler.trigger_once())
        assert fired == 0

    def test_tick_processes_due_schedules(self, tmp_db):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        sched = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
        )
        # Force due
        from core.database import get_db_connection

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", sched["id"]),
            )
            conn.commit()

        scheduler = TaskScheduler(tick_seconds=10)
        with mock.patch(
            "agent.scheduler.prepare_task_orchestration", return_value=None
        ):
            fired = asyncio.run(scheduler.trigger_once())

        assert fired == 1
        # When prepare_task_orchestration returns None (task not found path),
        # the schedule is recorded as errored and disabled.
        after = store.get(sched["id"])
        assert after["enabled"] is False
        assert after["last_status"] == "error"

    def test_due_schedule_prepares_assigned_agent_workflow_steps(self, tmp_db):
        agent = agent_store.get_agent_store().create_agent(
            name="Scheduled workflow bot",
            system_prompt="Run the assigned workflow.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "open_page",
                    "type": "browser_action",
                    "name": "Open page",
                    "tool_hint": "playwright",
                    "actions": [{"type": "navigate", "url": "https://example.test"}],
                    "on_failure": {
                        "type": "manual_handoff",
                        "prompt": "Complete browser setup, then continue.",
                    },
                },
                {
                    "id": "report",
                    "type": "llm",
                    "name": "Report",
                    "on_failure": "ask_user",
                },
            ],
        )
        task = agent_store.get_agent_store().create_task(
            title="Scheduled assigned workflow",
            assigned_agent_id=agent["id"],
            goal="Run the scheduled workflow.",
        )
        store = schedule_store.get_schedule_store()
        sched = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            provider_id="openai",
        )

        from core.database import get_db_connection

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", sched["id"]),
            )
            conn.commit()

        async def fake_execute(_execution):
            return None

        scheduler = TaskScheduler(tick_seconds=10)
        with mock.patch("agent.scheduler.execute_task_orchestration", fake_execute):
            fired = asyncio.run(scheduler.trigger_once())

        assert fired == 1
        after = store.get(sched["id"])
        assert after["last_status"] == "fired"
        assert after["last_run_id"]

        run = agent_store.get_agent_store().get_run(after["last_run_id"])
        steps = agent_store.get_agent_store().list_task_steps(task["id"])
        assert run is not None
        assert run["agent_id"] == agent["id"]
        assert [step["input"]["workflow_step_id"] for step in steps] == [
            "open_page",
            "report",
        ]
        assert steps[0]["input"]["workflow_type"] == "browser_action"
        assert steps[0]["input"]["actions"][0]["type"] == "navigate"

    @pytest.mark.parametrize(
        "run_status",
        ["blocked", "waiting_for_user", "waiting_user"],
    )
    def test_tick_skips_when_prior_run_needs_user_attention(
        self,
        tmp_db,
        run_status,
    ):
        store = schedule_store.get_schedule_store()
        task = _make_task()
        run = agent_store.get_agent_store().create_run(
            task_id=task["id"],
            title="Prior scheduled run",
        )
        agent_store.get_agent_store().update_run_status(run["id"], run_status)

        sched = store.create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 60},
            skip_if_active=True,
        )

        from core.database import get_db_connection

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", sched["id"]),
            )
            conn.commit()

        scheduler = TaskScheduler(tick_seconds=10)
        with mock.patch(
            "agent.scheduler.prepare_task_orchestration"
        ) as prepare_mock:
            fired = asyncio.run(scheduler.trigger_once())

        assert fired == 1
        prepare_mock.assert_not_called()
        after = store.get(sched["id"])
        assert after["fire_count"] == 0
        assert after["skip_count"] == 1
        assert after["last_status"] == "skipped"
        assert after["last_error"] == "previous run still active"
