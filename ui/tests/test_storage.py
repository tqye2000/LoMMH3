from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui.core.models import Project
from ui.core.storage import InstanceLockError, StorageError, Store, check_id, read_model, write_model


def test_project_round_trip_and_reopen(tmp_path: Path) -> None:
    store = Store(tmp_path)
    project = store.create_project("Demo")
    reopened = Store(tmp_path).load_project(project.id)
    assert reopened.name == "Demo"
    assert (tmp_path / "projects" / f"Demo--{project.id}" / "assets").is_dir()


def test_project_folder_name_is_readable_and_safe(tmp_path: Path) -> None:
    store = Store(tmp_path)
    project = store.create_project("../重返 旧地: v2")
    folder = store.project_dir(project.id)
    assert folder.parent == tmp_path / "projects"
    assert folder.name == f"重返-旧地-v2--{project.id}"
    assert [p.id for p in Store(tmp_path).list_projects()] == [project.id]


def test_legacy_id_only_project_folder_still_loads(tmp_path: Path) -> None:
    legacy = Project(id="prj_0123456789", name="Old", created_utc="t", updated_utc="t")
    write_model(tmp_path / "projects" / legacy.id / "project.json", legacy)
    store = Store(tmp_path)
    assert [p.name for p in store.list_projects()] == ["Old"]
    assert store.project_dir(legacy.id) == tmp_path / "projects" / legacy.id


def test_backup_recovers_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    project = Project(id="prj_0123456789", name="One", created_utc="t", updated_utc="t")
    write_model(path, project)
    project.name = "Two"
    write_model(path, project)
    path.write_text("{ truncated", encoding="utf-8")
    assert read_model(path, Project).name == "One"


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    path.write_text(json.dumps({"schema_version": 99, "id": "prj_0123456789", "name": "x",
                                "created_utc": "t", "updated_utc": "t"}), encoding="utf-8")
    with pytest.raises(StorageError):
        read_model(path, Project)


@pytest.mark.parametrize("bad", ["../etc", "prj_123", "prj_0123456789/..", ""])
def test_ids_are_path_safe(bad: str) -> None:
    with pytest.raises(StorageError):
        check_id(bad)


def test_activity_log_survives_partial_trailing_record(tmp_path: Path) -> None:
    store = Store(tmp_path)
    project = store.create_project("Demo")
    store.append_activity(project.id, "a", "first")
    with open(store.activity_path(project.id), "a", encoding="utf-8") as handle:
        handle.write('{"id": "evt_broken", "ts_ut')  # simulated crash mid-write
    store.append_activity(project.id, "b", "second")
    messages = [event.message for event in store.read_activity(project.id)]
    assert messages == ["second", "first"]


def test_second_instance_is_rejected(tmp_path: Path) -> None:
    first = Store(tmp_path)
    first.acquire_instance_lock()
    with pytest.raises(InstanceLockError):
        Store(tmp_path).acquire_instance_lock()
    first.release_instance_lock()
    Store(tmp_path).acquire_instance_lock()
