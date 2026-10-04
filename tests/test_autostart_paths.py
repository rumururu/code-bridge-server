"""Autostart must launch the server entry point after the system package split."""

from pathlib import Path

from system import autostart_service


def test_autostart_targets_existing_server_entry():
    root = Path(__file__).resolve().parents[1]
    assert autostart_service._get_server_dir() == root
    assert autostart_service._get_server_script_path() == root / "server_cli.py"
    assert autostart_service._get_server_script_path().is_file()


def test_autostart_uses_server_virtualenv_and_entry(monkeypatch):
    root = Path('/Volumes/test volume/code bridge/server')
    monkeypatch.setattr(autostart_service, '_get_server_dir', lambda: root)
    monkeypatch.setattr(Path, 'exists', lambda path: path == root / '.venv/bin/python')
    wrapper = autostart_service._generate_macos_wrapper()
    assert f'cd "{root}"' in wrapper
    assert f'exec "{root}/.venv/bin/python" "{root}/server_cli.py"' in wrapper
    assert '/system/server_cli.py' not in wrapper
    assert 'Wait for volume to mount' in wrapper
