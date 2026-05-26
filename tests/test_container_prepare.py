from __future__ import annotations

from pathlib import Path

from openjury.cli import container_prepare


def test_container_prepare_uses_yaml_extra_and_persistent_home(tmp_path, monkeypatch):
    project_dir = tmp_path / "OpenJury"
    (project_dir / "openjury").mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text("[project]\nname='openjury'\n", encoding="utf-8")

    image_path = tmp_path / "vllm.sif"
    image_path.write_text("fake image", encoding="utf-8")
    home_path = tmp_path / "container-home"

    captured: dict[str, object] = {}

    def fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(container_prepare, "find_apptainer_binary", lambda: "singularity")
    monkeypatch.setattr(container_prepare.subprocess, "run", fake_run)

    container_prepare.main([
        "--container_runtime", "apptainer",
        "--container_image", str(image_path),
        "--container_home", str(home_path),
        "--project_dir", str(project_dir),
    ])

    cmd = captured["cmd"]
    assert cmd[:2] == ["singularity", "exec"]
    assert cmd[2:4] == ["--home", f"{home_path}:{home_path}"]
    assert str(image_path) in cmd
    assert f"{project_dir}:{project_dir}" in cmd
    assert f"{home_path}:{home_path}" in cmd
    assert cmd[-2] == "-lc"
    assert 'OPENJURY_PYTHON="$(command -v python || command -v python3)"' in cmd[-1]
    assert f'"$OPENJURY_PYTHON" -m pip install --user -e "{project_dir}[yaml]"' in cmd[-1]
    assert "[vllm]" not in cmd[-1]
    assert '"$OPENJURY_PYTHON" -c "import openjury; import vllm"' in cmd[-1]
    assert captured["check"] is True
    assert captured["env"]["HOME"] == str(home_path)


def test_container_prepare_requires_apptainer_runtime(tmp_path):
    project_dir = tmp_path / "OpenJury"
    (project_dir / "openjury").mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text("[project]\nname='openjury'\n", encoding="utf-8")

    image_path = tmp_path / "vllm.sif"
    image_path.write_text("fake image", encoding="utf-8")

    try:
        container_prepare.main([
            "--container_runtime", "none",
            "--container_image", str(image_path),
            "--container_home", str(tmp_path / "container-home"),
            "--project_dir", str(project_dir),
        ])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse failure for runtime=none")
