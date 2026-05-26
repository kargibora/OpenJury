"""Prepare a persistent Apptainer home for containerized OpenJury jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from openjury import env as _env_mod
from openjury._logging import logger
from openjury.container_support import find_apptainer_binary, resolve_container_settings
from openjury.project_paths import normalize_project_dir


def _build_prepare_command(project_dir: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            'export PATH="$HOME/.local/bin:$PATH"',
            'OPENJURY_PYTHON="$(command -v python || command -v python3)"',
            'if [ -z "${OPENJURY_PYTHON:-}" ]; then',
            '  echo "ERROR: Neither python nor python3 is available inside the container." >&2',
            "  exit 1",
            "fi",
            f'"$OPENJURY_PYTHON" -m pip install --user -e "{project_dir}[yaml]"',
            "command -v openjury-evaluate >/dev/null",
            '"$OPENJURY_PYTHON" -c "import openjury; import vllm"',
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openjury-container-prepare",
        description=(
            "Prepare a persistent Apptainer/Singularity home with OpenJury installed "
            "for containerized VLLM SLURM jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  openjury-container-prepare\n"
            "  openjury-container-prepare --container_image /path/to/vllm-gemma4.sif\n"
            "  openjury-container-prepare --project_dir /path/to/OpenJury\n"
        ),
    )
    parser.add_argument(
        "--container_runtime",
        choices=["none", "apptainer"],
        default=None,
        help="Override $OPENJURY_CONTAINER_RUNTIME.",
    )
    parser.add_argument(
        "--container_image",
        default=None,
        help="Override $OPENJURY_CONTAINER_IMAGE.",
    )
    parser.add_argument(
        "--container_home",
        default=None,
        help="Override $OPENJURY_CONTAINER_HOME.",
    )
    parser.add_argument(
        "--project_dir",
        default=None,
        help="Override $OPENJURY_PROJECT_DIR.",
    )

    args = parser.parse_args(argv)
    _env_mod.load_dotenv()

    runtime, image, home = resolve_container_settings(
        cli_runtime=args.container_runtime,
        cli_image=args.container_image,
        cli_home=args.container_home,
        env_get=_env_mod.get,
    )
    if runtime != "apptainer":
        parser.error(
            "Container preparation requires OPENJURY container runtime 'apptainer'. "
            f"Resolved runtime: {runtime!r}."
        )

    container_bin = find_apptainer_binary()
    if container_bin is None:
        parser.error(
            "Neither 'apptainer' nor 'singularity' is available on PATH."
        )

    if not image:
        parser.error(
            "Container image is required. Set $OPENJURY_CONTAINER_IMAGE or pass --container_image."
        )
    image_path = Path(image).expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Container image does not exist: {image_path}")

    if not home:
        parser.error(
            "Container home is required. Set $OPENJURY_CONTAINER_HOME or pass --container_home."
        )
    home_path = Path(home).expanduser().resolve()
    home_path.mkdir(parents=True, exist_ok=True)

    auto_project = str(Path(__file__).resolve().parents[2])
    project_dir = normalize_project_dir(
        args.project_dir
        or _env_mod.get("OPENJURY_PROJECT_DIR", auto_project)
        or auto_project
    )

    bind_args: list[str] = []
    for bind_path in (project_dir, str(home_path)):
        bind_args += ["-B", f"{bind_path}:{bind_path}"]

    hf_home = _env_mod.get("HF_HOME")
    if hf_home:
        hf_path = Path(hf_home).expanduser().resolve()
        hf_path.mkdir(parents=True, exist_ok=True)
        bind_args += ["-B", f"{hf_path}:{hf_path}"]

    openjury_data = _env_mod.get("OPENJURY_DATA")
    if openjury_data:
        data_path = Path(openjury_data).expanduser().resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        bind_args += ["-B", f"{data_path}:{data_path}"]

    env = os.environ.copy()
    env["HOME"] = str(home_path)
    if hf_home:
        env["HF_HOME"] = str(Path(hf_home).expanduser().resolve())
    if openjury_data:
        env["OPENJURY_DATA"] = str(Path(openjury_data).expanduser().resolve())

    logger.info("Preparing OpenJury container home")
    logger.info("  Runtime: %s", container_bin)
    logger.info("  Image:   %s", image_path)
    logger.info("  Home:    %s", home_path)
    logger.info("  Project: %s", project_dir)

    subprocess.run(
        [
            container_bin,
            "exec",
            "--home",
            f"{home_path}:{home_path}",
            *bind_args,
            str(image_path),
            "bash",
            "-lc",
            _build_prepare_command(project_dir),
        ],
        check=True,
        env=env,
    )

    logger.info("✅ Container home prepared: %s", home_path)


if __name__ == "__main__":
    main()
