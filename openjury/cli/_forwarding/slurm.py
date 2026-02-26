"""Helpers for forwarding typed task configs to the SLURM CLI."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Literal


@dataclass(frozen=True)
class SlurmForwardRequest:
    """Execution options for forwarding a task config to ``openjury-slurm``."""

    mode: Literal["arena", "agreement", "generate"]
    submit: bool = False
    slurm_output_dir: str = "slurm_scripts"
    remote: bool = False
    cluster: str | None = None
    remote_project_dir: str | None = None
    wait_timeout: int | None = None
    warn_output_dir_semantics: bool = False


def forward_task_config_to_slurm(
    task_mode: Literal["arena", "agreement", "generate"],
    task_config,
    slurm_request: SlurmForwardRequest,
) -> None:
    """Write a temporary task config and invoke ``openjury-slurm``.

    The forwarded argv contains only SLURM/executor flags plus ``--config``.
    """
    if slurm_request.mode != task_mode:
        raise ValueError(
            f"SLURM forward mode mismatch: request={slurm_request.mode!r} "
            f"task_mode={task_mode!r}"
        )

    if task_mode == "arena":
        filename = "arena_config.json"
    elif task_mode == "agreement":
        filename = "agreement_config.json"
    else:
        filename = "generate_config.json"

    from openjury.slurm.generate_slurm import main as slurm_main

    with TemporaryDirectory(prefix="openjury_slurm_") as td:
        from pathlib import Path

        cfg_path = Path(td) / filename
        task_config.save(cfg_path)

        slurm_argv: list[str] = [
            "--mode",
            task_mode,
            "--config",
            str(cfg_path),
            "--output_dir",
            slurm_request.slurm_output_dir,
        ]
        if slurm_request.submit:
            slurm_argv.append("--submit")
        if slurm_request.remote:
            slurm_argv.append("--remote")
        if slurm_request.cluster:
            slurm_argv += ["--cluster", slurm_request.cluster]
        if slurm_request.remote_project_dir:
            slurm_argv += ["--remote_project_dir", slurm_request.remote_project_dir]
        if slurm_request.wait_timeout is not None:
            slurm_argv += ["--wait_timeout", str(slurm_request.wait_timeout)]

        slurm_main(slurm_argv)
