"""Shared helpers for Apptainer-family runtime integration."""

from __future__ import annotations

import os
import shutil

from openjury.models.utils import provider_from_model

SUPPORTED_CONTAINER_RUNTIMES = ("none", "apptainer")


def normalize_container_runtime(value: str | None) -> str:
    """Normalize a container runtime setting."""
    runtime = (value or "none").strip().lower()
    if runtime in {"", "none"}:
        return "none"
    if runtime == "apptainer":
        return runtime
    raise EnvironmentError(
        "Unsupported container runtime "
        f"{value!r}. Expected one of: {', '.join(SUPPORTED_CONTAINER_RUNTIMES)}."
    )


def resolve_container_settings(
    *,
    cli_runtime: str | None = None,
    cli_image: str | None = None,
    cli_home: str | None = None,
    env_get=None,
) -> tuple[str, str, str]:
    """Resolve container settings with CLI > env > default precedence."""
    getter = env_get or os.environ.get
    runtime = normalize_container_runtime(
        cli_runtime if cli_runtime is not None else getter("OPENJURY_CONTAINER_RUNTIME")
    )
    image = cli_image if cli_image is not None else (getter("OPENJURY_CONTAINER_IMAGE") or "")
    if cli_home is not None:
        home = cli_home
    else:
        home = getter("OPENJURY_CONTAINER_HOME") or ""
        if not home:
            hf_home = getter("HF_HOME") or ""
            if hf_home:
                home = os.path.join(hf_home, "openjury_container_home")
    return runtime, image, home


def find_apptainer_binary() -> str | None:
    """Return the preferred Apptainer-family binary available on this system."""
    for candidate in ("apptainer", "singularity"):
        if shutil.which(candidate):
            return candidate
    return None


def is_vllm_model(model: str | None) -> bool:
    """Return True when a model string uses the VLLM provider prefix."""
    if not model:
        return False
    try:
        return provider_from_model(model) == "VLLM"
    except ValueError:
        return False
