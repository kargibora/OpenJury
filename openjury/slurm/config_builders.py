"""Config-file builders for SLURM-generated OpenJury runs."""

from __future__ import annotations

import copy

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig


def build_arena_config_dict(
    task: ArenaConfig,
    output_dir: str,
) -> dict:
    """Build an ArenaConfig JSON dict from a typed arena task."""
    arena_cfg = copy.deepcopy(task)
    arena_cfg.output_dir = output_dir
    return arena_cfg.to_dict()


def build_agreement_config_dict(
    task: AgreementConfig,
    output_dir: str,
) -> dict:
    """Build an AgreementConfig JSON dict from a typed agreement task."""
    agreement_cfg = copy.deepcopy(task)
    agreement_cfg.output_dir = output_dir
    return agreement_cfg.to_dict()


def build_annotate_config_dict(
    task: AnnotateConfig,
    output_dir: str,
) -> dict:
    """Build an AnnotateConfig JSON dict from a typed annotate task."""
    annotate_cfg = copy.deepcopy(task)
    annotate_cfg.output_dir = output_dir
    return annotate_cfg.to_dict()
