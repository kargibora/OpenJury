"""Typed SLURM planning objects.

These types split task configuration (what to run) from execution
configuration (where/how SLURM should run it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, ModelEntry


@dataclass
class GenerateTaskConfig:
    """Task config for SLURM multi-model completion generation."""

    dataset: str
    models: list[ModelEntry]
    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42
    balance_by: str | None = None
    generation_max_tokens: int = 4096
    truncate_input_chars: int = 8192
    ignore_cache: bool = False

    @property
    def dataset_options(self):
        from openjury.datasets.options import DatasetOptions

        return DatasetOptions(
            name=self.dataset,
            n_instructions=self.n_instructions,
            language=self.language,
            seed=self.seed,
            balance_by=self.balance_by,
        )

    @property
    def model_names(self) -> list[str]:
        return [model.name for model in self.models]


TaskConfig: TypeAlias = ArenaConfig | AgreementConfig | GenerateTaskConfig | AnnotateConfig


@dataclass
class SlurmExecutionConfig:
    """Execution-only settings for SLURM planning and script rendering."""

    mode: str = "arena"
    stage: str = "all"
    judge_local: bool = True
    tag: str = ""
    partition: str = ""
    account: str = ""
    time_generate: str = ""
    time_judge: str = ""
    qos: str = "normal"
    project_dir: str = ""
    work_dir: str = ""
    logs_dir: str = ""


@dataclass
class SlurmRunPlan:
    """Typed SLURM run plan: task config plus execution config."""

    task: TaskConfig
    execution: SlurmExecutionConfig

    @property
    def dataset(self) -> str:
        return self.task.dataset

    @property
    def dataset_options(self):
        return self.task.dataset_options

    @property
    def models(self) -> list[ModelEntry]:
        return list(getattr(self.task, "models", []))

    @property
    def model_names(self) -> list[str]:
        if hasattr(self.task, "model_names"):
            return list(self.task.model_names)
        challenger = getattr(self.task, "challenger", None)
        if challenger is not None and getattr(challenger, "name", None):
            return [challenger.name]
        return [model.name for model in self.models]

    @property
    def judge(self) -> JudgeConfig:
        judge = getattr(self.task, "judge", None)
        if judge is None:
            raise AttributeError("This SLURM task has no judge configuration.")
        return judge

    @property
    def has_judge(self) -> bool:
        return hasattr(self.task, "judge")

    @property
    def n_instructions(self) -> int | None:
        return getattr(self.task, "n_instructions", None)

    @property
    def criteria(self) -> str:
        return getattr(self.task, "criteria", "default")

    @property
    def generation_max_tokens(self) -> int:
        return getattr(self.task, "generation_max_tokens", 4096)

    @property
    def truncate_input_chars(self) -> int:
        return getattr(self.task, "truncate_input_chars", 8192)

    @property
    def ignore_cache(self) -> bool:
        return bool(getattr(self.task, "ignore_cache", False))

    @property
    def ignore_score_cache(self) -> bool:
        return bool(getattr(self.task, "ignore_score_cache", False))

    @property
    def matchmaker(self) -> str:
        matchmaker = getattr(self.task, "matchmaker", None)
        if matchmaker is None:
            return "round_robin"
        return getattr(matchmaker, "strategy", matchmaker)

    @property
    def n_matches(self) -> int | None:
        matchmaker = getattr(self.task, "matchmaker", None)
        if matchmaker is None:
            return None
        return getattr(matchmaker, "n_matches", None)

    @property
    def truncate_instruction(self) -> int:
        return getattr(self.task, "truncate_instruction", 500)
