"""Typed config for generic annotation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openjury.arena.config import JudgeConfig, ModelEntry
from openjury.datasets.options import DatasetOptions
from openjury.models.factory import build_config_for_model


@dataclass
class AnnotatePairingConfig:
    """How annotation pairs are constructed."""

    source: str = "challenger_vs_dataset_inline"
    strategy: str | None = "random_single"
    seed: int = 42
    n_matches: int | None = None
    include_models: list[str] | None = None
    exclude_models: list[str] | None = None

    @classmethod
    def from_raw(
        cls,
        raw: "AnnotatePairingConfig | dict[str, Any] | None",
    ) -> "AnnotatePairingConfig":
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return cls(
                source=raw.source,
                strategy=raw.strategy,
                seed=raw.seed,
                n_matches=raw.n_matches,
                include_models=list(raw.include_models) if raw.include_models else None,
                exclude_models=list(raw.exclude_models) if raw.exclude_models else None,
            )
        source = raw.get("source", "challenger_vs_dataset_inline")
        if source == "dataset_inline":
            source = "challenger_vs_dataset_inline"
        if source == "dataset_pairs":
            default_strategy = "dataset_defined"
        elif source == "matchmaker":
            default_strategy = "round_robin"
        else:
            default_strategy = "random_single"
        return cls(
            source=source,
            strategy=raw.get("strategy", default_strategy),
            seed=int(raw.get("seed", 42) or 42),
            n_matches=raw.get("n_matches"),
            include_models=list(raw.get("include_models", []) or []) or None,
            exclude_models=list(raw.get("exclude_models", []) or []) or None,
        )

    def __post_init__(self) -> None:
        if self.source not in {
            "challenger_vs_dataset_inline",
            "dataset_pairs",
            "matchmaker",
        }:
            raise ValueError(
                "Unsupported pairing source "
                f"'{self.source}'. Available: challenger_vs_dataset_inline, dataset_pairs, matchmaker"
            )
        if self.source == "dataset_pairs":
            if self.strategy not in {None, "dataset_defined"}:
                raise ValueError(
                    "Unsupported pairing strategy "
                    f"'{self.strategy}' for source 'dataset_pairs'. "
                    "Available: dataset_defined"
                )
            self.strategy = "dataset_defined"
            return
        if self.source == "matchmaker":
            if self.strategy not in {"round_robin", "random_pairs", "balanced_random"}:
                raise ValueError(
                    f"Unsupported pairing strategy '{self.strategy}' for source 'matchmaker'. "
                    "Available: round_robin, random_pairs, balanced_random"
                )
            return
        if self.strategy not in {"random_single", "all"}:
            raise ValueError(
                f"Unsupported pairing strategy '{self.strategy}'. "
                "Available: random_single, all"
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "strategy": self.strategy,
            "seed": self.seed,
        }
        if self.n_matches is not None:
            data["n_matches"] = self.n_matches
        if self.include_models:
            data["include_models"] = list(self.include_models)
        if self.exclude_models:
            data["exclude_models"] = list(self.exclude_models)
        return data


# Keep the old importable name while the config language moves to ``pairing``.
AnnotateOpponentConfig = AnnotatePairingConfig
PairingConfig = AnnotatePairingConfig


@dataclass
class AnnotateGenerationConfig:
    """Generation settings for the challenger model."""

    max_tokens: int = 4096
    truncate_input_chars: int = 8192
    ignore_cache: bool = False
    gpu_memory_utilization: float = 0.9
    gpu_devices: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    use_tqdm: bool = False

    @classmethod
    def from_raw(
        cls,
        raw: "AnnotateGenerationConfig | dict[str, Any] | None",
    ) -> "AnnotateGenerationConfig":
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return cls(
                max_tokens=raw.max_tokens,
                truncate_input_chars=raw.truncate_input_chars,
                ignore_cache=raw.ignore_cache,
                gpu_memory_utilization=raw.gpu_memory_utilization,
                gpu_devices=raw.gpu_devices,
                api_base_url=raw.api_base_url,
                api_key_env=raw.api_key_env,
                use_tqdm=raw.use_tqdm,
            )
        return cls(
            max_tokens=int(raw.get("max_tokens", 4096) or 4096),
            truncate_input_chars=int(raw.get("truncate_input_chars", 8192) or 8192),
            ignore_cache=bool(raw.get("ignore_cache", False)),
            gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9) or 0.9),
            gpu_devices=raw.get("gpu_devices"),
            api_base_url=raw.get("api_base_url"),
            api_key_env=raw.get("api_key_env"),
            use_tqdm=bool(raw.get("use_tqdm", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "truncate_input_chars": self.truncate_input_chars,
            "ignore_cache": self.ignore_cache,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_tqdm": self.use_tqdm,
        }
        if self.gpu_devices is not None:
            data["gpu_devices"] = self.gpu_devices
        if self.api_base_url is not None:
            data["api_base_url"] = self.api_base_url
        if self.api_key_env is not None:
            data["api_key_env"] = self.api_key_env
        return data

    def to_model_config(self, entry: ModelEntry):
        effective_max_tokens = entry.max_tokens or self.max_tokens
        return build_config_for_model(
            entry.name,
            max_tokens=effective_max_tokens,
            tensor_parallel_size=entry.tensor_parallel_size,
            quantization=entry.quantization,
            chat_template=entry.chat_template,
            chat_template_file=entry.chat_template_file,
            gpu_memory_utilization=self.gpu_memory_utilization,
            gpu_devices=self.gpu_devices,
            api_base_url=self.api_base_url,
            api_key_env=self.api_key_env,
        )


@dataclass
class AnnotateConfig:
    """Configuration for generic annotation."""

    dataset: str
    judge: JudgeConfig
    challenger: ModelEntry | None = None
    models: list[ModelEntry] | None = None
    output_dir: str = "results/annotate"

    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42
    balance_by: str | None = None

    criteria: str = "default"
    pairing: AnnotatePairingConfig = field(default_factory=AnnotatePairingConfig)
    generation: AnnotateGenerationConfig = field(default_factory=AnnotateGenerationConfig)
    cache_variant: str | None = None
    ignore_score_cache: bool = False
    include_completions: bool = False
    include_raw_judge: bool = False

    @property
    def requires_challenger(self) -> bool:
        return self.pairing.source == "challenger_vs_dataset_inline"

    @property
    def requires_models(self) -> bool:
        return self.pairing.source == "matchmaker"

    @property
    def dataset_options(self) -> DatasetOptions:
        return DatasetOptions(
            name=self.dataset,
            n_instructions=self.n_instructions,
            language=self.language,
            seed=self.seed,
            balance_by=self.balance_by,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnnotateConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotateConfig":
        generation_raw = data.get("generation", {})
        pairing_raw = data.get("pairing")
        if pairing_raw is None and "opponents" in data:
            pairing_raw = data["opponents"]
        challenger_raw = data.get("challenger")
        return cls(
            dataset=data["dataset"],
            judge=JudgeConfig.from_raw(data["judge"]),
            challenger=ModelEntry.from_raw(challenger_raw) if challenger_raw else None,
            models=[ModelEntry.from_raw(raw) for raw in data.get("models", [])] or None,
            output_dir=data.get("output_dir", "results/annotate"),
            n_instructions=data.get("n_instructions"),
            language=data.get("language"),
            seed=int(data.get("seed", 42) or 42),
            balance_by=data.get("balance_by"),
            criteria=data.get("criteria", "default"),
            pairing=AnnotatePairingConfig.from_raw(pairing_raw),
            generation=AnnotateGenerationConfig.from_raw(generation_raw),
            cache_variant=data.get("cache_variant"),
            ignore_score_cache=bool(
                generation_raw.get(
                    "ignore_score_cache",
                    data.get("ignore_score_cache", False),
                )
            ),
            include_completions=bool(data.get("include_completions", False)),
            include_raw_judge=bool(data.get("include_raw_judge", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "dataset": self.dataset,
            "n_instructions": self.n_instructions,
            "language": self.language,
            "seed": self.seed,
            "balance_by": self.balance_by,
            "judge": self.judge.to_dict(),
            "criteria": self.criteria,
            "pairing": self.pairing.to_dict(),
            "generation": {
                **self.generation.to_dict(),
                "ignore_score_cache": self.ignore_score_cache,
            },
            "output_dir": self.output_dir,
            "include_completions": self.include_completions,
            "include_raw_judge": self.include_raw_judge,
        }
        if self.cache_variant:
            data["cache_variant"] = self.cache_variant
        if self.challenger is not None:
            data["challenger"] = self.challenger.to_dict()
        if self.models:
            data["models"] = [model.to_dict() for model in self.models]
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
