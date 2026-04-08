"""Typed config for standalone completion generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from openjury.arena.config import ModelEntry
from openjury.datasets.options import DatasetOptions


@dataclass
class GenerateRuntimeConfig:
    """Runtime-only options for standalone generation."""

    truncate_input_chars: int = 8192
    gpu_memory_utilization: float = 0.9
    gpu_devices: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    use_tqdm: bool = False
    base_model: bool = False
    ignore_cache: bool = False

    @classmethod
    def from_raw(
        cls,
        raw: GenerateRuntimeConfig | dict[str, Any] | None,
    ) -> GenerateRuntimeConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return cls(
                truncate_input_chars=raw.truncate_input_chars,
                gpu_memory_utilization=raw.gpu_memory_utilization,
                gpu_devices=raw.gpu_devices,
                api_base_url=raw.api_base_url,
                api_key_env=raw.api_key_env,
                use_tqdm=raw.use_tqdm,
                base_model=raw.base_model,
                ignore_cache=raw.ignore_cache,
            )
        return cls(
            truncate_input_chars=raw.get("truncate_input_chars", 8192),
            gpu_memory_utilization=raw.get("gpu_memory_utilization", 0.9),
            gpu_devices=raw.get("gpu_devices"),
            api_base_url=raw.get("api_base_url"),
            api_key_env=raw.get("api_key_env"),
            use_tqdm=raw.get("use_tqdm", False),
            base_model=raw.get("base_model", False),
            ignore_cache=raw.get("ignore_cache", False),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "truncate_input_chars": self.truncate_input_chars,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_tqdm": self.use_tqdm,
            "base_model": self.base_model,
            "ignore_cache": self.ignore_cache,
        }
        if self.gpu_devices is not None:
            data["gpu_devices"] = self.gpu_devices
        if self.api_base_url is not None:
            data["api_base_url"] = self.api_base_url
        if self.api_key_env is not None:
            data["api_key_env"] = self.api_key_env
        return data


@dataclass(init=False)
class GenerateConfig:
    """Configuration for ``openjury-generate`` local generation.

    Shared model settings are stored structurally on ``model_entry`` while a
    small set of compatibility properties (``model``, ``max_tokens``, etc.)
    remains available for older call sites.
    """

    dataset_options: DatasetOptions
    model_entry: ModelEntry
    runtime: GenerateRuntimeConfig
    output: str

    DATASET_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "dataset",
            "n_instructions",
            "language",
            "seed",
            "balance_by",
        }
    )

    MODEL_ENTRY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "max_tokens",
            "tensor_parallel_size",
            "quantization",
            "chat_template",
            "chat_template_file",
        }
    )
    RUNTIME_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "output",
            "truncate_input_chars",
            "gpu_memory_utilization",
            "gpu_devices",
            "api_base_url",
            "api_key_env",
            "use_tqdm",
            "base_model",
            "ignore_cache",
        }
    )

    def __init__(
        self,
        model: str | dict[str, Any] | ModelEntry,
        dataset: str | dict[str, Any] | DatasetOptions,
        output: str,
        runtime: GenerateRuntimeConfig | dict[str, Any] | None = None,
    ) -> None:
        self.dataset_options = self._normalize_dataset_options(dataset)
        self.model_entry = self._normalize_model_entry(model)
        self.runtime = GenerateRuntimeConfig.from_raw(runtime)
        self.output = output

    @staticmethod
    def _normalize_dataset_options(
        dataset: str | dict[str, Any] | DatasetOptions,
    ) -> DatasetOptions:
        if isinstance(dataset, DatasetOptions):
            return DatasetOptions(
                name=dataset.name,
                n_instructions=dataset.n_instructions,
                language=dataset.language,
                seed=dataset.seed,
                balance_by=dataset.balance_by,
            )
        if isinstance(dataset, dict):
            name = dataset.get("name") or dataset.get("dataset")
            if not name:
                raise ValueError("Nested dataset config must include 'name'.")
            return DatasetOptions(
                name=str(name),
                n_instructions=dataset.get("n_instructions"),
                language=dataset.get("language"),
                seed=int(dataset.get("seed", 42) or 42),
                balance_by=dataset.get("balance_by"),
            )
        return DatasetOptions(name=dataset)

    @staticmethod
    def _normalize_model_entry(
        model: str | dict[str, Any] | ModelEntry,
    ) -> ModelEntry:
        entry = (
            ModelEntry.from_raw(model.to_dict())
            if isinstance(model, ModelEntry)
            else ModelEntry.from_raw(model)
        )
        return ModelEntry(
            name=entry.name,
            gpus=entry.gpus,
            quantization=entry.quantization,
            chat_template=entry.chat_template,
            chat_template_file=entry.chat_template_file,
            max_tokens=entry.max_tokens,
            temperature=entry.temperature,
            top_p=entry.top_p,
            generation_kwargs=dict(entry.generation_kwargs),
        )

    def _replace_dataset_options(self, **updates: Any) -> None:
        current = self.dataset_options
        self.dataset_options = DatasetOptions(
            name=updates.get("name", current.name),
            n_instructions=updates.get("n_instructions", current.n_instructions),
            language=updates.get("language", current.language),
            seed=updates.get("seed", current.seed),
            balance_by=updates.get("balance_by", current.balance_by),
        )

    def _replace_model_entry(self, **updates: Any) -> None:
        current = self.model_entry
        self.model_entry = ModelEntry(
            name=updates.get("name", current.name),
            gpus=updates.get("gpus", current.gpus),
            quantization=updates.get("quantization", current.quantization),
            chat_template=updates.get("chat_template", current.chat_template),
            chat_template_file=updates.get(
                "chat_template_file",
                current.chat_template_file,
            ),
            max_tokens=updates.get("max_tokens", current.max_tokens),
            temperature=current.temperature,
            top_p=current.top_p,
            generation_kwargs=dict(current.generation_kwargs),
        )

    def to_model_config(self):
        """Build a typed backend config from the structured model view."""
        from openjury.models.factory import build_config_for_model

        entry = self.model_entry
        runtime = self.runtime
        return build_config_for_model(
            entry.name,
            max_tokens=entry.max_tokens or 4096,
            tensor_parallel_size=entry.tensor_parallel_size,
            gpu_memory_utilization=runtime.gpu_memory_utilization,
            quantization=entry.quantization,
            gpu_devices=runtime.gpu_devices,
            chat_template=entry.chat_template,
            chat_template_file=entry.chat_template_file,
            api_base_url=runtime.api_base_url,
            api_key_env=runtime.api_key_env,
        )

    def set_model_entry(self, entry: ModelEntry) -> None:
        """Replace the structured model entry with a detached copy."""
        self.model_entry = ModelEntry.from_raw(entry.to_dict())

    def set_runtime(self, runtime: GenerateRuntimeConfig) -> None:
        """Replace the structured runtime config with a detached copy."""
        self.runtime = GenerateRuntimeConfig.from_raw(runtime)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply partial config overrides, using ModelEntry for shared model fields."""
        if not overrides:
            return

        dataset_updates: dict[str, Any] = {}
        model_updates: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in self.DATASET_FIELDS:
                dataset_updates["name" if key == "dataset" else key] = value
            elif key in self.MODEL_ENTRY_FIELDS:
                mapped_key = {
                    "model": "name",
                    "tensor_parallel_size": "gpus",
                }.get(key, key)
                model_updates[mapped_key] = value
            elif key in self.RUNTIME_FIELDS:
                setattr(self.runtime, key, value)
        if dataset_updates:
            self._replace_dataset_options(**dataset_updates)
        if model_updates:
            self._replace_model_entry(**model_updates)

    @classmethod
    def load(cls, path: str | Path) -> GenerateConfig:
        """Load config from JSON or YAML file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML configs. "
                    "Install with: pip install pyyaml"
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def upgrade_legacy_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Upgrade a legacy flat generate config to the nested schema.

        If the input already uses the nested shape written by ``to_dict()``,
        it is returned unchanged (as a shallow copy).
        """
        upgraded = dict(data)
        dataset_raw = upgraded.get("dataset")
        model_raw = upgraded.get("model")
        runtime_raw = upgraded.get("runtime")

        dataset_nested = isinstance(dataset_raw, dict)
        model_nested = isinstance(model_raw, dict)
        runtime_nested = isinstance(runtime_raw, dict)

        if dataset_nested and model_nested and runtime_nested:
            return upgraded

        if "dataset" not in upgraded or "model" not in upgraded:
            raise ValueError("Legacy generate config must include 'dataset' and 'model'.")

        dataset_block: dict[str, Any]
        if dataset_nested:
            dataset_block = dict(dataset_raw)
        else:
            dataset_block = {"name": dataset_raw}
        for key in ("n_instructions", "language", "seed", "balance_by"):
            if key in upgraded and key not in dataset_block:
                dataset_block[key] = upgraded.pop(key)
        dataset_block.setdefault("seed", 42)

        if model_nested:
            model_block = dict(model_raw)
        else:
            model_entry = ModelEntry.from_raw(model_raw)
            model_block = model_entry.to_dict()
        for old_key, new_key in (
            ("max_tokens", "max_tokens"),
            ("tensor_parallel_size", "gpus"),
            ("quantization", "quantization"),
            ("chat_template", "chat_template"),
            ("chat_template_file", "chat_template_file"),
        ):
            if old_key in upgraded and new_key not in model_block:
                model_block[new_key] = upgraded.pop(old_key)

        runtime_block = dict(runtime_raw) if runtime_nested else {}
        for key in GenerateRuntimeConfig.__annotations__:
            if key in upgraded and key not in runtime_block:
                runtime_block[key] = upgraded.pop(key)

        upgraded["dataset"] = dataset_block
        upgraded["model"] = model_block
        upgraded["runtime"] = runtime_block
        return upgraded

    @classmethod
    def from_dict(cls, data: dict) -> GenerateConfig:
        """Load from a nested config dict."""
        dataset_raw = data.get("dataset")
        model_raw = data.get("model")
        runtime_raw = data.get("runtime")

        if not isinstance(dataset_raw, dict):
            raise ValueError(
                "GenerateConfig expects a nested 'dataset' object. "
                "Legacy flat generate configs are no longer supported."
            )
        if not isinstance(model_raw, dict):
            raise ValueError(
                "GenerateConfig expects a nested 'model' object. "
                "Use the format written by GenerateConfig.to_dict()."
            )
        if not isinstance(runtime_raw, dict):
            raise ValueError(
                "GenerateConfig expects a nested 'runtime' object. "
                "Use the format written by GenerateConfig.to_dict()."
            )

        return cls(
            model=ModelEntry.from_raw(model_raw),
            dataset=dataset_raw,
            runtime=runtime_raw,
            output=data.get("output", ""),
        )

    def to_dict(self) -> dict:
        data: dict[str, object] = {
            "dataset": {
                "name": self.dataset_options.name,
                "seed": self.dataset_options.seed,
            },
            "model": self.model_entry.to_dict(),
            "output": self.output,
            "runtime": self.runtime.to_dict(),
        }
        if self.dataset_options.n_instructions is not None:
            data["dataset"]["n_instructions"] = self.dataset_options.n_instructions
        if self.dataset_options.language is not None:
            data["dataset"]["language"] = self.dataset_options.language
        if self.dataset_options.balance_by is not None:
            data["dataset"]["balance_by"] = self.dataset_options.balance_by
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
