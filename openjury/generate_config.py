"""Typed config for standalone completion generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerateConfig:
    """Configuration for ``openjury-generate`` local generation.

    This config is intentionally task-focused (what to generate). Local-only
    execution details are limited to the output parquet path used by the
    standalone generate command.
    """

    model: str
    dataset: str
    output: str
    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42
    balance_by: str | None = None
    max_tokens: int = 4096
    truncate_input_chars: int = 8192
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    gpu_devices: str | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    use_tqdm: bool = False
    base_model: bool = False
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
    def from_dict(cls, data: dict) -> GenerateConfig:
        return cls(
            model=data["model"],
            dataset=data["dataset"],
            output=data.get("output", ""),
            n_instructions=data.get("n_instructions"),
            language=data.get("language"),
            seed=data.get("seed", 42),
            balance_by=data.get("balance_by"),
            max_tokens=data.get("max_tokens", 4096),
            truncate_input_chars=data.get("truncate_input_chars", 8192),
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
            quantization=data.get("quantization"),
            gpu_devices=data.get("gpu_devices"),
            chat_template=data.get("chat_template"),
            chat_template_file=data.get("chat_template_file"),
            api_base_url=data.get("api_base_url"),
            api_key_env=data.get("api_key_env"),
            use_tqdm=data.get("use_tqdm", False),
            base_model=data.get("base_model", False),
            ignore_cache=data.get("ignore_cache", False),
        )

    def to_dict(self) -> dict:
        data: dict[str, object] = {
            "model": self.model,
            "dataset": self.dataset,
            "output": self.output,
            "n_instructions": self.n_instructions,
            "language": self.language,
            "seed": self.seed,
            "balance_by": self.balance_by,
            "max_tokens": self.max_tokens,
            "truncate_input_chars": self.truncate_input_chars,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_tqdm": self.use_tqdm,
            "base_model": self.base_model,
            "ignore_cache": self.ignore_cache,
        }
        if self.quantization is not None:
            data["quantization"] = self.quantization
        if self.gpu_devices is not None:
            data["gpu_devices"] = self.gpu_devices
        if self.chat_template is not None:
            data["chat_template"] = self.chat_template
        if self.chat_template_file is not None:
            data["chat_template_file"] = self.chat_template_file
        if self.api_base_url is not None:
            data["api_base_url"] = self.api_base_url
        if self.api_key_env is not None:
            data["api_key_env"] = self.api_key_env
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
