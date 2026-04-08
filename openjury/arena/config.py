"""Core data structures for the K-model arena framework.

**Input configs** — :class:`ArenaConfig` and :class:`AgreementConfig`
drive the arena and agreement pipelines respectively.
Load from a JSON/YAML file or construct programmatically::

    config = ArenaConfig.load("arena.json")
    config = AgreementConfig.load("agreement.json")

**Output types** — :class:`ArenaResult`, :class:`MatchResult`, etc.
are produced by the arena pipeline and serialised to ``arena.json``.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any


# ═════════════════════════════════════════════════════════════════════
#  Input configuration — drives the arena pipeline
# ═════════════════════════════════════════════════════════════════════


def _dataclass_field_default(dc_field) -> Any:
    """Return the effective default value for a dataclass field."""
    if dc_field.default is not MISSING:
        return dc_field.default
    if dc_field.default_factory is not MISSING:
        return dc_field.default_factory()
    return MISSING


def _dataclass_kwargs_from_raw(cls, raw: dict[str, Any]) -> dict[str, Any]:
    """Filter a raw dict down to keys accepted by the dataclass."""
    field_names = {dc_field.name for dc_field in fields(cls)}
    return {key: value for key, value in raw.items() if key in field_names}


def _dataclass_to_sparse_dict(instance: Any) -> dict[str, Any]:
    """Serialize a dataclass, omitting fields whose values match defaults."""
    out: dict[str, Any] = {}
    for dc_field in fields(instance):
        value = getattr(instance, dc_field.name)
        default = _dataclass_field_default(dc_field)
        if default is not MISSING and value == default:
            continue
        out[dc_field.name] = value
    return out


@dataclass
class ModelEntry:
    """A model participating in the arena.

    In a config file, can be specified as a plain string
    (``"VLLM/Qwen/Qwen2.5-0.5B-Instruct"``) or as a dict with
    per-model overrides::

        models:
          - VLLM/Qwen/Qwen2.5-0.5B-Instruct          # string shorthand
          - name: VLLM/meta-llama/Llama-3.1-8B-Instruct
            gpus: 4
            quantization: fp8
            generation_kwargs:
              temperature: 0.7
              top_p: 0.9
          - name: OpenRouter/qwen/qwen3-235b-a22b      # API model
          - name: local-model
            completions: path/to/completions.parquet    # pre-existing

    Attributes:
        name: Full model spec (e.g. ``"VLLM/Qwen/Qwen2.5-0.5B-Instruct"``).
        gpus: Number of GPUs to allocate (ignored for API providers).
        tp: Tensor-parallelism size. Defaults to ``gpus`` if not set.
        quantization: Weight quantization (``awq``, ``gptq``, ``fp8``, …).
        completions: Path to a pre-existing parquet file.  When set the
            model is not generated — the file is loaded directly.
        max_tokens: Generation max tokens (overrides global default).
        temperature: Sampling temperature (default: ``None`` → uses
            the backend default of 0.6).
        top_p: Top-p / nucleus sampling (default: ``None`` → uses
            the backend default of 0.95).
        generation_kwargs: Extra sampling kwargs passed through to the
            backend (e.g. ``top_k``, ``repetition_penalty``).
    """

    name: str
    gpus: int = 1
    tp: int | None = None
    quantization: str | None = None
    completions: str | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        """Last component of the model path (e.g. ``Qwen2.5-0.5B-Instruct``)."""
        return self.name.rsplit("/", 1)[-1]

    @property
    def tensor_parallel_size(self) -> int:
        return self.tp if self.tp is not None else self.gpus

    @classmethod
    def from_raw(cls, raw: str | dict) -> ModelEntry:
        """Parse a config entry: plain string or dict."""
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, dict):
            return cls(**_dataclass_kwargs_from_raw(cls, raw))
        raise TypeError(f"Expected str or dict for model entry, got {type(raw)}")

    def to_dict(self) -> str | dict[str, Any]:
        """Serialize to a sparse JSON-compatible model entry."""
        sparse = _dataclass_to_sparse_dict(self)
        if set(sparse.keys()) == {"name"}:
            return self.name
        return sparse

    def to_model_config(self):
        """Build a typed :class:`~openjury.models.config.ModelConfig` for this entry.

        Calls :func:`~openjury.models.factory.build_config_for_model`
        internally so the correct provider-specific subclass is returned
        (e.g. ``VLLMConfig`` for ``VLLM/…`` models).

        Returns:
            A fully configured :class:`ModelConfig` subclass instance.
        """
        from openjury.models.factory import build_config_for_model

        return build_config_for_model(
            self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            chat_template=self.chat_template,
            chat_template_file=self.chat_template_file,
            generation_kwargs=self.generation_kwargs or None,
        )


@dataclass
class JudgeConfig:
    """Configuration for the arena judge.

    Example (JSON/YAML)::

        judge:
          model: VLLM/Qwen/Qwen3-32B
          gpus: 2
          mode: samplewise
          max_tokens: 2048
          temperature: 0.0
          top_p: 1.0
          enable_thinking: true
    """

    model: str
    gpus: int = 1
    tp: int | None = None
    mode: str = "samplewise"       # "samplewise" or "pairwise"
    max_tokens: int = 2048
    temperature: float = 0.0       # deterministic by default
    top_p: float = 1.0             # no nucleus filtering by default
    quantization: str | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None
    pairwise_prompt_style: str = "criteria"
    provide_explanation: bool = False
    no_swap: bool = False           # disable swap debiasing in pairwise mode
    enable_thinking: bool | None = None  # for Qwen3 thinking mode
    max_model_len: int | None = None   # vLLM/SGLang max sequence length (None = model default)
    enforce_eager: bool = False        # disable CUDA graphs (saves GPU memory)
    n_trials: int = 1                  # Average-of-K: run judge K times per sample and aggregate
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.n_trials > 1 and self.temperature == 0.0:
            import warnings
            warnings.warn(
                f"n_trials={self.n_trials} with temperature=0.0 produces identical "
                f"outputs. Setting n_trials=1. Use temperature > 0 for multi-trial.",
                stacklevel=2,
            )
            object.__setattr__(self, "n_trials", 1)

    @property
    def tensor_parallel_size(self) -> int:
        return self.tp if self.tp is not None else self.gpus

    @classmethod
    def from_raw(cls, raw: str | dict) -> JudgeConfig:
        """Parse: plain model string or full dict."""
        if isinstance(raw, str):
            return cls(model=raw)
        if isinstance(raw, dict):
            kw = _dataclass_kwargs_from_raw(cls, raw)
            # Normalize legacy "rubric" value → "criteria"
            if kw.get("pairwise_prompt_style") == "rubric":
                kw["pairwise_prompt_style"] = "criteria"
            return cls(**kw)
        raise TypeError(f"Expected str or dict for judge config, got {type(raw)}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict, omitting default-valued fields."""
        return _dataclass_to_sparse_dict(self)

    def to_model_config(self):
        """Build a typed :class:`~openjury.models.config.ModelConfig` for this judge.

        Calls :func:`~openjury.models.factory.build_config_for_model`
        internally so the correct provider-specific subclass is returned
        (e.g. ``VLLMConfig`` for ``VLLM/…`` models).

        Returns:
            A fully configured :class:`ModelConfig` subclass instance.
        """
        from openjury.models.factory import build_config_for_model

        return build_config_for_model(
            self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            chat_template=self.chat_template,
            chat_template_file=self.chat_template_file,
            enable_thinking=self.enable_thinking,
            generation_kwargs=self.generation_kwargs or None,
            # vLLM/SGLang engine-level options (passed via **extra)
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
        )


@dataclass
class MatchmakerConfig:
    """Matchmaker configuration.

    Example::

        matchmaker:
          strategy: balanced_random
          n_matches: 500
    """

    strategy: str = "round_robin"   # round_robin | random_pairs | balanced_random
    n_matches: int | None = None    # budget (null = auto)

    @classmethod
    def from_raw(cls, raw: str | dict | None) -> MatchmakerConfig:
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(strategy=raw)
        if isinstance(raw, dict):
            return cls(**raw)
        raise TypeError(f"Expected str or dict for matchmaker config, got {type(raw)}")


@dataclass
class ArenaConfig:
    """Complete arena pipeline configuration.

    Load from a JSON or YAML file, or construct programmatically::

        config = ArenaConfig.load("arena.json")
        config = ArenaConfig.load("arena.yaml")

    Minimal JSON example::

        {
            "dataset": "alpaca-eval",
            "models": [
                "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
                "VLLM/Qwen/Qwen2.5-1.5B-Instruct"
            ],
            "judge": {"model": "VLLM/Qwen/Qwen3-32B", "gpus": 2},
            "output_dir": "results/arena/"
        }

    Full YAML example::

        dataset: alpaca-eval
        n_instructions: 100

        models:
          - VLLM/Qwen/Qwen2.5-0.5B-Instruct
          - name: VLLM/meta-llama/Llama-3.1-8B-Instruct
            gpus: 4
            quantization: fp8
          - name: OpenRouter/qwen/qwen3-235b-a22b

        judge:
          model: VLLM/Qwen/Qwen3-32B
          gpus: 2
          mode: samplewise

        criteria: default
        matchmaker:
          strategy: round_robin

        generation:
          max_tokens: 4096
          ignore_cache: false
          ignore_score_cache: false

        output_dir: results/arena/
        include_completions: false
        include_raw_judge: false

        ratings:
          bt_regularization: 0.01
          elo_k: 32.0
    """

    # ── Core ─────────────────────────────────────────────────────
    dataset: str
    models: list[ModelEntry]
    judge: JudgeConfig
    output_dir: str = "results/arena/"

    # ── Dataset ──────────────────────────────────────────────────
    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42
    balance_by: str | None = None

    # ── Criteria ─────────────────────────────────────────────────
    criteria: str = "default"       # name or path to JSON criteria file

    # ── Matchmaker ───────────────────────────────────────────────
    matchmaker: MatchmakerConfig = field(default_factory=MatchmakerConfig)

    # ── Generation defaults ──────────────────────────────────────
    generation_max_tokens: int = 4096
    truncate_input_chars: int = 8192
    ignore_cache: bool = False
    ignore_score_cache: bool = False

    # ── Rating parameters ────────────────────────────────────────
    bt_regularization: float = 0.01
    elo_k: float = 32.0

    # ── Output options ───────────────────────────────────────────
    include_completions: bool = False
    include_raw_judge: bool = False

    # ── Convenience ──────────────────────────────────────────────

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def dataset_options(self):
        """Shared dataset selection options for loaders/executors."""
        from openjury.datasets.options import DatasetOptions
        return DatasetOptions(
            name=self.dataset,
            n_instructions=self.n_instructions,
            language=self.language,
            seed=self.seed,
            balance_by=self.balance_by,
        )

    # ── Loaders ──────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> ArenaConfig:
        """Load config from JSON or YAML file.

        YAML requires ``pyyaml`` to be installed.
        """
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
    def from_dict(cls, data: dict) -> ArenaConfig:
        """Construct from a raw dict (parsed JSON/YAML)."""
        # Parse models — accept strings or dicts
        raw_models = data.get("models", [])
        models = [ModelEntry.from_raw(m) for m in raw_models]

        # Parse judge — accept string or dict
        judge = JudgeConfig.from_raw(data.get("judge", data.get("judge_model", "")))

        # Parse matchmaker — accept string or dict
        matchmaker = MatchmakerConfig.from_raw(data.get("matchmaker"))

        # Parse generation block (flattened or nested)
        gen = data.get("generation", {})
        generation_max_tokens = gen.get("max_tokens", data.get("generation_max_tokens", 4096))
        truncate_input_chars = gen.get("truncate_input_chars", data.get("truncate_input_chars", 8192))
        ignore_cache = gen.get("ignore_cache", data.get("ignore_cache", False))
        ignore_score_cache = gen.get("ignore_score_cache", data.get("ignore_score_cache", False))

        # Parse ratings block (flattened or nested)
        ratings = data.get("ratings", {})
        bt_reg = ratings.get("bt_regularization", data.get("bt_regularization", 0.01))
        elo_k = ratings.get("elo_k", data.get("elo_k", 32.0))

        return cls(
            dataset=data["dataset"],
            models=models,
            judge=judge,
            output_dir=data.get("output_dir", "results/arena/"),
            n_instructions=data.get("n_instructions"),
            language=data.get("language"),
            seed=data.get("seed", 42),
            balance_by=data.get("balance_by"),
            criteria=data.get("criteria", data.get("rubric", "default")),
            matchmaker=matchmaker,
            generation_max_tokens=generation_max_tokens,
            truncate_input_chars=truncate_input_chars,
            ignore_cache=ignore_cache,
            ignore_score_cache=ignore_score_cache,
            bt_regularization=bt_reg,
            elo_k=elo_k,
            include_completions=data.get("include_completions", False),
            include_raw_judge=data.get("include_raw_judge", False),
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-serialisable dict."""
        # ── models ───────────────────────────────────────────────
        models_out = [m.to_dict() for m in self.models]

        return {
            "dataset": self.dataset,
            "n_instructions": self.n_instructions,
            "language": self.language,
            "seed": self.seed,
            "balance_by": self.balance_by,
            "models": models_out,
            "judge": self.judge.to_dict(),
            "criteria": self.criteria,
            "matchmaker": {
                "strategy": self.matchmaker.strategy,
                "n_matches": self.matchmaker.n_matches,
            },
            "generation": {
                "max_tokens": self.generation_max_tokens,
                "truncate_input_chars": self.truncate_input_chars,
                "ignore_cache": self.ignore_cache,
                "ignore_score_cache": self.ignore_score_cache,
            },
            "output_dir": self.output_dir,
            "ratings": {
                "bt_regularization": self.bt_regularization,
                "elo_k": self.elo_k,
            },
            "include_completions": self.include_completions,
            "include_raw_judge": self.include_raw_judge,
        }

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════
#  Agreement pipeline config
# ═════════════════════════════════════════════════════════════════════


@dataclass
class AgreementConfig:
    """Complete agreement pipeline configuration.

    Load from a JSON file or construct programmatically::

        config = AgreementConfig.load("agreement.json")

    JSON example::

        {
            "dataset": "lmsys",
            "n_instructions": 250,
            "judge": {
                "model": "OpenRouter/deepseek/deepseek-v3.2",
                "gpus": 4,
                "mode": "samplewise",
                "max_tokens": 4096,
                "temperature": 0.0,
                "enable_thinking": false
            },
            "criteria": "default",
            "output_dir": "results/agreement/lmsys_deepseek3",
            "language": null,
            "seed": 42
        }
    """

    # ── Core ─────────────────────────────────────────────────────
    dataset: str
    judge: JudgeConfig
    output_dir: str = "results/agreement"

    # ── Dataset ──────────────────────────────────────────────────
    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42
    balance_by: str | None = None

    # ── Criteria ─────────────────────────────────────────────────────
    criteria: str = "default"

    # ── Scoring ──────────────────────────────────────────────────
    ignore_score_cache: bool = False
    truncate_instruction: int = 500
    include_completions: bool = True

    @property
    def dataset_options(self):
        """Shared dataset selection options for loaders/executors."""
        from openjury.datasets.options import DatasetOptions
        return DatasetOptions(
            name=self.dataset,
            n_instructions=self.n_instructions,
            language=self.language,
            seed=self.seed,
            balance_by=self.balance_by,
        )

    # ── Loaders ──────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> AgreementConfig:
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
    def from_dict(cls, data: dict) -> AgreementConfig:
        """Construct from a raw dict (parsed JSON/YAML)."""
        judge = JudgeConfig.from_raw(data.get("judge", data.get("judge_model", "")))

        gen = data.get("generation", {})
        ignore_score_cache = gen.get(
            "ignore_score_cache",
            data.get("ignore_score_cache", False),
        )

        return cls(
            dataset=data["dataset"],
            judge=judge,
            output_dir=data.get("output_dir", "results/agreement"),
            n_instructions=data.get("n_instructions"),
            language=data.get("language"),
            seed=data.get("seed", 42),
            balance_by=data.get("balance_by"),
            criteria=data.get("criteria", data.get("rubric", "default")),
            ignore_score_cache=ignore_score_cache,
            truncate_instruction=data.get("truncate_instruction", 500),
            include_completions=data.get("include_completions", True),
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-serialisable dict."""
        return {
            "dataset": self.dataset,
            "n_instructions": self.n_instructions,
            "judge": self.judge.to_dict(),
            "criteria": self.criteria,
            "generation": {
                "ignore_score_cache": self.ignore_score_cache,
            },
            "output_dir": self.output_dir,
            "language": self.language,
            "seed": self.seed,
            "balance_by": self.balance_by,
            "include_completions": self.include_completions,
        }

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════
#  Matchmaker types
# ═════════════════════════════════════════════════════════════════════


@dataclass
class Match:
    """A scheduled pairwise matchup."""

    model_a: str
    model_b: str
    instruction_index: int


# ═════════════════════════════════════════════════════════════════════
#  Scoring results
# ═════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    """Samplewise criteria scores for one model on one instruction.

    Produced when the judge scores each model's completion independently.
    """

    model: str
    instruction_index: int
    scores: dict[str, float]
    sample_id: str = ""
    completion: str = ""
    raw_judge_output: str = ""


@dataclass
class MatchResult:
    """Result of a single pairwise matchup.

    ``preference``: 0.0 = model_a wins, 0.5 = tie, 1.0 = model_b wins.
    """

    model_a: str
    model_b: str
    instruction_index: int
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    preference: float
    instruction: str = ""
    instruction_id: str = ""
    instruction_metadata: dict[str, Any] = field(default_factory=dict)
    human_preference: float | None = None
    completion_a: str = ""
    completion_b: str = ""
    raw_judge_output: str = ""
    raw_judge_output_swapped: str | None = None
    # ── Per-swap position scores (before averaging) ────────────
    scores_a_original: dict[str, float] | None = None
    scores_b_original: dict[str, float] | None = None
    preference_original: float | None = None
    scores_a_swapped: dict[str, float] | None = None
    scores_b_swapped: dict[str, float] | None = None
    preference_swapped: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "instruction_index": self.instruction_index,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "preference": self.preference,
        }
        if self.instruction:
            d["instruction"] = self.instruction
        if self.instruction_id:
            d["instruction_id"] = self.instruction_id
        if self.instruction_metadata:
            d["instruction_metadata"] = self.instruction_metadata
        if self.human_preference is not None:
            d["human_preference"] = self.human_preference
        if self.completion_a:
            d["completion_a"] = self.completion_a
        if self.completion_b:
            d["completion_b"] = self.completion_b
        if self.raw_judge_output:
            d["raw_judge_output"] = self.raw_judge_output
        if self.raw_judge_output_swapped is not None:
            d["raw_judge_output_swapped"] = self.raw_judge_output_swapped
        # ── Per-swap position scores ─────────────────────────────
        if self.scores_a_original is not None:
            d["scores_a_original"] = self.scores_a_original
        if self.scores_b_original is not None:
            d["scores_b_original"] = self.scores_b_original
        if self.preference_original is not None:
            d["preference_original"] = self.preference_original
        if self.scores_a_swapped is not None:
            d["scores_a_swapped"] = self.scores_a_swapped
        if self.scores_b_swapped is not None:
            d["scores_b_swapped"] = self.scores_b_swapped
        if self.preference_swapped is not None:
            d["preference_swapped"] = self.preference_swapped
        return d


# ═════════════════════════════════════════════════════════════════════
#  Arena result (aggregate)
# ═════════════════════════════════════════════════════════════════════


@dataclass
class ArenaResult:
    """Complete arena evaluation result — the canonical output artifact.

    Contains everything needed to reproduce analysis: per-model samplewise
    scores, all pairwise match results, computed ratings, and metadata.
    """

    # ── Identity ─────────────────────────────────────────────────
    models: list[str]
    dataset: str
    judge_model: str
    judge_mode: str  # "samplewise" or "pairwise"
    matchmaker_strategy: str
    criteria_name: str
    criteria_definition: dict[str, Any]
    n_instructions: int

    # ── Per-instruction metadata (aligned with instruction indices) ──
    instruction_metadata: list[dict[str, Any]] = field(default_factory=list)

    # ── Per-model samplewise scores (populated in samplewise mode) ──
    model_scores: dict[str, list[ModelScore]] = field(default_factory=dict)

    # ── Pairwise match results ───────────────────────────────────
    matches: list[MatchResult] = field(default_factory=list)

    # ── Computed ratings ─────────────────────────────────────────
    bt_strengths: dict[str, float] = field(default_factory=dict)
    elo_ratings: dict[str, float] = field(default_factory=dict)
    win_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregate_win_rates: dict[str, float] = field(default_factory=dict)
    dimension_weights: dict[str, float] = field(default_factory=dict)
    dimension_weight_accuracy: float = 0.0

    # ── Extra metadata ───────────────────────────────────────────
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    # ── System prompt (for reproducibility / debugging) ──────────
    system_prompt: str = ""

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_models(self) -> int:
        return len(self.models)

    def leaderboard(self) -> list[tuple[str, float, float]]:
        """Return models sorted by BT strength (descending).

        Returns:
            List of ``(model_name, bt_strength, elo)`` tuples.
        """
        entries = []
        for m in self.models:
            entries.append((
                m,
                self.bt_strengths.get(m, 0.0),
                self.elo_ratings.get(m, 1500.0),
            ))
        entries.sort(key=lambda x: -x[1])
        return entries
