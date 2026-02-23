"""Factory function for creating model backends from a provider/model string.

This provides backward compatibility with the old ``make_model("Provider/model-name")``
API while using the new ModelRegistry under the hood.

It also supports passing a Pydantic config object or a dict for backend-specific settings.

Example::

    # Simple (backward-compatible):
    model = make_model("VLLM/meta-llama/Llama-3.1-8B-Instruct")

    # With config:
    from openjury.models.config import VLLMConfig
    model = make_model(
        "VLLM/meta-llama/Llama-3.1-8B-Instruct",
        config=VLLMConfig(tensor_parallel_size=4),
    )

    # With max_tokens shortcut:
    model = make_model("ChatOpenAI/gpt-4o", max_tokens=4096)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openjury.models.config import ModelConfig
from openjury.models.registry import ModelRegistry
from openjury._logging import logger

# Lightweight utilities (no CUDA / torch imports)
from openjury.models.utils import provider_from_model, is_local_provider  # noqa: F401

# Import backends so they auto-register via decorators
import openjury.models.backends  # noqa: F401

# ── Provider → Config class mapping ────────────────────────────────
# Providers not listed here fall back to the base ModelConfig, which
# every backend can accept (they upcast it internally).

_PROVIDER_CONFIG_MAP: dict[str, type[ModelConfig]] = {}


def _ensure_config_map():
    """Lazily populate the provider→config mapping to avoid circular imports."""
    if _PROVIDER_CONFIG_MAP:
        return
    from openjury.models.config import VLLMConfig, OpenAIConfig, LiteLLMConfig, LlamaCppConfig

    _PROVIDER_CONFIG_MAP.update({
        "VLLM": VLLMConfig,
        "ChatOpenAI": OpenAIConfig,
        "OpenRouter": OpenAIConfig,
        "LiteLLM": LiteLLMConfig,
        "LlamaCpp": LlamaCppConfig,
    })


def build_config_for_model(
    model: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    # VLLM-specific (ignored for API providers)
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    quantization: str | None = None,
    gpu_devices: str | None = None,
    chat_template: str | None = None,
    chat_template_file: str | None = None,
    # API-specific (ignored for local providers)
    api_base_url: str | None = None,
    api_key_env: str | None = None,
    # Pass-through for any backend
    generation_kwargs: dict[str, Any] | None = None,
    **extra,
) -> ModelConfig:
    """Build the correct config type for a model string.

    Inspects the ``Provider/`` prefix and constructs the matching config,
    silently skipping irrelevant kwargs. This lets CLI code pass all flags
    and have only the relevant ones applied.

    Args:
        model: Model string (e.g. ``"VLLM/Qwen/Qwen3-32B"``).
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p / nucleus sampling threshold.
        tensor_parallel_size: VLLM tensor parallelism.
        gpu_memory_utilization: VLLM GPU memory fraction.
        quantization: VLLM quantization method.
        gpu_devices: VLLM GPU device IDs.
        chat_template: Explicit vLLM Jinja chat template override.
        chat_template_file: Path to a file containing a vLLM chat template.
            Ignored if ``chat_template`` is provided.
        api_base_url: API base URL (OpenAI, LiteLLM).
        api_key_env: Environment variable for API key.
        generation_kwargs: Extra sampling/API kwargs passed through to the
            backend.  Merged into the config's ``generation_kwargs`` field.
        **extra: Additional provider-specific kwargs.

    Returns:
        A config instance of the appropriate type.
    """
    _ensure_config_map()
    provider = provider_from_model(model)
    config_cls = _PROVIDER_CONFIG_MAP.get(provider, ModelConfig)

    # Collect all non-None kwargs
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if generation_kwargs:
        kwargs["generation_kwargs"] = generation_kwargs

    # Map generic names → provider-specific field names
    if issubclass(config_cls, ModelConfig):
        from openjury.models.config import VLLMConfig, OpenAIConfig, LiteLLMConfig

        if issubclass(config_cls, VLLMConfig):
            if tensor_parallel_size is not None:
                kwargs["tensor_parallel_size"] = tensor_parallel_size
            if gpu_memory_utilization is not None:
                kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            if quantization is not None:
                kwargs["quantization"] = quantization
            if gpu_devices is not None:
                kwargs["gpu_devices"] = gpu_devices
            if chat_template is not None:
                kwargs["chat_template"] = chat_template
            elif chat_template_file is not None:
                kwargs["chat_template"] = Path(chat_template_file).read_text(
                    encoding="utf-8"
                )

        if issubclass(config_cls, OpenAIConfig):
            if api_base_url is not None:
                kwargs["base_url"] = api_base_url
            if api_key_env is not None:
                kwargs["api_key_env"] = api_key_env

        if issubclass(config_cls, LiteLLMConfig):
            if api_base_url is not None:
                kwargs["api_base"] = api_base_url
            if api_key_env is not None:
                kwargs["api_key_env"] = api_key_env

    # Only pass kwargs that the config class actually accepts
    valid_fields = set(config_cls.model_fields.keys())
    filtered = {k: v for k, v in {**kwargs, **extra}.items() if k in valid_fields}

    return config_cls(**filtered)


def make_model(
    model: str,
    max_tokens: int | None = None,
    config: ModelConfig | dict | None = None,
    chat_template: str | None = None,
    chat_template_file: str | None = None,
) -> Any:
    """Create a model backend from a 'Provider/model-name' string.

    Parses the provider prefix, creates the appropriate config, and
    instantiates the backend via the ModelRegistry.

    Args:
        model: Model string in format ``Provider/model-name``.
            Examples: ``VLLM/meta-llama/Llama-3.1-8B``, ``Dummy/test``,
            ``ChatOpenAI/gpt-4o``, ``LiteLLM/anthropic/claude-3-5-sonnet``.
        max_tokens: Maximum tokens to generate. Overrides config.max_tokens
            if provided.
        config: Optional backend-specific config. If a dict is passed, it
            will be used to construct the appropriate config object.
            If None, a default config is created with the given max_tokens.
        chat_template: Optional vLLM chat template override. Ignored by
            non-vLLM providers.
        chat_template_file: Optional file path containing a vLLM chat
            template. Ignored if ``chat_template`` is provided.

    Returns:
        A ModelBackend instance.

    Raises:
        ValueError: If provider is not registered.
    """
    parts = model.split("/", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Model string must be in format 'Provider/model-name', got: '{model}'"
        )

    provider = parts[0]
    model_name = parts[1]

    logger.info("Loading [model]%s[/model](model=%s)", provider, model_name)

    if provider == "VLLM" and (chat_template is not None or chat_template_file is not None):
        base_kwargs: dict[str, Any] = {}
        if isinstance(config, dict):
            base_kwargs.update(config)
        elif isinstance(config, ModelConfig):
            base_kwargs.update(config.model_dump())
        if max_tokens is not None:
            base_kwargs["max_tokens"] = max_tokens

        config = build_config_for_model(
            model,
            **base_kwargs,
            chat_template=chat_template,
            chat_template_file=chat_template_file,
        )
    elif isinstance(config, dict):
        if max_tokens is not None:
            config.setdefault("max_tokens", max_tokens)
        config = build_config_for_model(model, **config)

    # Build config: only override max_tokens when explicitly provided by caller.
    # This avoids clobbering provider-specific configs (e.g. Arena/JudgeConfig).
    if config is None:
        if max_tokens is not None:
            config = ModelConfig(max_tokens=max_tokens)
        else:
            config = ModelConfig()
    elif isinstance(config, ModelConfig) and max_tokens is not None:
        # Override max_tokens if explicitly passed
        config = config.model_copy(update={"max_tokens": max_tokens})

    return ModelRegistry.create(provider=provider, model_name=model_name, config=config)
