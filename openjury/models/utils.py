"""Lightweight model string utilities — no heavy imports.

This module intentionally avoids importing anything that triggers CUDA,
torch, or vLLM initialization so that it can be safely used on login
nodes (e.g. for SLURM script generation).
"""

from __future__ import annotations

# Providers that run locally and need GPUs
_LOCAL_PROVIDERS = frozenset({"VLLM", "LlamaCpp"})

# Providers that call remote APIs (require network access)
_NETWORK_PROVIDERS = frozenset({"ChatOpenAI", "OpenRouter", "LiteLLM"})

# Provider → default env-var name for the API key
_API_KEY_ENV: dict[str, str] = {
    "ChatOpenAI": "OPENAI_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "LiteLLM": "LITELLM_API_KEY",
}


def provider_from_model(model: str) -> str:
    """Extract the provider prefix from a ``'Provider/model-name'`` string.

    Example::

        >>> provider_from_model("VLLM/Qwen/Qwen2.5-32B-Instruct")
        'VLLM'
        >>> provider_from_model("OpenRouter/deepseek/deepseek-chat-v3.1")
        'OpenRouter'
    """
    parts = model.split("/", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Model string must be in format 'Provider/model-name', got: '{model}'"
        )
    return parts[0]


def is_local_provider(provider: str) -> bool:
    """Return True if the provider runs locally on GPU (VLLM, LlamaCpp).

    Useful for deciding whether to request GPUs in SLURM scripts, or
    whether to skip VLLM-specific flags when building configs.

    Local providers: VLLM, LlamaCpp.
    API providers: ChatOpenAI, OpenRouter, LiteLLM, Dummy — no GPUs needed.
    """
    return provider in _LOCAL_PROVIDERS


def needs_network(model: str) -> bool:
    """Return ``True`` if *model* requires network access (API provider).

    On HPC clusters where compute nodes are air-gapped, jobs for these
    models must run on login nodes (or a partition with internet).

    Example::

        >>> needs_network("VLLM/Qwen/Qwen2.5-32B-Instruct")
        False
        >>> needs_network("OpenRouter/deepseek/deepseek-chat-v3.1")
        True
        >>> needs_network("Dummy/test")
        False
    """
    return provider_from_model(model) in _NETWORK_PROVIDERS


def api_key_env_for_model(model: str) -> str | None:
    """Return the default env-var name for the API key, or ``None``.

    Example::

        >>> api_key_env_for_model("OpenRouter/deepseek/deepseek-chat-v3.1")
        'OPENROUTER_API_KEY'
        >>> api_key_env_for_model("VLLM/Qwen/Qwen2.5-32B-Instruct")
    """
    return _API_KEY_ENV.get(provider_from_model(model))
