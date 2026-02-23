"""OpenAI-compatible backend (ChatOpenAI, OpenRouter).

Wraps LangChain's ChatOpenAI to work with the ModelBackend protocol.
Also provides an OpenRouter variant that sets the correct base URL and
API key automatically.

API keys are read from environment variables — no hardcoded secrets.
"""

from __future__ import annotations

import os
from typing import Any

from openjury.env import load_dotenv
from openjury.models.config import OpenAIConfig, ModelConfig
from openjury.models.registry import ModelRegistry

# Load .env so API keys are available (env vars already set win)
load_dotenv()


def _langchain_to_str(result: Any) -> str:
    """Extract text content from a LangChain response object."""
    if hasattr(result, "content"):
        return result.content
    return str(result)


@ModelRegistry.register("ChatOpenAI")
class OpenAIBackend:
    """Backend for OpenAI and OpenAI-compatible APIs via LangChain.

    Args:
        model_name: OpenAI model name (e.g. "gpt-4o", "gpt-4o-mini").
        config: OpenAIConfig with API key env var, base URL, etc.
    """

    def __init__(self, model_name: str, config: OpenAIConfig | ModelConfig | None = None):
        from langchain_openai import ChatOpenAI
        from openjury._logging import logger

        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, OpenAIConfig):
            config = OpenAIConfig(**config.model_dump())

        self.model_name = model_name
        self.config = config

        if config.top_p != 0.95:
            logger.warning(
                "OpenAI/OpenRouter backend ignores top_p in this path (got top_p=%s)",
                config.top_p,
            )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
        }

        # Merge extra generation_kwargs.
        # `top_p` is intentionally not forwarded for this backend path to keep
        # behaviour consistent with the explicit config fields used here.
        _explicit = {"model", "max_tokens", "temperature", "timeout", "top_p"}
        for k, v in config.generation_kwargs.items():
            if k not in _explicit:
                kwargs[k] = v

        api_key = os.environ.get(config.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key

        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.organization:
            kwargs["organization"] = config.organization

        self._model = ChatOpenAI(**kwargs)

    def batch(self, inputs: list, **kwargs) -> list[str]:
        results = self._model.batch(inputs, **kwargs)
        return [_langchain_to_str(r) for r in results]

    def invoke(self, input, **kwargs) -> str:
        result = self._model.invoke(input, **kwargs)
        return _langchain_to_str(result)

    async def ainvoke(self, input, **kwargs) -> str:
        result = await self._model.ainvoke(input, **kwargs)
        return _langchain_to_str(result)


@ModelRegistry.register("OpenRouter")
class OpenRouterBackend(OpenAIBackend):
    """Backend for OpenRouter API (OpenAI-compatible with different base URL).

    Reads the API key from ``OPENROUTER_API_KEY`` environment variable.
    """

    def __init__(self, model_name: str, config: OpenAIConfig | ModelConfig | None = None):
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, OpenAIConfig):
            config = OpenAIConfig(**config.model_dump())

        # Override base URL and API key env for OpenRouter
        config = config.model_copy(
            update={
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            }
        )
        super().__init__(model_name=model_name, config=config)
