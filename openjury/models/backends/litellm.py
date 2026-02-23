"""LiteLLM unified backend for 100+ LLM providers.

LiteLLM provides a single interface for OpenAI, Anthropic, Cohere,
HuggingFace, Together, and many more. It handles retries, rate limiting,
and response normalization automatically.

On networkless servers, use this with locally-hosted endpoints
(e.g., a VLLM server exposed via OpenAI-compatible API) by setting
``api_base`` in the config.

Example::

    from openjury.models.config import LiteLLMConfig

    # For a local VLLM server:
    config = LiteLLMConfig(api_base="http://localhost:8000/v1", max_tokens=4096)
    backend = LiteLLMBackend(model_name="openai/my-local-model", config=config)

    # For cloud APIs (requires network):
    config = LiteLLMConfig(max_tokens=4096)
    backend = LiteLLMBackend(model_name="anthropic/claude-3-5-sonnet", config=config)
"""

from __future__ import annotations

import os
from typing import Any

from openjury.env import load_dotenv
from openjury.models.config import LiteLLMConfig, ModelConfig
from openjury.models.registry import ModelRegistry

# Load .env so API keys are available (env vars already set win)
load_dotenv()


def _extract_text(response: Any) -> str:
    """Extract text from a LiteLLM response."""
    return response.choices[0].message.content


@ModelRegistry.register("LiteLLM")
class LiteLLMBackend:
    """Unified backend via LiteLLM for many LLM providers.

    Args:
        model_name: LiteLLM model string (e.g. "anthropic/claude-3-5-sonnet").
        config: LiteLLMConfig with API settings and retry configuration.
    """

    def __init__(self, model_name: str, config: LiteLLMConfig | ModelConfig | None = None):
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "LiteLLM is not installed. Install it with: pip install litellm"
            )

        if config is None:
            config = LiteLLMConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, LiteLLMConfig):
            config = LiteLLMConfig(**config.model_dump())

        self.model_name = model_name
        self.config = config

        # Set API key if specified
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if api_key:
                os.environ["LITELLM_API_KEY"] = api_key

    def _completion_kwargs(self) -> dict[str, Any]:
        """Build kwargs for litellm.completion / acompletion."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_retries": self.config.num_retries,
            "timeout": self.config.timeout,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        # Merge extra generation_kwargs (e.g. frequency_penalty, stop)
        _explicit = {"model", "max_tokens", "temperature", "top_p",
                     "num_retries", "timeout", "api_base"}
        for k, v in self.config.generation_kwargs.items():
            if k not in _explicit:
                kwargs[k] = v

        return kwargs

    @staticmethod
    def _to_messages(input_item: Any) -> list[dict]:
        """Convert input to OpenAI-style messages."""
        role_map = {"human": "user", "ai": "assistant", "system": "system"}

        if hasattr(input_item, "to_messages"):
            lc_messages = input_item.to_messages()
            return [
                {"role": role_map.get(msg.type, msg.type), "content": msg.content}
                for msg in lc_messages
            ]
        if isinstance(input_item, list) and input_item and isinstance(input_item[0], tuple):
            return [
                {"role": role_map.get(role, role), "content": content}
                for role, content in input_item
            ]
        if isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
            return input_item
        if isinstance(input_item, str):
            return [{"role": "user", "content": input_item}]
        raise ValueError(f"Unsupported input type: {type(input_item)}")

    def batch(self, inputs: list, **kwargs) -> list[str]:
        """Process a batch of inputs sequentially via LiteLLM.

        Args:
            inputs: List of inputs in any supported format.

        Returns:
            List of generated text strings.
        """
        import litellm

        completion_kwargs = self._completion_kwargs()
        results = []
        for inp in inputs:
            messages = self._to_messages(inp)
            response = litellm.completion(messages=messages, **completion_kwargs)
            results.append(_extract_text(response))
        return results

    def invoke(self, input, **kwargs) -> str:
        """Process a single input via LiteLLM.

        Args:
            input: Input in any supported format.

        Returns:
            Generated text string.
        """
        import litellm

        messages = self._to_messages(input)
        response = litellm.completion(messages=messages, **self._completion_kwargs())
        return _extract_text(response)

    async def ainvoke(self, input, **kwargs) -> str:
        """Async single input processing via LiteLLM.

        Args:
            input: Input in any supported format.

        Returns:
            Generated text string.
        """
        import litellm

        messages = self._to_messages(input)
        response = await litellm.acompletion(
            messages=messages, **self._completion_kwargs()
        )
        return _extract_text(response)
