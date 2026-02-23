"""LlamaCpp backend for local GGUF model inference.

Runs quantized models via llama.cpp on CPU or single GPU.
Model path must point to a locally cached .gguf file.

Example::

    from openjury.models.config import LlamaCppConfig

    config = LlamaCppConfig(n_gpu_layers=-1, n_ctx=4096, max_tokens=2048)
    backend = LlamaCppBackend(model_name="/path/to/model.gguf", config=config)
    output = backend.invoke("What is 2+2?")
"""

from __future__ import annotations

from typing import Any

from openjury.models.config import LlamaCppConfig, ModelConfig
from openjury.models.registry import ModelRegistry


def _langchain_to_str(result: Any) -> str:
    """Extract text content from a LangChain response object."""
    if hasattr(result, "content"):
        return result.content
    return str(result)


@ModelRegistry.register("LlamaCpp")
class LlamaCppBackend:
    """Backend for llama.cpp via LangChain's LlamaCpp wrapper.

    Args:
        model_name: Path to the local .gguf model file.
        config: LlamaCppConfig with GPU/context settings.
    """

    def __init__(self, model_name: str, config: LlamaCppConfig | ModelConfig | None = None):
        from langchain_community.llms import LlamaCpp

        if config is None:
            config = LlamaCppConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, LlamaCppConfig):
            config = LlamaCppConfig(**config.model_dump())

        self.model_name = model_name
        self.config = config

        self._model = LlamaCpp(
            model_path=model_name,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            verbose=config.verbose,
        )

    def batch(self, inputs: list, **kwargs) -> list[str]:
        results = self._model.batch(inputs, **kwargs)
        return [_langchain_to_str(r) for r in results]

    def invoke(self, input, **kwargs) -> str:
        result = self._model.invoke(input, **kwargs)
        return _langchain_to_str(result)

    async def ainvoke(self, input, **kwargs) -> str:
        result = await self._model.ainvoke(input, **kwargs)
        return _langchain_to_str(result)
