"""Pydantic configuration models for all model backends.

All model backends share a base ModelConfig. Backend-specific configs
extend it with provider-specific fields (e.g., tensor parallelism for VLLM).

Configs can be loaded from YAML/JSON or constructed in Python.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Base configuration shared by all model backends."""

    max_tokens: int = Field(default=8192, description="Maximum tokens to generate")
    temperature: float = Field(
        default=0.6, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling"
    )
    timeout: float = Field(
        default=120.0, description="Timeout in seconds for API calls"
    )
    generation_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra generation kwargs passed through to the backend. "
            "For VLLM: merged into SamplingParams (e.g. repetition_penalty, "
            "min_p, frequency_penalty). For API backends: merged into the "
            "API request body. Keys that duplicate explicit fields "
            "(temperature, top_p, max_tokens) are silently skipped."
        ),
    )


class VLLMConfig(ModelConfig):
    """Configuration for local VLLM inference.

    Supports multi-GPU setups via tensor_parallel_size and
    pipeline_parallel_size. All paths should point to locally cached
    model weights (no network access required at inference time).

    GPU pinning:
        Use ``gpu_devices`` to restrict which GPUs this model uses.
        E.g., ``gpu_devices="2,3"`` pins to GPU 2 and 3.
        Combined with ``tensor_parallel_size=2``, the model is sharded
        across those two GPUs. If None, uses all visible GPUs.
    """

    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism (split model layers across GPUs)",
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of pipeline parallel stages (split model sequentially across GPUs)",
    )
    gpu_devices: str | None = Field(
        default=None,
        description=(
            "Comma-separated GPU device IDs to pin this model to. "
            "E.g., '0,1' pins to GPU 0 and 1. None = use all visible GPUs. "
            "Useful for running different models on different GPUs simultaneously."
        ),
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory to use for model weights and KV cache",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Maximum sequence length the model can handle. None = use model's default.",
    )
    quantization: Literal["awq", "gptq", "squeezellm", "fp8"] | None = Field(
        default=None,
        description="Quantization method for the model weights",
    )
    top_k: int = Field(default=-1, description="Top-k sampling (-1 = disabled)")
    dtype: str = Field(
        default="auto",
        description="Data type for model weights (auto, float16, bfloat16, float32)",
    )
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code in model repository"
    )
    enforce_eager: bool = Field(
        default=False,
        description="Enforce eager execution instead of CUDA graphs",
    )
    enable_chunked_prefill: bool | None = Field(
        default=None,
        description="Enable chunked prefill for long prompts",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Enable thinking/reasoning mode for models that support it "
            "(e.g. Qwen3). Passed as ``enable_thinking`` to the tokenizer's "
            "``apply_chat_template``. None = let the model decide (default "
            "behaviour). Set ``True`` to force thinking on, ``False`` to "
            "force it off."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "Optional explicit Jinja chat template override for vLLM. "
            "If provided, forces ``LLM.chat()`` with this template even "
            "when the tokenizer does not define one."
        ),
    )


class SGLangConfig(ModelConfig):
    """Configuration for local SGLang inference.

    Mirrors the subset of ``sglang.Engine`` / ``ServerArgs`` fields that map
    cleanly onto the current framework abstractions.
    """

    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism (maps to SGLang tp_size)",
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of pipeline parallel stages (maps to SGLang pp_size)",
    )
    gpu_devices: str | None = Field(
        default=None,
        description=(
            "Comma-separated GPU device IDs to pin this model to. "
            "None = use all visible GPUs."
        ),
    )
    mem_fraction_static: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Static GPU memory fraction reserved by SGLang. "
            "Roughly analogous to vLLM's gpu_memory_utilization."
        ),
    )
    context_length: int | None = Field(
        default=None,
        description="Maximum context length passed to SGLang.",
    )
    quantization: str | None = Field(
        default=None,
        description="Quantization mode passed to SGLang.",
    )
    dtype: str = Field(
        default="auto",
        description="Data type for model weights (auto, float16, bfloat16, float32)",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code in model repository",
    )
    top_k: int = Field(default=-1, description="Top-k sampling (-1 = disabled)")
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Enable thinking/reasoning mode for models whose chat template "
            "supports it. Passed through tokenizer.apply_chat_template."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "Optional explicit Jinja chat template override. If provided, "
            "it is applied through the tokenizer before Engine.generate()."
        ),
    )


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI-compatible API backends.

    Works with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint.
    API keys are read from environment variables by default.
    """

    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name containing the API key",
    )
    base_url: str | None = Field(
        default=None,
        description="Override API base URL (for OpenRouter, local servers, etc.)",
    )
    organization: str | None = Field(
        default=None, description="OpenAI organization ID"
    )


class LiteLLMConfig(ModelConfig):
    """Configuration for LiteLLM unified backend.

    LiteLLM provides a unified interface for 100+ providers.
    Model names follow the format: provider/model-name.
    API keys are read from standard environment variables.

    Note: On networkless servers, this backend should only be used
    with locally-hosted endpoints (e.g., VLLM-served models exposed
    via an OpenAI-compatible API).
    """

    api_key_env: str | None = Field(
        default=None,
        description="Environment variable name for the API key",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL for local/self-hosted endpoints",
    )
    num_retries: int = Field(
        default=5,
        ge=0,
        description="Number of retries on transient failures",
    )
    fallbacks: list[str] | None = Field(
        default=None,
        description="List of fallback model names if primary model fails",
    )


class LlamaCppConfig(ModelConfig):
    """Configuration for llama.cpp local inference.

    For running GGUF quantized models on CPU or single GPU.
    Model path must point to a locally cached .gguf file.
    """

    n_gpu_layers: int = Field(
        default=-1,
        description="Number of layers to offload to GPU (-1 = all)",
    )
    n_ctx: int = Field(default=4096, description="Context window size")
    n_batch: int = Field(default=512, description="Batch size for prompt processing")
    verbose: bool = Field(default=False, description="Enable verbose llama.cpp output")


class InferenceConfig(BaseModel):
    """Configuration for the inference engine (retry logic, batching, etc.)."""

    max_retries: int = Field(
        default=5, ge=0, description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay in seconds for exponential backoff",
    )
    use_tqdm: bool = Field(
        default=False, description="Show progress bar during inference"
    )
    batch_size: int | None = Field(
        default=None,
        description="Batch size for inference. None = process all at once.",
    )
