"""VLLM backend for local GPU inference with proper chat template handling.

This backend uses ``vllm.LLM.chat()`` (not ``generate()``) so that the
model's chat template is applied correctly. Supports multi-GPU via
tensor parallelism and pipeline parallelism.

All model weights must be pre-cached locally — no network access is
required at inference time.

Example::

    from openjury.models.config import VLLMConfig

    config = VLLMConfig(tensor_parallel_size=4, max_tokens=4096)
    backend = VLLMBackend(model_name="meta-llama/Llama-3.1-8B-Instruct", config=config)
    outputs = backend.batch(["What is 2+2?", "Tell me a joke"])
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

from openjury._logging import logger
from openjury.models.config import VLLMConfig, ModelConfig
from openjury.models.registry import ModelRegistry


def _to_messages(input_item: Any) -> list[dict]:
    """Convert various input formats to OpenAI-style message dicts.

    Supported formats:
        - LangChain ``ChatPromptValue`` (has ``to_messages()``)
        - List of tuples: ``[("system", "..."), ("user", "...")]``
        - List of dicts: ``[{"role": "user", "content": "..."}]``
        - Plain string: wrapped as ``[{"role": "user", "content": "..."}]``

    Args:
        input_item: Input in any of the supported formats.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    role_map = {"human": "user", "ai": "assistant", "system": "system"}

    # LangChain ChatPromptValue
    if hasattr(input_item, "to_messages"):
        lc_messages = input_item.to_messages()
        return [
            {"role": role_map.get(msg.type, msg.type), "content": msg.content}
            for msg in lc_messages
        ]

    # List of tuples: [("system", "..."), ("user", "...")]
    if isinstance(input_item, list) and input_item and isinstance(input_item[0], tuple):
        return [
            {"role": role_map.get(role, role), "content": content}
            for role, content in input_item
        ]

    # Already formatted message dicts
    if isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
        return input_item

    # Plain string → user message
    if isinstance(input_item, str):
        return [{"role": "user", "content": input_item}]

    raise ValueError(f"Unsupported input type: {type(input_item)}")


def _to_raw_text(input_item: Any) -> str:
    """Extract raw text for ``llm.generate()`` fallback (base models)."""
    if isinstance(input_item, str):
        return input_item
    if hasattr(input_item, "to_string"):
        return input_item.to_string()
    if isinstance(input_item, list) and input_item:
        if isinstance(input_item[0], dict):
            return "\n".join(str(msg.get("content", "")) for msg in input_item)
        if isinstance(input_item[0], tuple):
            return "\n".join(str(content) for _, content in input_item)
    raise ValueError(f"Cannot extract raw text from: {type(input_item)}")


@ModelRegistry.register("VLLM")
class VLLMBackend:
    """VLLM backend with chat-template auto-detection and fallback support.

    Uses ``LLM.chat()`` when a chat template is available (tokenizer-defined
    or explicitly supplied via ``config.chat_template``). Falls back to
    ``LLM.generate()`` for base/pretrained models that do not define a chat
    template.

    Args:
        model_name: HuggingFace model ID or local path to cached weights.
        config: VLLMConfig with GPU/sampling settings.
    """

    def __init__(self, model_name: str, config: VLLMConfig | ModelConfig | None = None):
        # Import vllm here to avoid hard dependency for users who don't use this backend.
        import os
        from vllm import LLM, SamplingParams

        if config is None:
            config = VLLMConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, VLLMConfig):
            # Upgrade base config to VLLMConfig preserving shared fields
            config = VLLMConfig(**config.model_dump())

        self.model_name = model_name
        self.config = config

        # Pin to specific GPUs if requested (e.g., gpu_devices="0,1")
        #
        # HPC / SLURM note:
        #   SLURM already sets CUDA_VISIBLE_DEVICES to the allocated GPUs.
        #   For example, --gres=gpu:4 might set CUDA_VISIBLE_DEVICES="0,1,2,3".
        #   Our gpu_devices values are indices *within* SLURM's allocation,
        #   so --model_A_gpus "0,1" means "use the first 2 GPUs that SLURM
        #   gave us", NOT physical GPU IDs.
        #
        #   When no gpu_devices is set, VLLM sees all SLURM-allocated GPUs
        #   and uses tensor_parallel_size of them.
        self._original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if config.gpu_devices is not None:
            # If SLURM already set CUDA_VISIBLE_DEVICES, subset it
            slurm_devices = self._original_cuda_devices
            if slurm_devices is not None:
                slurm_list = slurm_devices.split(",")
                requested = [int(d.strip()) for d in config.gpu_devices.split(",")]
                # Map logical indices to SLURM-allocated physical IDs
                try:
                    mapped = [slurm_list[i] for i in requested]
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapped)
                except IndexError:
                    logger.warning(
                        "gpu_devices=%s requests indices beyond SLURM allocation "
                        "(%s). Using gpu_devices as-is.",
                        config.gpu_devices, slurm_devices,
                    )
                    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices

        vllm_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "enforce_eager": config.enforce_eager,
        }

        if config.max_model_len is not None:
            vllm_kwargs["max_model_len"] = config.max_model_len
        if config.quantization is not None:
            vllm_kwargs["quantization"] = config.quantization
        if config.seed is not None:
            vllm_kwargs["seed"] = config.seed
        if config.enable_chunked_prefill is not None:
            vllm_kwargs["enable_chunked_prefill"] = config.enable_chunked_prefill

        self.llm = LLM(model=model_name, **vllm_kwargs)

        sampling_kwargs: dict[str, Any] = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.top_k != -1:
            sampling_kwargs["top_k"] = config.top_k

        # Merge extra generation_kwargs (e.g. repetition_penalty, min_p)
        # Skip keys already set explicitly above to avoid conflicts.
        _explicit = {"max_tokens", "temperature", "top_p", "top_k"}
        for k, v in config.generation_kwargs.items():
            if k not in _explicit:
                sampling_kwargs[k] = v

        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Chat-template mode selection:
        # 1) explicit config.chat_template → force chat() with that template
        # 2) tokenizer-provided template → chat()
        # 3) no template → generate() fallback (base/pretrained models)
        self.chat_template = config.chat_template
        if self.chat_template:
            self._use_generate = False
            logger.info(
                "VLLM using explicit chat template override for [model]%s[/model]",
                model_name,
            )
        else:
            tokenizer = self.llm.get_tokenizer()
            if not getattr(tokenizer, "chat_template", None):
                warnings.warn(
                    f"Model '{model_name}' tokenizer does not define a chat template. "
                    "Falling back to llm.generate() (no chat formatting). "
                    "Provide VLLMConfig(chat_template=...) to force chat mode.",
                )
                self._use_generate = True
            else:
                self._use_generate = False
                logger.info(
                    "VLLM using tokenizer chat template for [model]%s[/model]",
                    model_name,
                )

        # Chat template kwargs (e.g. enable_thinking for Qwen3)
        self.chat_template_kwargs: dict[str, Any] | None = None
        if config.enable_thinking is not None:
            self.chat_template_kwargs = {
                "enable_thinking": config.enable_thinking,
            }

    def batch(self, inputs: list, **kwargs) -> list[str]:
        """Process a batch of inputs using ``chat()`` or ``generate()``.

        Args:
            inputs: List of inputs in any supported format.

        Returns:
            List of generated text strings.
        """
        if self._use_generate:
            prompts = [_to_raw_text(inp) for inp in inputs]
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            messages_batch = [_to_messages(inp) for inp in inputs]
            chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
            }
            if self.chat_template is not None:
                chat_kwargs["chat_template"] = self.chat_template
            if self.chat_template_kwargs is not None:
                chat_kwargs["chat_template_kwargs"] = self.chat_template_kwargs

            outputs = self.llm.chat(
                messages_batch,
                self.sampling_params,
                **chat_kwargs,
            )
        return [out.outputs[0].text for out in outputs]

    def invoke(self, input_item, **kwargs) -> str:
        """Process a single input.

        Args:
            input_item: Input in any supported format.

        Returns:
            Generated text string.
        """
        return self.batch([input_item], **kwargs)[0]

    async def ainvoke(self, input_item, **kwargs) -> str:
        """Async wrapper — runs sync invoke in a thread pool.

        VLLM doesn't have native async support, so we run it in an executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke(input_item, **kwargs)
        )

    def cleanup(self) -> None:
        """Release GPU memory so the next model can load on the same GPUs.

        VLLM holds onto GPU memory through multiple mechanisms:
            - The model weights on GPU
            - The KV cache pre-allocated blocks
            - NCCL communicators (for tensor parallelism)
            - Ray worker processes (for multi-GPU)
            - PyTorch CUDA context and cached allocations

        A simple ``del self.llm`` is NOT enough — the distributed state
        and CUDA context remain. This method does a thorough cleanup
        suitable for HPC sequential pipelines.

        Steps:
            1. Tear down VLLM's distributed/parallel state (NCCL, Ray).
            2. Delete the LLM engine and sampling params.
            3. Run aggressive garbage collection.
            4. Clear the PyTorch CUDA cache and reset peak stats.
            5. Restore ``CUDA_VISIBLE_DEVICES`` to the original value.
        """
        import gc
        import os

        logger.info("Cleaning up VLLM model [model]%s[/model]", self.model_name)

        # 1. Tear down VLLM's distributed state (NCCL communicators, Ray)
        #    This is essential for multi-GPU (tensor_parallel > 1).
        #    Without this, the next LLM() call hangs waiting for stale NCCL groups.
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
            logger.debug("VLLM distributed state destroyed")
        except ImportError:
            # Older vllm versions — try the legacy path
            try:
                from vllm.model_executor.parallel_utils.parallel_state import (
                    destroy_model_parallel,
                )
                destroy_model_parallel()
            except ImportError:
                logger.debug("VLLM destroy_model_parallel not available, skipping")

        # 2. Delete the engine (releases references to GPU tensors)
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "sampling_params"):
            del self.sampling_params

        # 3. Aggressive garbage collection
        #    Run multiple rounds to break reference cycles.
        gc.collect()
        gc.collect()

        # 4. Free CUDA cache and reset memory stats
        try:
            import torch
            import torch.distributed as dist

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory tracking for clean reporting of next model
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                logger.debug("CUDA cache cleared across %d devices", torch.cuda.device_count())

            # Tear down PyTorch distributed process group if it exists
            # (VLLM may have initialized it for NCCL communication)
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.debug("PyTorch distributed process group destroyed")
        except ImportError:
            pass

        # 5. Shut down Ray if VLLM started it (multi-GPU with Ray backend)
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.debug("Ray runtime shut down")
        except ImportError:
            pass

        # 6. Restore original CUDA_VISIBLE_DEVICES
        if self._original_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_devices
        elif "CUDA_VISIBLE_DEVICES" in os.environ and self.config.gpu_devices is not None:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        logger.info("GPU memory released")
