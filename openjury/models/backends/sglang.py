"""SGLang backend for local GPU inference.

This backend uses the offline ``sglang.Engine`` API and manually renders chat
prompts with a HuggingFace tokenizer before calling ``Engine.generate()``.
The dependency is optional and imported lazily at runtime.
"""

from __future__ import annotations

import asyncio
from typing import Any

from openjury._logging import logger
from openjury.models.config import ModelConfig, SGLangConfig
from openjury.models.registry import ModelRegistry


def _to_messages(input_item: Any) -> list[dict]:
    """Convert various input formats to OpenAI-style message dicts."""
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


def _to_raw_text(input_item: Any) -> str:
    """Extract raw text for ``Engine.generate()`` fallback."""
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


@ModelRegistry.register("SGLang")
class SGLangBackend:
    """Local SGLang backend using the offline ``Engine`` API."""

    def __init__(self, model_name: str, config: SGLangConfig | ModelConfig | None = None):
        import gc
        import os

        import sglang as sgl
        from transformers import AutoTokenizer

        if config is None:
            config = SGLangConfig()
        elif isinstance(config, ModelConfig) and not isinstance(config, SGLangConfig):
            config = SGLangConfig(**config.model_dump())

        self.model_name = model_name
        self.config = config
        self._gc = gc
        self._original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        if config.gpu_devices is not None:
            slurm_devices = self._original_cuda_devices
            if slurm_devices is not None:
                slurm_list = slurm_devices.split(",")
                requested = [int(d.strip()) for d in config.gpu_devices.split(",")]
                try:
                    mapped = [slurm_list[i] for i in requested]
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapped)
                except IndexError:
                    logger.warning(
                        "gpu_devices=%s requests indices beyond SLURM allocation "
                        "(%s). Using gpu_devices as-is.",
                        config.gpu_devices,
                        slurm_devices,
                    )
                    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices

        engine_kwargs: dict[str, Any] = {
            "model_path": model_name,
            "tp_size": config.tensor_parallel_size,
            "pp_size": config.pipeline_parallel_size,
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.mem_fraction_static is not None:
            engine_kwargs["mem_fraction_static"] = config.mem_fraction_static
        if config.context_length is not None:
            engine_kwargs["context_length"] = config.context_length
        if config.quantization is not None:
            engine_kwargs["quantization"] = config.quantization
        if config.chat_template is not None:
            engine_kwargs["chat_template"] = config.chat_template

        self.llm = sgl.Engine(**engine_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if config.chat_template is not None:
            self.tokenizer.chat_template = config.chat_template

        self._use_generate = not bool(getattr(self.tokenizer, "chat_template", None))
        if self._use_generate:
            logger.warning(
                "SGLang tokenizer for [model]%s[/model] has no chat template. "
                "Falling back to raw Engine.generate() prompts.",
                model_name,
            )

        self.sampling_params: dict[str, Any] = {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.top_k != -1:
            self.sampling_params["top_k"] = config.top_k

        for key, value in config.generation_kwargs.items():
            if key not in {"max_new_tokens", "temperature", "top_p", "top_k"}:
                self.sampling_params[key] = value

        self.chat_template_kwargs: dict[str, Any] = {}
        if config.enable_thinking is not None:
            self.chat_template_kwargs["enable_thinking"] = config.enable_thinking

    def _render_prompt(self, input_item: Any) -> str:
        if self._use_generate:
            return _to_raw_text(input_item)
        messages = _to_messages(input_item)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    @staticmethod
    def _extract_text(output: Any) -> str:
        if isinstance(output, dict):
            if "text" in output:
                return str(output["text"])
            if "output_text" in output:
                return str(output["output_text"])
        if hasattr(output, "text"):
            return str(output.text)
        return str(output)

    def batch(self, inputs: list[Any], **kwargs) -> list[str]:
        prompts = [self._render_prompt(inp) for inp in inputs]
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return [self._extract_text(out) for out in outputs]

    def invoke(self, input_item: Any, **kwargs) -> str:
        return self.batch([input_item], **kwargs)[0]

    async def ainvoke(self, input_item: Any, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input_item, **kwargs))

    def cleanup(self) -> None:
        import os

        logger.info("Cleaning up SGLang model [model]%s[/model]", self.model_name)

        if hasattr(self, "llm"):
            try:
                self.llm.shutdown()
            except Exception as exc:
                logger.warning("SGLang shutdown failed for %s: %s", self.model_name, exc)
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        self._gc.collect()
        self._gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        except ImportError:
            pass

        if self._original_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_devices
        elif "CUDA_VISIBLE_DEVICES" in os.environ and self.config.gpu_devices is not None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
