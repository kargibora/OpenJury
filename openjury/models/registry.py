"""Model backend registry.

Provides a decorator-based pattern to register model backends by name.
New backends can be added without modifying existing code — just decorate
the class and it becomes available via ``ModelRegistry.create()``.

Example::

    @ModelRegistry.register("MyBackend")
    class MyBackend:
        def __init__(self, model_name: str, config: ModelConfig): ...
        def batch(self, inputs, **kw) -> list[str]: ...
        def invoke(self, inp, **kw) -> str: ...
        async def ainvoke(self, inp, **kw) -> str: ...

    model = ModelRegistry.create("MyBackend", "model-name", config)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Interface that every model backend must satisfy.

    All backends must implement batch/invoke/ainvoke so they can be used
    interchangeably by the inference engine.
    """

    def batch(self, inputs: list[Any], **kwargs) -> list[str]:
        """Process a batch of inputs and return a list of string outputs."""
        ...

    def invoke(self, input: Any, **kwargs) -> str:
        """Process a single input and return a string output."""
        ...

    async def ainvoke(self, input: Any, **kwargs) -> str:
        """Async version of invoke."""
        ...


class ModelRegistry:
    """Central registry mapping provider names to backend classes.

    Usage::

        # Register a backend (typically via decorator)
        @ModelRegistry.register("VLLM")
        class VLLMBackend: ...

        # Create an instance
        model = ModelRegistry.create("VLLM", "meta-llama/Llama-3.1-8B", config)

        # List available providers
        ModelRegistry.list_providers()  # -> ["VLLM", "Dummy", ...]
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model backend class under a provider name.

        Args:
            name: Provider name used in model strings (e.g. "VLLM", "OpenAI").
        """

        def decorator(model_class: type) -> type:
            if name in cls._registry:
                raise ValueError(
                    f"Provider '{name}' is already registered to {cls._registry[name].__name__}. "
                    f"Cannot re-register to {model_class.__name__}."
                )
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, provider: str, model_name: str, config: Any = None) -> ModelBackend:
        """Instantiate a model backend by provider name.

        Args:
            provider: Registered provider name (e.g. "VLLM", "Dummy").
            model_name: Model identifier (e.g. "meta-llama/Llama-3.1-8B").
            config: Backend-specific configuration (Pydantic model or dict).

        Returns:
            An instance satisfying the ModelBackend protocol.

        Raises:
            ValueError: If provider is not registered.
        """
        if provider not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {available}"
            )
        backend_cls = cls._registry[provider]
        if config is None:
            return backend_cls(model_name=model_name)
        return backend_cls(model_name=model_name, config=config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a list of all registered provider names."""
        return list(cls._registry.keys())

    @classmethod
    def _reset(cls):
        """Clear the registry. Only used in tests."""
        cls._registry.clear()
