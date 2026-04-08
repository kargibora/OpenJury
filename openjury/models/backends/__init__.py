"""Backend implementations for all supported model providers.

Importing this module auto-registers every backend into the ModelRegistry.
"""

# Import all backends so their @ModelRegistry.register decorators run
from openjury.models.backends.dummy import DummyBackend  # noqa: F401
from openjury.models.backends.vllm import VLLMBackend  # noqa: F401
from openjury.models.backends.openai import OpenAIBackend  # noqa: F401
from openjury.models.backends.openai import OpenRouterBackend  # noqa: F401
from openjury.models.backends.litellm import LiteLLMBackend  # noqa: F401
from openjury.models.backends.llamacpp import LlamaCppBackend  # noqa: F401
from openjury.models.backends.sglang import SGLangBackend  # noqa: F401

__all__ = [
    "DummyBackend",
    "VLLMBackend",
    "SGLangBackend",
    "OpenAIBackend",
    "OpenRouterBackend",
    "LiteLLMBackend",
    "LlamaCppBackend",
]
