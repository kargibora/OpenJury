"""Dummy model backend for testing.

Returns a fixed message for every input — useful for pipeline testing
without hitting real models or GPUs.
"""

from __future__ import annotations

from openjury.models.registry import ModelRegistry


@ModelRegistry.register("Dummy")
class DummyBackend:
    """Returns the model name suffix as output for every input.

    Model string format: ``Dummy/your message here``
    The part after "Dummy/" is returned as the completion.

    Example::

        model = DummyBackend(model_name="hello world")
        model.invoke("anything")  # -> "hello world"
    """

    def __init__(self, model_name: str, config=None):
        self.model_name = model_name
        self.message = model_name

    def batch(self, inputs: list, **kwargs) -> list[str]:
        return [self.message] * len(inputs)

    def invoke(self, input, **kwargs) -> str:
        return self.message

    async def ainvoke(self, input, **kwargs) -> str:
        return self.message
