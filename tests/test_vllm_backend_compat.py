from __future__ import annotations

import sys
import types

from openjury.models.backends.vllm import VLLMBackend
from openjury.models.config import VLLMConfig


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.top_k = kwargs.get("top_k", -1)


class _FakeOutput:
    def __init__(self, texts):
        self.outputs = [types.SimpleNamespace(text=text) for text in texts]


class _FakeTokenizer:
    chat_template = "{{ messages }}"


class _FakeLLM:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.chat_calls = []
        self.generate_calls = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    def chat(self, messages_batch, sampling_params, **kwargs):
        self.chat_calls.append((messages_batch, sampling_params, kwargs))
        n = getattr(sampling_params, "n", 1)
        return [_FakeOutput([f"chat-{idx}-{i}" for i in range(n)]) for idx, _ in enumerate(messages_batch)]

    def generate(self, prompts, sampling_params):
        self.generate_calls.append((prompts, sampling_params))
        n = getattr(sampling_params, "n", 1)
        return [_FakeOutput([f"generate-{idx}-{i}" for i in range(n)]) for idx, _ in enumerate(prompts)]


def test_vllm_backend_matches_v019_call_surface(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    backend = VLLMBackend(
        "google/gemma-4-E4B-it",
        config=VLLMConfig(
            max_tokens=64,
            tensor_parallel_size=2,
            enable_thinking=False,
            generation_kwargs={"min_p": 0.1},
        ),
    )

    assert backend.llm.kwargs["tensor_parallel_size"] == 2
    assert backend.sampling_params.max_tokens == 64
    assert backend.sampling_params.min_p == 0.1

    outputs = backend.batch(["hello"])
    assert outputs == ["chat-0-0"]

    messages_batch, sampling_params, chat_kwargs = backend.llm.chat_calls[0]
    assert messages_batch == [[{"role": "user", "content": "hello"}]]
    assert sampling_params.max_tokens == 64
    assert chat_kwargs["add_generation_prompt"] is True
    assert chat_kwargs["chat_template_kwargs"] == {"enable_thinking": False}

    multi_outputs = backend.batch_multi(["hello"], n=2)
    assert multi_outputs == [["chat-0-0", "chat-0-1"]]
