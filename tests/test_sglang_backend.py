from __future__ import annotations

import sys
import types

from openjury.models.backends.sglang import SGLangBackend
from openjury.models.config import SGLangConfig


def test_sglang_backend_renders_chat_templates_and_generates(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            rendered = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
            if kwargs.get("enable_thinking") is True:
                rendered += " | thinking=on"
            return rendered

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            captured["tokenizer_model_name"] = model_name
            captured["trust_remote_code"] = trust_remote_code
            return FakeTokenizer()

    class FakeEngine:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

        def generate(self, prompts, sampling_params=None):
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params
            return [{"text": f"OUT::{prompt}"} for prompt in prompts]

        def shutdown(self):
            captured["shutdown_called"] = True

    fake_sglang = types.ModuleType("sglang")
    fake_sglang.Engine = FakeEngine
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = SGLangBackend(
        "Qwen/Qwen3-8B",
        config=SGLangConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            mem_fraction_static=0.8,
            max_tokens=128,
            temperature=0.0,
            top_p=0.9,
            top_k=20,
            enable_thinking=True,
        ),
    )

    outputs = backend.batch(
        [
            [("system", "Judge"), ("user", "Hello")],
            "Fallback prompt",
        ]
    )

    assert captured["engine_kwargs"] == {
        "model_path": "Qwen/Qwen3-8B",
        "tp_size": 2,
        "pp_size": 1,
        "dtype": "auto",
        "trust_remote_code": True,
        "mem_fraction_static": 0.8,
    }
    assert captured["sampling_params"] == {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 20,
    }
    assert captured["prompts"] == [
        "system:Judge | user:Hello | thinking=on",
        "user:Fallback prompt | thinking=on",
    ]
    assert outputs == [
        "OUT::system:Judge | user:Hello | thinking=on",
        "OUT::user:Fallback prompt | thinking=on",
    ]

    backend.cleanup()
    assert captured["shutdown_called"] is True
