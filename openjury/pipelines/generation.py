"""Generate completions for instructions using any model backend.

Uses the new model registry and inference engine. The model is created
via ``make_model()`` which auto-detects the backend from the provider prefix.

After inference, the model is cleaned up (GPU memory freed) so the next
model in the pipeline can reuse the same GPUs.
"""

from __future__ import annotations

import pandas as pd
from langchain.prompts import ChatPromptTemplate

from openjury.models.config import ModelConfig
from openjury.models.factory import make_model
from openjury.inference import do_inference


def _cleanup_model(model) -> None:
    """Free GPU memory if the model supports it (e.g. VLLMBackend)."""
    if hasattr(model, "cleanup"):
        model.cleanup()


def truncate(s: str, max_len: int | None = None):
    if max_len is not None:
        return s[:max_len]
    else:
        return s


def generate_instructions(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 4096,
    use_tqdm: bool = False,
    system_prompt: str | None = None,
    config: "ModelConfig | None" = None,
    force_async: bool = False,
) -> pd.DataFrame:
    chat_model = make_model(model, max_tokens=max_tokens, config=config)

    # TODO improve prompt to generate instructions
    if system_prompt is None:
        system_prompt = (
            "You are an helpful assistant that answer queries asked by users."
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": truncate(user_prompt, max_len=truncate_input_chars),
            }
            for user_prompt in instructions
        ]
    )

    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        force_async=force_async,
    )

    # Free GPU memory so the next model can load on the same GPUs
    _cleanup_model(chat_model)

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )
    return df_outputs


def generate_base(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 4096,
    use_tqdm: bool = False,
    config: "ModelConfig | None" = None,
) -> pd.DataFrame:
    model = make_model(model, max_tokens=max_tokens, config=config)

    inputs = [
        truncate(instruction, max_len=truncate_input_chars)
        for instruction in instructions
    ]

    completions = model.batch(
        inputs=inputs,
        max_tokens=max_tokens,
    )

    # Free GPU memory so the next model can load on the same GPUs
    _cleanup_model(model)

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )

    return df_outputs
