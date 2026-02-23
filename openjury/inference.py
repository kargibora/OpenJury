"""Unified inference engine with retry logic, progress tracking, and batching.

Works with any ModelBackend. Supports two execution strategies:

- **Batch mode** (default): Sends all inputs to ``chat_model.batch()`` in one
  call, letting the engine (e.g. vLLM) handle internal scheduling. This is
  dramatically faster for local models.
- **Async mode** (``force_async=True``): Fires concurrent ``ainvoke()`` calls
  via ``asyncio.gather``. Best for API-based models where each request is an
  independent HTTP call.

Both modes support progress bars (``use_tqdm``) and exponential-backoff retry
on transient errors.

Example::

    from openjury.models.factory import make_model
    from openjury.inference import do_inference

    # Local model — uses batch() internally
    model = make_model("VLLM/meta-llama/Llama-3.1-8B-Instruct")
    outputs = do_inference(model, ["Hello!", "What is 2+2?"])

    # API model — uses async ainvoke() for concurrency
    model = make_model("OpenRouter/deepseek/deepseek-chat-v3.1")
    outputs = do_inference(model, prompts, force_async=True, use_tqdm=True)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm as async_tqdm

from openjury._logging import logger


def _is_retryable_error(e: Exception) -> bool:
    """Check if an error is transient and worth retrying.

    Catches rate-limit (429) and server errors (500, 502, 503).
    """
    error_str = str(e).lower()
    return any(
        code in error_str
        for code in ["429", "500", "502", "503", "rate", "overloaded", "timeout"]
    )


def do_inference(
    chat_model: Any,
    inputs: list[Any],
    use_tqdm: bool = False,
    max_retries: int = 5,
    base_delay: float = 1.0,
    batch_size: int | None = None,
    force_async: bool = False,
) -> list[str]:
    """Run inference on a list of inputs with automatic retry logic.

    **Path selection:**

    - ``force_async=True``: Uses ``ainvoke()`` with ``asyncio.gather`` for
      concurrent requests. Best for **API-based models** (OpenAI, OpenRouter,
      LiteLLM) where each request is an independent HTTP call.
    - Default (``force_async=False``): Uses ``batch()`` which sends all inputs
      at once to the engine. Best for **local models** (VLLM, LlamaCpp) where
      the engine handles internal batching and scheduling.

    Both paths support ``use_tqdm`` progress bars and exponential backoff
    on transient errors.

    Args:
        chat_model: Any object implementing the ModelBackend protocol
            (batch, invoke, ainvoke methods).
        inputs: List of inputs to process.
        use_tqdm: If True, show a progress bar.
        max_retries: Maximum number of retry attempts per item/batch.
        base_delay: Base delay in seconds for exponential backoff.
        batch_size: Max items per ``batch()`` call. None = all at once.
            Useful for memory-constrained environments. Ignored in async mode.
        force_async: If True, use async ``ainvoke()`` path instead of
            ``batch()``. Set this for API-based models only.

    Returns:
        List of string outputs, in the same order as inputs.
    """
    if force_async:
        results = _async_inference(
            chat_model, inputs, max_retries, base_delay, use_tqdm,
        )
    else:
        results = _batch_inference(
            chat_model, inputs, max_retries, base_delay, use_tqdm, batch_size,
        )

    # Normalize: LangChain sometimes returns AIMessage objects instead of strings
    return [x.content if hasattr(x, "content") else str(x) for x in results]


# ═════════════════════════════════════════════════════════════════════
#  Async path — for API-based models
# ═════════════════════════════════════════════════════════════════════


def _async_inference(
    chat_model: Any,
    inputs: list[Any],
    max_retries: int,
    base_delay: float,
    use_tqdm: bool = False,
) -> list[Any]:
    """Async inference with per-item retry and optional progress bar.

    Each input is processed independently via ``ainvoke()``, so a failure
    on one item doesn't block others. Uses ``asyncio.gather`` to maintain
    order.

    Best for API-based models where each request is an independent HTTP call.
    Do NOT use for VLLM — it blocks internally so you get zero parallelism.
    """

    async def _process_all():
        sem = asyncio.Semaphore(64)  # Cap concurrent requests

        async def process_single(input_item, pbar=None):
            async with sem:
                for attempt in range(max_retries):
                    try:
                        result = await chat_model.ainvoke(input_item)
                        if pbar is not None:
                            pbar.update(1)
                        return result
                    except Exception as e:
                        if attempt == max_retries - 1 or not _is_retryable_error(e):
                            raise
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "Retry %d/%d: %s. Waiting %.1fs...",
                            attempt + 1, max_retries, e, delay,
                        )
                        await asyncio.sleep(delay)

        if use_tqdm:
            with async_tqdm(total=len(inputs), desc="Inference (async)") as pbar:
                return await asyncio.gather(
                    *[process_single(inp, pbar) for inp in inputs]
                )
        else:
            return await asyncio.gather(
                *[process_single(inp) for inp in inputs]
            )

    return asyncio.run(_process_all())


# ═════════════════════════════════════════════════════════════════════
#  Batch path — for local models (VLLM, LlamaCpp)
# ═════════════════════════════════════════════════════════════════════


def _batch_inference(
    chat_model: Any,
    inputs: list[Any],
    max_retries: int,
    base_delay: float,
    use_tqdm: bool = False,
    batch_size: int | None = None,
) -> list[Any]:
    """Sync batch inference with chunk-level retry.

    Sends inputs to ``chat_model.batch()`` which lets the engine (e.g. vLLM)
    handle internal scheduling and batching optimally.

    On failure, splits the failing chunk into smaller sub-chunks to work
    around OOM or rate limit issues.

    Args:
        batch_size: If set, process inputs in chunks of this size.
            If None, send all inputs at once (optimal for vLLM).
    """
    # Split into chunks if batch_size is specified
    if batch_size and batch_size < len(inputs):
        chunks = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    else:
        chunks = [inputs]

    results: list[Any] = []
    pbar = sync_tqdm(total=len(inputs), desc="Inference (batch)") if use_tqdm else None

    try:
        for chunk in chunks:
            chunk_results = _batch_with_retry(
                chat_model, chunk, max_retries, base_delay,
            )
            results.extend(chunk_results)
            if pbar is not None:
                pbar.update(len(chunk_results))
    finally:
        if pbar is not None:
            pbar.close()

    return results


def _batch_with_retry(
    chat_model: Any,
    inputs: list[Any],
    max_retries: int,
    base_delay: float,
) -> list[Any]:
    """Execute a single batch with retry + progressive chunk splitting.

    On transient failure, splits the batch in half (2^attempt) and retries
    each sub-batch independently.
    """
    for attempt in range(max_retries):
        # Split into 2^attempt chunks on retry
        num_chunks = 2**attempt
        chunk_size = max(1, len(inputs) // num_chunks)
        sub_chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        try:
            results = []
            for sub_chunk in sub_chunks:
                results.extend(chat_model.batch(inputs=sub_chunk))
            return results
        except Exception as e:
            if attempt == max_retries - 1 or not _is_retryable_error(e):
                raise
            delay = base_delay * (2**attempt)
            next_chunks = 2 ** (attempt + 1)
            logger.warning(
                "Batch retry %d/%d: %s. Waiting %.1fs, splitting into %d chunks...",
                attempt + 1, max_retries, e, delay, next_chunks,
            )
            time.sleep(delay)

    raise RuntimeError("Inference failed after all retries")
