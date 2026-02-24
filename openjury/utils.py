import asyncio
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.asyncio import tqdm
from openjury.common.paths import data_root, set_langchain_cache
from openjury.common.timing import Timeblock
from openjury.io.download import download_hf, download_all
from openjury.io.tabular import read_df


def do_inference(chat_model, inputs, use_tqdm: bool = False):
    # Retries on rate-limit/server errors with exponential backoff.
    # Async path retries individual calls; batch path splits into 4^attempt chunks on failure.
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    if use_tqdm:
        # perform inference asynchronously to be able to update tqdm, chat_model.batch does not work as it blocks until
        # all requests are received
        async def process_with_real_progress(chat_model, inputs, pbar):
            async def process_single(input_item, max_retries=5, base_delay=1.0):
                for attempt in range(max_retries):
                    try:
                        result = await chat_model.ainvoke(input_item, **invoke_kwargs)
                        pbar.update(1)
                        return result
                    except Exception as e:
                        is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                        if attempt == max_retries - 1 or not is_rate_limit:
                            raise
                        delay = base_delay * (2**attempt)
                        print(
                            f"Retry because of a server error, {attempt + 1}/{max_retries}: {e}. Waiting {delay}s..."
                        )
                        await asyncio.sleep(delay)

            # asyncio.gather preserves order (unlike as_completed)
            results = await asyncio.gather(*[process_single(inp) for inp in inputs])
            return results

        with tqdm(total=len(inputs)) as pbar:
            res = asyncio.run(
                process_with_real_progress(
                    chat_model=chat_model, inputs=inputs, pbar=pbar
                )
            )
    else:

        def batch_with_retry(batch_inputs, max_retries=5, base_delay=1.0):
            for attempt in range(max_retries):
                num_chunks = 4**attempt
                chunk_size = max(1, len(batch_inputs) // num_chunks)
                chunks = [
                    batch_inputs[i : i + chunk_size]
                    for i in range(0, len(batch_inputs), chunk_size)
                ]
                try:
                    results = []
                    for chunk in chunks:
                        results.extend(chat_model.batch(inputs=chunk, **invoke_kwargs))
                    return results
                except Exception as e:
                    is_server_error = (
                        "429" in str(e)
                        or "500" in str(e)
                        or "502" in str(e)
                        or "503" in str(e)
                        or "rate" in str(e).lower()
                    )
                    if attempt == max_retries - 1 or not is_server_error:
                        raise
                    delay = base_delay * (2**attempt)
                    next_chunks = 4 ** (attempt + 1)
                    print(
                        f"Retry because of a server error, {attempt + 1}/{max_retries}: {e}. Waiting {delay}s, then splitting into {next_chunks} chunks..."
                    )
                    time.sleep(delay)

        res = batch_with_retry(inputs)

    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
    # is it because of using Chat and barebones models?
    # when using OpenAI, the output is AIMessage not a string...
    res = [x.content if hasattr(x, "content") else x for x in res]
    return res


class DummyModel:
    def __init__(self, name: str):
        self.name = name
        self.message = "/".join(name.split("/")[1:])

    def batch(self, inputs, **invoke_kwargs) -> list[str]:
        return [self.message] * len(inputs)

    def invoke(self, input, **invoke_kwargs) -> str:
        return self.message

    async def ainvoke(self, input, **invoke_kwargs):
        return self.message


def make_model(
    model: str,
    max_tokens: int | None = 8192,
    chat_template: str | None = None,
    chat_template_file: str | None = None,
):
    """Compatibility wrapper around the refactored model factory."""
    from openjury.models.factory import make_model as _make_model

    return _make_model(
        model,
        max_tokens=max_tokens,
        chat_template=chat_template,
        chat_template_file=chat_template_file,
    )


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = data_root / "cache"
    cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


def compute_cohen_kappa(y1: list[str], y2: list[str]) -> float:
    """
    Compute Cohen's kappa coefficient for inter-rater agreement.

    Args:
        y1: List of labels from first rater
        y2: List of labels from second rater

    Returns:
        Cohen's kappa coefficient (float between -1 and 1)
    """
    if len(y1) != len(y2):
        raise ValueError("Both lists must have the same length")

    if len(y1) == 0:
        raise ValueError("Lists cannot be empty")

    # Get all unique categories
    categories = sorted(set(y1) | set(y2))
    n = len(y1)

    # Build confusion matrix
    matrix = {}
    for cat1 in categories:
        matrix[cat1] = {cat2: 0 for cat2 in categories}

    for label1, label2 in zip(y1, y2):
        matrix[label1][label2] += 1

    # Compute observed agreement (p_o)
    observed_agreement = sum(matrix[cat][cat] for cat in categories) / n

    # Compute expected agreement (p_e)
    expected_agreement = 0
    for cat in categories:
        # Marginal probabilities
        p1 = sum(matrix[cat][c] for c in categories) / n  # rater 1
        p2 = sum(matrix[c][cat] for c in categories) / n  # rater 2
        expected_agreement += p1 * p2

    # Compute Cohen's kappa
    if expected_agreement == 1:
        return 1.0 if observed_agreement == 1 else 0.0

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa


if __name__ == "__main__":
    download_all()
