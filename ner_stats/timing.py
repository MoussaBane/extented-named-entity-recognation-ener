"""Inference timing utilities for sentence-level processing."""

from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from .cva import classify_sentence_tokens_by_cosine_similarity


Sentence = str
InferenceFn = Callable[[Sentence], Any]


def measure_sentence_inference_time(
    sentences: Iterable[Sentence],
    inference_fn: InferenceFn,
    warmup_sentences: int = 0,
) -> Dict[str, Any]:
    """
    Measure inference time per sentence and average over a dataset.

    Args:
        sentences: Iterable of sentence strings.
        inference_fn: Callable that performs inference for one sentence.
        warmup_sentences: Number of initial sentences to run without timing.

    Returns:
        Dictionary with per-sentence timings, average time, total time, and outputs.
    """
    sentence_list = list(sentences)
    if not sentence_list:
        raise ValueError("sentences must contain at least one sentence.")

    warmup_count = max(0, min(int(warmup_sentences), len(sentence_list)))

    for sentence in sentence_list[:warmup_count]:
        inference_fn(sentence)

    per_sentence_times: List[float] = []
    outputs: List[Any] = []

    for sentence in sentence_list:
        start_time = perf_counter()
        output = inference_fn(sentence)
        elapsed = perf_counter() - start_time
        per_sentence_times.append(elapsed)
        outputs.append(output)

    total_time = float(np.sum(per_sentence_times))
    average_time = total_time / len(per_sentence_times)

    return {
        "per_sentence_times": per_sentence_times,
        "average_time": average_time,
        "total_time": total_time,
        "outputs": outputs,
    }


def measure_sentence_token_classification_time(
    sentences: Iterable[Sentence],
    class_vectors: Dict[str, Any],
    model_name: str = "dbmdz/bert-base-turkish-cased",
    warmup_sentences: int = 0,
) -> Dict[str, Any]:
    """
    Measure inference time for the cosine-similarity token classifier over sentences.

    Returns the same structure as measure_sentence_inference_time, with token-label
    predictions stored under the 'outputs' key.
    """

    def infer(sentence: Sentence) -> Any:
        return classify_sentence_tokens_by_cosine_similarity(
            sentence,
            class_vectors,
            model_name=model_name,
        )

    return measure_sentence_inference_time(
        sentences=sentences,
        inference_fn=infer,
        warmup_sentences=warmup_sentences,
    )
