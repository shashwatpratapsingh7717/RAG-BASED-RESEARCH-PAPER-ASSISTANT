import time
import pandas as pd

def normalize_source(
    source: str
) -> str:
    return (
        source
        .strip()
        .lower()
        .replace("\\", "/")
        .split("/")[-1]
    )

def evaluate_retrieval(
    pipeline,
    questions,
    top_k_values=(3, 5)
):

    if not questions:
        raise ValueError(
            "No evaluation questions."
        )

    max_k = max(
        top_k_values
    )

    hit_counts = {
        k: 0
        for k in top_k_values
    }
    details = []
    latencies = []
    for item in questions:
        question = item["question"]
        expected_source = normalize_source(
            item["expected_source"]
        )

        start = time.perf_counter()
        results, _ = (
            pipeline.retriever.retrieve(
                question,
                top_k=max_k
            )
        )

        latency = (
            time.perf_counter()
            - start
        )

        latencies.append(
            latency
        )

        retrieved_sources = [
            normalize_source(
                result["source"]
            )

            for result in results
        ]

        row = {
            "question":
                question,
            "expected_source":
                expected_source,
            "top_1":
                retrieved_sources[0]
                if retrieved_sources
                else "",
            "latency_seconds":
                round(latency, 4)
        }

        for k in top_k_values:
            hit = (
                expected_source
                in retrieved_sources[:k]
            )

            hit_counts[k] += int(hit)
            row[
                f"hit_at_{k}"
            ] = hit
        details.append(row)
    total = len(
        questions
    )

    return {
        "hit_at_3":
            hit_counts.get(3, 0)
            / total,
        "hit_at_5":
            hit_counts.get(5, 0)
            / total,
        "avg_retrieval_latency":
            sum(latencies)
            / len(latencies),
        "details":
            pd.DataFrame(details)
    }