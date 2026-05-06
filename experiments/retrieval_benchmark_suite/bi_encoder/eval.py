import time
from typing import Dict, List

import pandas as pd

from experiments.retrieval_benchmark_suite.shared.metrics import evaluate_predictions


def evaluate_bi_encoder(
    prompts_df: pd.DataFrame,
    router,
    model_name: str,
    dataset_name: str,
) -> Dict:
    rows: List[Dict] = []
    prompts = prompts_df["prompt"].astype(str).tolist()
    expected = prompts_df["expected_service"].astype(str).tolist()

    for idx, prompt in enumerate(prompts):
        started = time.perf_counter()
        ranked_labels, ranked_scores = router.rank_with_scores(prompt, k=3)
        latency_ms = (time.perf_counter() - started) * 1000.0

        rank_1 = ranked_labels[0] if len(ranked_labels) > 0 else ""
        rank_2 = ranked_labels[1] if len(ranked_labels) > 1 else ""
        rank_3 = ranked_labels[2] if len(ranked_labels) > 2 else ""
        predicted_service = rank_1
        expected_service = expected[idx]

        rows.append(
            {
                "prompt": prompt,
                "expected_service": expected_service,
                "predicted_service": predicted_service,
                "rank_1": rank_1,
                "rank_2": rank_2,
                "rank_3": rank_3,
                "similarity_scores": "|".join(f"{score:.6f}" for score in ranked_scores),
                "is_correct": bool(predicted_service == expected_service),
                "top3_hit": bool(expected_service in ranked_labels[:3]),
                "latency_ms": round(latency_ms, 6),
                "model_name": model_name,
                "method": "bi-encoder",
                "dataset": dataset_name,
            }
        )

    return evaluate_predictions(rows)
