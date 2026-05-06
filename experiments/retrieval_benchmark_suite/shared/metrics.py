import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def evaluate_predictions(rows: List[Dict]) -> Dict:
    predictions_df = pd.DataFrame(rows)
    if len(predictions_df) == 0:
        raise ValueError("No prediction rows generated.")

    accuracy = float(predictions_df["is_correct"].mean())
    top3_accuracy = float(predictions_df["top3_hit"].mean())
    per_class_accuracy = (
        predictions_df.groupby("expected_service")["is_correct"]
        .mean()
        .sort_index()
        .astype(float)
        .to_dict()
    )
    latency_values = predictions_df["latency_ms"].astype(float).to_numpy()
    latency_avg_ms = float(np.mean(latency_values))
    latency_p95_ms = float(np.percentile(latency_values, 95))
    best_class, best_score = max(per_class_accuracy.items(), key=lambda item: item[1])
    worst_class, worst_score = min(per_class_accuracy.items(), key=lambda item: item[1])

    return {
        "predictions_df": predictions_df,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "total_samples": int(len(predictions_df)),
        "per_class_accuracy": per_class_accuracy,
        "latency_avg_ms": latency_avg_ms,
        "latency_p95_ms": latency_p95_ms,
        "best_class": best_class,
        "best_class_accuracy": float(best_score),
        "worst_class": worst_class,
        "worst_class_accuracy": float(worst_score),
    }


def write_outputs(
    output_dir: str,
    method: str,
    model_name: str,
    dataset_name: str,
    eval_result: Dict,
):
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    metrics_path = os.path.join(output_dir, "metrics_summary.csv")

    predictions_df = eval_result["predictions_df"].copy()
    ordered_columns = [
        "prompt",
        "expected_service",
        "predicted_service",
        "rank_1",
        "rank_2",
        "rank_3",
        "similarity_scores",
        "is_correct",
        "model_name",
        "method",
        "dataset",
    ]
    predictions_df = predictions_df[ordered_columns]
    predictions_df.to_csv(predictions_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "method": method,
                "model": model_name,
                "dataset": dataset_name,
                "accuracy": round(eval_result["accuracy"], 6),
                "top3_accuracy": round(eval_result["top3_accuracy"], 6),
                "total_samples": int(eval_result["total_samples"]),
                "latency_avg_ms": round(eval_result["latency_avg_ms"], 6),
                "latency_p95_ms": round(eval_result["latency_p95_ms"], 6),
                "best_class": eval_result["best_class"],
                "best_class_accuracy": round(eval_result["best_class_accuracy"], 6),
                "worst_class": eval_result["worst_class"],
                "worst_class_accuracy": round(eval_result["worst_class_accuracy"], 6),
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)


def collect_method_metrics(outputs_root: str, dataset_name: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for metrics_file in Path(outputs_root).glob(f"*/{dataset_name}/metrics_summary.csv"):
        df = pd.read_csv(metrics_file)
        if len(df):
            rows.append(df.iloc[-1].to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
