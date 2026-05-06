import os
from typing import Dict, List

import pandas as pd

from experiments.tool_orchestra_benchmark.src.config import BenchmarkConfig
from experiments.tool_orchestra_benchmark.src.models import RouterBase


class Evaluator:
    def __init__(self, config: BenchmarkConfig, router: RouterBase):
        self.config = config
        self.router = router

    def run(
        self,
        prompts_df: pd.DataFrame,
        model_name: str,
        log_progress: bool = False,
        checkpoint_every: int = 0,
    ) -> Dict:
        prompts = prompts_df["prompt"].astype(str).tolist()
        expected = prompts_df["expected_service"].astype(str).tolist()
        total = len(prompts)
        predictions: List[Dict] = []
        latency_rows: List[Dict] = []
        top3_hits: List[bool] = []

        for idx, prompt in enumerate(prompts):
            output = self.router.predict_with_metadata(prompt)
            expected_service = expected[idx]
            predicted_service = output.predicted_service
            is_correct = predicted_service == expected_service
            is_top3 = expected_service in output.ranked_labels[:3]

            top3_hits.append(is_top3)
            predictions.append(
                {
                    "prompt": prompt,
                    "expected_service": expected_service,
                    "predicted_service": predicted_service,
                    "is_correct": bool(is_correct),
                    "model_name": model_name,
                    "dataset_name": self.config.dataset_name,
                }
            )
            latency_rows.append(
                {
                    "sample_index": idx,
                    "latency_ms": round(output.latency_ms, 6),
                    "predicted_service": predicted_service,
                }
            )
            if log_progress:
                print(
                    f"[{idx + 1}/{total}] idx={idx} predicted={predicted_service} "
                    f"expected={expected_service} correct={is_correct} latency_ms={output.latency_ms:.1f}",
                    flush=True,
                )
            if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
                self._write_checkpoints(
                    predictions=predictions,
                    latency_rows=latency_rows,
                    top3_hits=top3_hits,
                    model_name=model_name,
                    completed=idx + 1,
                    total=total,
                )

        predictions_df = pd.DataFrame(predictions)
        latency_df = pd.DataFrame(latency_rows)

        accuracy = float(predictions_df["is_correct"].mean()) if len(predictions_df) else 0.0
        top3_accuracy = float(pd.Series(top3_hits).mean()) if top3_hits else 0.0
        per_class_accuracy = (
            predictions_df.groupby("expected_service")["is_correct"]
            .mean()
            .sort_index()
            .astype(float)
            .to_dict()
        )

        best_class, best_score = max(per_class_accuracy.items(), key=lambda item: item[1])
        worst_class, worst_score = min(per_class_accuracy.items(), key=lambda item: item[1])

        metrics_summary = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dataset": self.config.dataset_name,
                    "accuracy": round(accuracy, 6),
                    "top3_accuracy": round(top3_accuracy, 6),
                    "total_samples": int(len(predictions_df)),
                    "best_class": best_class,
                    "best_class_accuracy": round(best_score, 6),
                    "worst_class": worst_class,
                    "worst_class_accuracy": round(worst_score, 6),
                }
            ]
        )

        return {
            "predictions_df": predictions_df,
            "latency_df": latency_df,
            "metrics_summary_df": metrics_summary,
            "per_class_accuracy": per_class_accuracy,
        }

    def _write_checkpoints(
        self,
        predictions: List[Dict],
        latency_rows: List[Dict],
        top3_hits: List[bool],
        model_name: str,
        completed: int,
        total: int,
    ):
        os.makedirs(self.config.dataset_outputs_dir, exist_ok=True)
        os.makedirs(self.config.visualizations_dir, exist_ok=True)

        checkpoint_predictions_df = pd.DataFrame(predictions)
        checkpoint_latency_df = pd.DataFrame(latency_rows)
        checkpoint_predictions_df.to_csv(self.config.predictions_path, index=False)
        checkpoint_latency_df.to_csv(self.config.latency_log_path, index=False)

        if len(checkpoint_predictions_df):
            accuracy = float(checkpoint_predictions_df["is_correct"].mean())
            top3_accuracy = float(pd.Series(top3_hits).mean()) if top3_hits else 0.0
            per_class = (
                checkpoint_predictions_df.groupby("expected_service")["is_correct"]
                .mean()
                .sort_index()
                .astype(float)
                .to_dict()
            )
            best_class, best_score = max(per_class.items(), key=lambda item: item[1])
            worst_class, worst_score = min(per_class.items(), key=lambda item: item[1])
            checkpoint_metrics = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "dataset": self.config.dataset_name,
                        "accuracy": round(accuracy, 6),
                        "top3_accuracy": round(top3_accuracy, 6),
                        "total_samples": int(len(checkpoint_predictions_df)),
                        "best_class": best_class,
                        "best_class_accuracy": round(best_score, 6),
                        "worst_class": worst_class,
                        "worst_class_accuracy": round(worst_score, 6),
                    }
                ]
            )
            checkpoint_metrics.to_csv(self.config.metrics_summary_path, index=False)

        print(
            f"[checkpoint] saved {completed}/{total} rows to {self.config.dataset_outputs_dir}",
            flush=True,
        )

    def persist_outputs(self, eval_result: Dict):
        os.makedirs(self.config.dataset_outputs_dir, exist_ok=True)
        os.makedirs(self.config.visualizations_dir, exist_ok=True)
        eval_result["predictions_df"].to_csv(self.config.predictions_path, index=False)
        eval_result["latency_df"].to_csv(self.config.latency_log_path, index=False)
        eval_result["metrics_summary_df"].to_csv(self.config.metrics_summary_path, index=False)
