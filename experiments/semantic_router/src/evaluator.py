import os
from typing import Dict, List

import pandas as pd

from experiments.semantic_router.src.config import RouterConfig
from experiments.semantic_router.src.dataset_utils import (
    load_known_services_from_descriptions,
    load_prompts_dataset,
    redirect_unknown_services_to_general_qa,
)
from experiments.semantic_router.src.router import SemanticRouter


class Evaluator:
    def __init__(self, config: RouterConfig, router: SemanticRouter):
        self.config = config
        self.router = router

    def run(self) -> Dict:
        data = load_prompts_dataset(self.config.dataset_prompts_path)
        known_services = load_known_services_from_descriptions(self.config.dataset_descriptions_path)
        data = redirect_unknown_services_to_general_qa(data, known_services)
        prompts = data["prompt"].astype(str).tolist()
        expected = data["expected_service"].astype(str).tolist()

        routed = self.router.route_batch(prompts)
        predictions: List[Dict] = []
        top3_hits: List[bool] = []

        for idx, route_output in enumerate(routed):
            predicted_service = route_output["predicted_service"]
            expected_service = expected[idx]
            ranked_services = [item["service"] for item in route_output["ranked"]]
            top3_ok = expected_service in ranked_services[: min(3, len(ranked_services))]
            correct = predicted_service == expected_service
            top3_hits.append(top3_ok)
            predictions.append(
                {
                    "prompt": prompts[idx],
                    "expected_service": expected_service,
                    "predicted_service": predicted_service,
                    "similarity_score": round(route_output["similarity_score"], 6),
                    "correct": bool(correct),
                }
            )

        predictions_df = pd.DataFrame(predictions)
        accuracy = float(predictions_df["correct"].mean())
        top3_accuracy = float(pd.Series(top3_hits).mean())

        per_class_accuracy = (
            predictions_df.groupby("expected_service")["correct"]
            .mean()
            .sort_index()
            .astype(float)
            .to_dict()
        )

        return {
            "predictions_df": predictions_df,
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "total_samples": int(len(predictions_df)),
            "per_class_accuracy": per_class_accuracy,
        }

    def persist_outputs(self, model_name: str, eval_result: Dict):
        os.makedirs(self.config.dataset_outputs_dir, exist_ok=True)
        os.makedirs(self.config.visualizations_dir, exist_ok=True)

        predictions_df = eval_result["predictions_df"].copy()
        predictions_df.to_csv(self.config.predictions_path, index=False)

        metrics_row = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dataset": self.config.dataset_name,
                    "accuracy": round(eval_result["accuracy"], 6),
                    "top3_accuracy": round(eval_result["top3_accuracy"], 6),
                    "total_samples": int(eval_result["total_samples"]),
                }
            ]
        )

        if os.path.exists(self.config.metrics_summary_path):
            history = pd.read_csv(self.config.metrics_summary_path)
            history = pd.concat([history, metrics_row], ignore_index=True)
        else:
            history = metrics_row
        history.to_csv(self.config.metrics_summary_path, index=False)

        per_class_rows = pd.DataFrame(
            [
                {"model": model_name, "service": service, "accuracy": round(score, 6)}
                for service, score in eval_result["per_class_accuracy"].items()
            ]
        )
        if os.path.exists(self.config.per_class_history_path):
            per_class_history = pd.read_csv(self.config.per_class_history_path)
            per_class_history = pd.concat([per_class_history, per_class_rows], ignore_index=True)
        else:
            per_class_history = per_class_rows
        per_class_history.to_csv(self.config.per_class_history_path, index=False)
