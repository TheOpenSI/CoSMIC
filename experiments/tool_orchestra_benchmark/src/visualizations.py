from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(predictions_df: pd.DataFrame, output_path: str):
    labels = sorted(
        set(predictions_df["expected_service"].astype(str).tolist())
        | set(predictions_df["predicted_service"].astype(str).tolist())
    )
    matrix = pd.crosstab(
        predictions_df["expected_service"],
        predictions_df["predicted_service"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_per_class_accuracy(per_class_accuracy: Dict[str, float], output_path: str):
    per_class_df = pd.DataFrame(
        [{"service": key, "accuracy": value} for key, value in per_class_accuracy.items()]
    ).sort_values("service")

    plt.figure(figsize=(12, 5))
    sns.barplot(data=per_class_df, x="service", y="accuracy", color="#4C72B0")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Service Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(outputs_dir: str, dataset_name: str, output_path: str):
    metrics_files = Path(outputs_dir).glob(f"*-routing-model-outputs/{dataset_name}/metrics_summary.csv")
    rows = []
    for metrics_file in metrics_files:
        df = pd.read_csv(metrics_file)
        if len(df):
            rows.append(df.iloc[-1].to_dict())

    if not rows:
        return

    comparison_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="model", y="accuracy", palette="deep")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("Top-1 Accuracy")
    plt.title(f"Model Comparison ({dataset_name})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
