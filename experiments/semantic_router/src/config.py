import os
from dataclasses import dataclass


@dataclass
class RouterConfig:
    root_dir: str
    model_alias: str
    dataset_filename: str = "dataset_3_services.csv"
    similarity_metric: str = "cosine"

    @property
    def dataset_prompts_path(self) -> str:
        return os.path.join(self.root_dir, "datasets", self.dataset_filename)

    @property
    def dataset_descriptions_path(self) -> str:
        return os.path.join(self.root_dir, "datasets", "dataset_descriptions.xlsx")

    @property
    def module_dir(self) -> str:
        return os.path.join(self.root_dir, "experiments", "semantic_router")

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.module_dir, "outputs")

    @property
    def model_outputs_dir(self) -> str:
        model_outputs_name = f"{self.model_alias.lower()}-embedding-model-outputs"
        return os.path.join(self.outputs_dir, model_outputs_name)

    @property
    def dataset_name(self) -> str:
        return os.path.splitext(self.dataset_filename)[0]

    @property
    def dataset_outputs_dir(self) -> str:
        return os.path.join(self.model_outputs_dir, self.dataset_name)

    @property
    def visualizations_dir(self) -> str:
        return os.path.join(self.dataset_outputs_dir, "visualizations")

    @property
    def predictions_path(self) -> str:
        return os.path.join(self.dataset_outputs_dir, "predictions.csv")

    @property
    def metrics_summary_path(self) -> str:
        return os.path.join(self.dataset_outputs_dir, "metrics_summary.csv")

    @property
    def per_class_history_path(self) -> str:
        return os.path.join(self.dataset_outputs_dir, "per_class_metrics_history.csv")

    @property
    def model_run_summary_path(self) -> str:
        return os.path.join(self.model_outputs_dir, "all_datasets_metrics_summary.csv")

    @property
    def model_accuracy_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "model_accuracy.png")

    @property
    def per_class_accuracy_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "per_class_accuracy.png")

    @property
    def confusion_matrix_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "confusion_matrix.png")

    @property
    def resolved_model_name(self) -> str:
        model_map = {
            "miniLM": "all-MiniLM-L6-v2",
            "bge": "bge-base-en-v1.5",
            "openai": "text-embedding-3-large",
        }
        if self.model_alias not in model_map:
            raise ValueError(f"Unsupported model alias: {self.model_alias}")
        return model_map[self.model_alias]


def build_config(root_dir: str, model_alias: str, dataset_filename: str = "dataset_3_services.csv") -> RouterConfig:
    return RouterConfig(
        root_dir=root_dir,
        model_alias=model_alias,
        dataset_filename=dataset_filename,
    )
