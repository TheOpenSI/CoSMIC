import os
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    root_dir: str
    model_alias: str
    dataset_path: str
    descriptions_path: str

    @property
    def module_dir(self) -> str:
        return os.path.join(self.root_dir, "experiments", "tool_orchestra_benchmark")

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.module_dir, "outputs")

    @property
    def dataset_name(self) -> str:
        return os.path.splitext(os.path.basename(self.dataset_path))[0]

    @property
    def model_outputs_dir(self) -> str:
        return os.path.join(self.outputs_dir, f"{self.model_alias.lower()}-routing-model-outputs")

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
    def latency_log_path(self) -> str:
        return os.path.join(self.dataset_outputs_dir, "latency_log.csv")

    @property
    def confusion_matrix_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "confusion_matrix.png")

    @property
    def per_class_accuracy_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "per_class_accuracy.png")

    @property
    def model_accuracy_plot_path(self) -> str:
        return os.path.join(self.visualizations_dir, "model_accuracy.png")

    @property
    def resolved_model_name(self) -> str:
        model_map = {
            "openai": "text-embedding-3-large",
            "toolorchestra": "nvidia/Nemotron-Orchestrator-8B",
        }
        if self.model_alias not in model_map:
            raise ValueError(f"Unsupported model alias: {self.model_alias}")
        return model_map[self.model_alias]


def build_config(
    root_dir: str,
    model_alias: str,
    dataset_path: str,
    descriptions_path: str,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        root_dir=root_dir,
        model_alias=model_alias,
        dataset_path=dataset_path,
        descriptions_path=descriptions_path,
    )
