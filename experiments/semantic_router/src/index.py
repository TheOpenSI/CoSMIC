from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.semantic_router.src.config import RouterConfig
from experiments.semantic_router.src.dataset_utils import normalize_service_name
from experiments.semantic_router.src.embedder import EmbedderBase


class ServiceIndex:
    def __init__(self, config: RouterConfig, embedder: EmbedderBase):
        self.config = config
        self.embedder = embedder
        self.service_to_vector: Dict[str, np.ndarray] = {}
        self.service_names: List[str] = []
        self.embedding_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize_key(text: str) -> str:
        return normalize_service_name(text)

    def _resolve_service_descriptions(
        self,
        services: List[str],
        descriptions_df: pd.DataFrame,
    ) -> List[Tuple[str, str]]:
        if "Dataset" not in descriptions_df.columns or "Description" not in descriptions_df.columns:
            raise ValueError("dataset_descriptions.xlsx must contain columns: Dataset, Description")

        dataset_to_description = {}
        for _, row in descriptions_df.iterrows():
            dataset_name = self._normalize_key(row["Dataset"])
            description = str(row["Description"]).strip()
            if dataset_name and description:
                dataset_to_description[dataset_name] = description

        # Ensure universal fallback exists, even if spreadsheet omits it.
        if "general_qa" not in dataset_to_description:
            dataset_to_description["general_qa"] = (
                "General question answering service for ambiguous or unmatched prompts."
            )

        pairs = []
        for service in services:
            key = self._normalize_key(service)
            if key in dataset_to_description:
                pairs.append((key, dataset_to_description[key]))
                continue

            relaxed_match = None
            relaxed_service = key.replace("_", "")
            for dataset_key, description in dataset_to_description.items():
                dataset_compact = dataset_key.replace("_", "")
                if dataset_compact == relaxed_service or relaxed_service in dataset_compact or dataset_compact in relaxed_service:
                    relaxed_match = description
                    relaxed_name = dataset_key
                    break

            if relaxed_match is None:
                pairs.append(("general_qa", dataset_to_description["general_qa"]))
                continue

            pairs.append((relaxed_name, relaxed_match))

        # Remove duplicates while keeping first occurrence order.
        unique = {}
        for service_name, description in pairs:
            if service_name not in unique:
                unique[service_name] = description
        return list(unique.items())

    def build(self, services: List[str]):
        try:
            descriptions_df = pd.read_excel(self.config.dataset_descriptions_path)
        except Exception as exc:
            raise RuntimeError(
                "Unable to read dataset_descriptions.xlsx. Ensure openpyxl is installed: pip install openpyxl"
            ) from exc
        service_description_pairs = self._resolve_service_descriptions(services, descriptions_df)

        service_names = [item[0] for item in service_description_pairs]
        descriptions = [item[1] for item in service_description_pairs]
        vectors = self.embedder.embed(descriptions)

        self.service_names = service_names
        self.embedding_matrix = vectors
        self.service_to_vector = {
            service: vectors[idx]
            for idx, service in enumerate(service_names)
        }
