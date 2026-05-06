from typing import Dict, List

import numpy as np

from experiments.semantic_router.src.embedder import EmbedderBase
from experiments.semantic_router.src.index import ServiceIndex


class SemanticRouter:
    def __init__(self, service_index: ServiceIndex, embedder: EmbedderBase):
        self.service_index = service_index
        self.embedder = embedder

    @staticmethod
    def _cosine_similarity(query_embedding: np.ndarray, service_embeddings: np.ndarray) -> np.ndarray:
        numerator = service_embeddings @ query_embedding
        denominator = np.linalg.norm(service_embeddings, axis=1) * np.linalg.norm(query_embedding)
        denominator[denominator == 0.0] = 1.0
        return numerator / denominator

    def route(self, prompt: str) -> Dict:
        prompt_embedding = self.embedder.embed([prompt])[0]
        scores = self._cosine_similarity(prompt_embedding, self.service_index.embedding_matrix)
        ranked_indices = np.argsort(-scores)

        ranked = [
            {
                "service": self.service_index.service_names[idx],
                "score": float(scores[idx]),
            }
            for idx in ranked_indices
        ]

        return {
            "predicted_service": ranked[0]["service"],
            "similarity_score": ranked[0]["score"],
            "ranked": ranked,
        }

    def route_batch(self, prompts: List[str]) -> List[Dict]:
        prompt_embeddings = self.embedder.embed(prompts)
        outputs = []

        for prompt_embedding in prompt_embeddings:
            scores = self._cosine_similarity(prompt_embedding, self.service_index.embedding_matrix)
            ranked_indices = np.argsort(-scores)
            ranked = [
                {
                    "service": self.service_index.service_names[idx],
                    "score": float(scores[idx]),
                }
                for idx in ranked_indices
            ]
            outputs.append(
                {
                    "predicted_service": ranked[0]["service"],
                    "similarity_score": ranked[0]["score"],
                    "ranked": ranked,
                }
            )

        return outputs
