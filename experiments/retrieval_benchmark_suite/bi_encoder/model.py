from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BiEncoderRouter:
    def __init__(
        self,
        service_descriptions: Dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.service_labels = sorted(service_descriptions.keys())
        self.service_texts = [service_descriptions[label] for label in self.service_labels]
        self.model_name = model_name
        self.device = torch.device(device)
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.service_embeddings = self._encode_texts(self.service_texts)

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        denom = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return pooled / denom

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
            embeddings = self._mean_pool(output, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def _rank(self, prompt: str) -> Tuple[List[str], List[float]]:
        query_embedding = self._encode_texts([prompt])[0]
        scores = np.dot(self.service_embeddings, query_embedding)
        order = np.argsort(-scores)
        ranked_labels = [self.service_labels[idx] for idx in order]
        ranked_scores = [float(scores[idx]) for idx in order]
        return ranked_labels, ranked_scores

    def predict(self, prompt: str) -> str:
        ranked, _ = self._rank(prompt)
        return ranked[0]

    def predict_topk(self, prompt: str, k: int) -> List[str]:
        ranked, _ = self._rank(prompt)
        return ranked[:k]

    def rank_with_scores(self, prompt: str, k: int = 3) -> Tuple[List[str], List[float]]:
        ranked, scores = self._rank(prompt)
        return ranked[:k], scores[:k]
