from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class ColBERTRouter:
    def __init__(
        self,
        service_descriptions: Dict[str, str],
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cpu",
    ):
        self.service_labels = sorted(service_descriptions.keys())
        self.service_texts = [service_descriptions[label] for label in self.service_labels]
        self.model_name = model_name
        self.device = torch.device(device)
        torch.manual_seed(0)

        self.tokenizer, self.model, self.model_name = self._load_model_with_fallback(model_name)
        self.model.eval()
        self.service_token_embeddings = self._encode_tokens(self.service_texts)

    def _load_model_with_fallback(self, preferred_model_name: str):
        candidates = [preferred_model_name, "sentence-transformers/all-MiniLM-L6-v2"]
        last_error = None
        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModel.from_pretrained(candidate).to(self.device)
                return tokenizer, model, candidate
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Unable to load any ColBERT-style backbone from candidates: {candidates}") from last_error

    def _encode_tokens(self, texts: List[str]) -> List[np.ndarray]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded).last_hidden_state
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        attention = encoded["attention_mask"]
        embeddings: List[np.ndarray] = []
        for idx in range(output.size(0)):
            valid_positions = attention[idx].bool()
            token_matrix = output[idx][valid_positions].detach().cpu().numpy()
            embeddings.append(token_matrix)
        return embeddings

    @staticmethod
    def _maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        if query_tokens.size == 0 or doc_tokens.size == 0:
            return 0.0
        similarities = np.matmul(query_tokens, doc_tokens.T)
        max_per_query = similarities.max(axis=1)
        return float(np.mean(max_per_query))

    def _rank(self, prompt: str) -> Tuple[List[str], List[float]]:
        query_tokens = self._encode_tokens([prompt])[0]
        scores = [self._maxsim_score(query_tokens, doc_tokens) for doc_tokens in self.service_token_embeddings]
        order = np.argsort(-np.array(scores))
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
