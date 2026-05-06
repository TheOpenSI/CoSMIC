import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderRouter:
    def __init__(
        self,
        service_descriptions: Dict[str, str],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        # Store routing targets in a stable order so ranking output is deterministic.
        self.service_labels = sorted(service_descriptions.keys())
        self.service_texts = [service_descriptions[label] for label in self.service_labels]
        self.model_name = model_name
        self.device = torch.device(device)
        torch.manual_seed(0)

        # Initialize the model/tokenizer pair. If the preferred checkpoint is unstable on this system,
        # fall back to a known working checkpoint.
        self.tokenizer, self.model, self.model_name = self._load_stable_model(preferred_model=model_name)

    def _load_stable_model(self, preferred_model: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        candidates = [
            preferred_model,
            "cross-encoder/stsb-distilroberta-base",
            "cross-encoder/stsb-roberta-base",
        ]
        probe_query = "What is photosynthesis?"
        probe_doc = "Photosynthesis is the process by which green plants convert light into chemical energy."
        last_error = None

        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForSequenceClassification.from_pretrained(candidate).to(self.device)
                model = model.float()
                model.eval()

                encoded = tokenizer(
                    [probe_query],
                    [probe_doc],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    probe_scores = self._safe_scores(model(**encoded).logits)
                if np.isfinite(probe_scores).all():
                    return tokenizer, model, candidate
                last_error = RuntimeError(f"Non-finite probe logits for model: {candidate}")
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"Unable to initialize a stable cross-encoder from candidates: {candidates}"
        ) from last_error

    def _safe_scores(self, logits: torch.Tensor) -> np.ndarray:
        if logits.ndim == 2 and logits.shape[-1] > 1:
            # Binary/multi-class heads: use positive/relevance logit.
            logits = logits[:, -1]
        else:
            logits = logits.squeeze(-1)

        scores = logits.detach().float().cpu().numpy()
        finite_mask = np.isfinite(scores)
        if finite_mask.all():
            return scores

        # Replace non-finite values with a deterministic, very low score.
        repaired = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        if not np.isfinite(repaired).all():
            repaired = np.full_like(scores, fill_value=-1e9, dtype=np.float32)
        return repaired.astype(np.float32)

    def _rank(self, prompt: str) -> Tuple[List[str], List[float]]:
        pairs = [(prompt, service_text) for service_text in self.service_texts]
        encoded = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = self._safe_scores(outputs.logits)
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
