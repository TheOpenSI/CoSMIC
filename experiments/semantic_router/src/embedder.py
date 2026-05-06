from typing import List

import numpy as np


class EmbedderBase:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError()


class HFTransformerEmbedder(EmbedderBase):
    def __init__(self, model_name: str):
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "transformers is required for local embedding models. Install with: pip install transformers"
            ) from exc

        self._torch = self._load_torch()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _load_torch():
        try:
            import torch
            return torch
        except Exception as exc:
            raise ImportError(
                "torch is required for local embedding models. Install with: pip install torch"
            ) from exc

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask, torch_module):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch_module.sum(last_hidden_state * mask, dim=1)
        counts = torch_module.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: List[str]) -> np.ndarray:
        with self._torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"], self._torch)
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)


class MiniLMEmbedder(HFTransformerEmbedder):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")


class BGEEmbedder(HFTransformerEmbedder):
    def __init__(self):
        super().__init__("BAAI/bge-base-en-v1.5")


class OpenAIEmbedder(EmbedderBase):
    def __init__(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError(
                "openai is required for --model openai. Install with: pip install openai"
            ) from exc

        self.client = OpenAI()
        self.model_name = "text-embedding-3-large"

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def build_embedder(model_alias: str) -> EmbedderBase:
    if model_alias == "miniLM":
        return MiniLMEmbedder()
    if model_alias == "bge":
        return BGEEmbedder()
    if model_alias == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unsupported model alias: {model_alias}")
