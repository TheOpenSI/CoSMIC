import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List

from experiments.tool_orchestra_benchmark.src.dataset_utils import normalize_service_name


@dataclass
class PredictionOutput:
    predicted_service: str
    ranked_labels: List[str]
    latency_ms: float


class RouterBase:
    def predict(self, prompt: str) -> str:
        return self.predict_with_metadata(prompt).predicted_service

    def predict_with_metadata(self, prompt: str) -> PredictionOutput:
        raise NotImplementedError()


class OpenAIEmbeddingRouter(RouterBase):
    def __init__(self, services: List[str]):
        from experiments.semantic_router.src.config import build_config as build_semantic_config
        from experiments.semantic_router.src.embedder import build_embedder
        from experiments.semantic_router.src.index import ServiceIndex
        from experiments.semantic_router.src.router import SemanticRouter

        self.services = sorted({normalize_service_name(service) for service in services})
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        semantic_config = build_semantic_config(
            root_dir=root_dir,
            model_alias="openai",
            dataset_filename="dataset_3_services.csv",
        )
        embedder = build_embedder("openai")
        service_index = ServiceIndex(config=semantic_config, embedder=embedder)
        service_index.build(services=self.services)
        self.router = SemanticRouter(service_index=service_index, embedder=embedder)

    def predict_with_metadata(self, prompt: str) -> PredictionOutput:
        started = time.perf_counter()
        route_output = self.router.route(prompt)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = [normalize_service_name(item["service"]) for item in route_output["ranked"]]
        predicted = normalize_service_name(route_output["predicted_service"])
        return PredictionOutput(predicted_service=predicted, ranked_labels=ranked, latency_ms=elapsed_ms)


class ToolOrchestraRouter(RouterBase):
    def __init__(
        self,
        services: List[str],
        service_descriptions: Dict[str, str],
        model_name: str = "nvidia/Nemotron-Orchestrator-8B",
        device: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "ToolOrchestra routing requires torch and transformers."
            ) from exc

        self.torch = torch
        self.services = sorted({normalize_service_name(service) for service in services})
        self.service_descriptions = service_descriptions
        self.model_name = model_name

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        elif device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        else:
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        self.model.eval()

    @staticmethod
    def _normalize_text_candidates(text: str) -> List[str]:
        chunks = re.split(r"[\n,;|:/]+", str(text))
        candidates = []
        for chunk in chunks:
            normalized = normalize_service_name(chunk)
            if normalized:
                candidates.append(normalized)
        return candidates

    def _extract_ranked_labels(self, completion: str) -> List[str]:
        known = set(self.services)
        candidates = self._normalize_text_candidates(completion)
        ranked: List[str] = []

        for candidate in candidates:
            if candidate in known and candidate not in ranked:
                ranked.append(candidate)

        if ranked:
            return ranked

        compact_completion = normalize_service_name(completion).replace("_", "")
        for service in self.services:
            compact_service = service.replace("_", "")
            if compact_service and compact_service in compact_completion:
                ranked.append(service)
        return ranked

    def _build_prompt(self, prompt: str) -> str:
        labels_text = "\n".join(f"- {label}" for label in self.services)
        description_lines = []
        for label in self.services:
            if label in self.service_descriptions:
                description_lines.append(f"- {label}: {self.service_descriptions[label]}")
        descriptions_text = "\n".join(description_lines)

        return (
            "You are a strict classifier.\n"
            "Return exactly one label from the allowed labels list.\n"
            "Output must contain only the label token and nothing else.\n\n"
            "Allowed labels:\n"
            f"{labels_text}\n\n"
            "Label descriptions:\n"
            f"{descriptions_text}\n\n"
            f"User prompt: {prompt}\n"
            "Label:"
        )

    def predict_with_metadata(self, prompt: str) -> PredictionOutput:
        started = time.perf_counter()
        input_text = self._build_prompt(prompt)
        encoded = self.tokenizer(input_text, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with self.torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        completion = decoded[len(input_text):].strip() if decoded.startswith(input_text) else decoded.strip()
        ranked = self._extract_ranked_labels(completion)
        predicted = ranked[0] if ranked else "general_qa"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not ranked:
            ranked = [predicted]
        return PredictionOutput(predicted_service=predicted, ranked_labels=ranked[:3], latency_ms=elapsed_ms)


def build_router(
    model_alias: str,
    services: List[str],
    service_descriptions: Dict[str, str],
    device: str,
) -> RouterBase:
    if model_alias == "openai":
        return OpenAIEmbeddingRouter(services=services)
    if model_alias == "toolorchestra":
        return ToolOrchestraRouter(
            services=services,
            service_descriptions=service_descriptions,
            device=device,
        )
    raise ValueError(f"Unsupported model alias: {model_alias}")
