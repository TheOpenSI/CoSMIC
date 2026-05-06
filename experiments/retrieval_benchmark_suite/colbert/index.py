from typing import Dict

from experiments.retrieval_benchmark_suite.colbert.model import ColBERTRouter


def build_colbert_router(
    service_descriptions: Dict[str, str],
    model_name: str,
    device: str,
) -> ColBERTRouter:
    return ColBERTRouter(
        service_descriptions=service_descriptions,
        model_name=model_name,
        device=device,
    )
