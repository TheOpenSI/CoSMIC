from typing import Dict

from experiments.retrieval_benchmark_suite.bi_encoder.model import BiEncoderRouter


def build_bi_encoder_router(
    service_descriptions: Dict[str, str],
    model_name: str,
    device: str,
) -> BiEncoderRouter:
    return BiEncoderRouter(
        service_descriptions=service_descriptions,
        model_name=model_name,
        device=device,
    )
