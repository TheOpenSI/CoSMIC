from typing import Dict

from experiments.retrieval_benchmark_suite.bi_encoder.eval import evaluate_bi_encoder
from experiments.retrieval_benchmark_suite.bi_encoder.index import build_bi_encoder_router


def run_bi_encoder(
    prompts_df,
    service_descriptions: Dict[str, str],
    dataset_name: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
):
    router = build_bi_encoder_router(
        service_descriptions=service_descriptions,
        model_name=model_name,
        device=device,
    )
    return evaluate_bi_encoder(
        prompts_df=prompts_df,
        router=router,
        model_name=model_name,
        dataset_name=dataset_name,
    )
