from typing import Dict

from experiments.retrieval_benchmark_suite.colbert.eval import evaluate_colbert
from experiments.retrieval_benchmark_suite.colbert.index import build_colbert_router


def run_colbert(
    prompts_df,
    service_descriptions: Dict[str, str],
    dataset_name: str,
    model_name: str = "colbert-ir/colbertv2.0",
    device: str = "cpu",
):
    router = build_colbert_router(
        service_descriptions=service_descriptions,
        model_name=model_name,
        device=device,
    )
    return evaluate_colbert(
        prompts_df=prompts_df,
        router=router,
        model_name=router.model_name,
        dataset_name=dataset_name,
    )
