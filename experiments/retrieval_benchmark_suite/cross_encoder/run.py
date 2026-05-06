from typing import Dict

from experiments.retrieval_benchmark_suite.cross_encoder.eval import evaluate_cross_encoder
from experiments.retrieval_benchmark_suite.cross_encoder.model import CrossEncoderRouter


def run_cross_encoder(
    prompts_df,
    service_descriptions: Dict[str, str],
    dataset_name: str,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu",
):
    router = CrossEncoderRouter(
        service_descriptions=service_descriptions,
        model_name=model_name,
        device=device,
    )
    return evaluate_cross_encoder(
        prompts_df=prompts_df,
        router=router,
        model_name=model_name,
        dataset_name=dataset_name,
    )
