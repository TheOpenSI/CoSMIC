from typing import Dict

import pandas as pd

from experiments.retrieval_benchmark_suite.shared.utils import normalize_service_name


def load_prompts_dataset(dataset_path: str) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    columns_lower = {col.lower(): col for col in data.columns}

    prompt_col = columns_lower.get("prompt")
    expected_col = columns_lower.get("expected_service") or columns_lower.get("service")
    if prompt_col is None or expected_col is None:
        raise ValueError(f"Dataset {dataset_path} must contain prompt and expected_service columns.")

    parsed = pd.DataFrame(
        {
            "prompt": data[prompt_col].astype(str).fillna(""),
            "expected_service": data[expected_col].astype(str).fillna(""),
        }
    )
    parsed["prompt"] = parsed["prompt"].astype(str).str.strip()
    parsed["expected_service"] = parsed["expected_service"].apply(normalize_service_name)
    parsed = parsed[(parsed["prompt"] != "") & (parsed["expected_service"] != "")]
    return parsed.reset_index(drop=True)


def load_service_descriptions(descriptions_path: str) -> Dict[str, str]:
    descriptions_df = pd.read_excel(descriptions_path)
    columns_lower = {col.lower(): col for col in descriptions_df.columns}

    service_col = columns_lower.get("dataset") or columns_lower.get("service")
    desc_col = columns_lower.get("description")
    if service_col is None or desc_col is None:
        raise ValueError(
            f"Descriptions file {descriptions_path} must contain Dataset/Service and Description columns."
        )

    mapping: Dict[str, str] = {}
    for _, row in descriptions_df.iterrows():
        label = normalize_service_name(row[service_col])
        description = str(row[desc_col]).strip()
        if label and description:
            mapping[label] = description
    return mapping
