import re
from typing import Dict

import pandas as pd


def normalize_service_name(service_name: str) -> str:
    text = str(service_name).strip().lower()
    text = text.replace("generic_qa", "general_qa")
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "general_qa"
    return text


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
    parsed = parsed.reset_index(drop=True)
    return parsed


def load_service_descriptions(descriptions_path: str) -> Dict[str, str]:
    try:
        descriptions_df = pd.read_excel(descriptions_path)
    except Exception:
        return {}

    if "Dataset" not in descriptions_df.columns or "Description" not in descriptions_df.columns:
        return {}

    mapping: Dict[str, str] = {}
    for _, row in descriptions_df.iterrows():
        label = normalize_service_name(row["Dataset"])
        description = str(row["Description"]).strip()
        if label and description:
            mapping[label] = description
    return mapping
