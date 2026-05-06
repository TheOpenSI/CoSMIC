import re

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

    prompt_col = None
    service_col = None

    if "prompt" in columns_lower:
        prompt_col = columns_lower["prompt"]
    if "expected_service" in columns_lower:
        service_col = columns_lower["expected_service"]
    elif "service" in columns_lower:
        service_col = columns_lower["service"]

    if prompt_col is None and len(data.columns) >= 1:
        prompt_col = data.columns[0]
    if service_col is None and len(data.columns) >= 2:
        service_col = data.columns[1]

    if prompt_col is None or service_col is None:
        raise ValueError(
            f"Dataset {dataset_path} is missing usable prompt/service columns."
        )

    parsed = pd.DataFrame(
        {
            "prompt": data[prompt_col].astype(str).fillna(""),
            "expected_service": data[service_col].astype(str).fillna(""),
        }
    )

    parsed["prompt"] = parsed["prompt"].astype(str).str.strip()
    parsed["expected_service"] = parsed["expected_service"].apply(normalize_service_name)
    parsed = parsed[(parsed["prompt"] != "") & (parsed["expected_service"] != "")]
    parsed = parsed.reset_index(drop=True)
    return parsed


def load_known_services_from_descriptions(descriptions_path: str) -> set[str]:
    descriptions_df = pd.read_excel(descriptions_path)
    if "Dataset" not in descriptions_df.columns:
        return {"general_qa"}

    known = {
        normalize_service_name(value)
        for value in descriptions_df["Dataset"].astype(str).tolist()
        if str(value).strip() != ""
    }
    known.add("general_qa")
    return known


def redirect_unknown_services_to_general_qa(
    prompts_df: pd.DataFrame,
    known_services: set[str],
) -> pd.DataFrame:
    redirected = prompts_df.copy()
    redirected["expected_service"] = redirected["expected_service"].apply(
        lambda service: service if service in known_services else "general_qa"
    )
    return redirected
