import re
from pathlib import Path
from typing import List


def normalize_service_name(service_name: str) -> str:
    text = str(service_name).strip().lower()
    text = text.replace("generic_qa", "general_qa")
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "general_qa"
    return text


def dataset_name_from_path(dataset_path: str) -> str:
    return Path(dataset_path).stem


def canonical_dataset_paths(root_dir: str) -> List[str]:
    names = [
        "dataset_3_services.csv",
        "dataset_5_services.csv",
        "dataset_8_services.csv",
        "dataset_10_services.csv",
    ]
    return [str(Path(root_dir) / "datasets" / name) for name in names]
