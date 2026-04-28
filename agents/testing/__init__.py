"""Scalability testing harness for OpenSI-CoSMIC (tiered domain specialists)."""

from agents.testing.coordinator_factory import (
    build_root_agent,
    default_csv_path,
    load_tier,
    tier_keys,
)
from agents.testing.specialists import ALL_KEYS, SPECIALISTS

__all__ = [
    "ALL_KEYS",
    "SPECIALISTS",
    "build_root_agent",
    "default_csv_path",
    "load_tier",
    "tier_keys",
]
