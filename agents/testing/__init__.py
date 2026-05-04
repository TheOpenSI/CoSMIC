"""Scalability testing harness for OpenSI-CoSMIC (tiered domain specialists)."""

from agents.testing.coordinator import (
    FALLBACK_ROUTING_SERVICE,
    build_root_agent,
    default_csv_path,
    load_tier,
    tier_keys,
    tier_prediction_labels,
)
from agents.testing.specialists import ALL_KEYS, SPECIALISTS
from agents.testing.runner import RoutingOutcome, run_agent_routing

__all__ = [
    "ALL_KEYS",
    "FALLBACK_ROUTING_SERVICE",
    "SPECIALISTS",
    "RoutingOutcome",
    "build_root_agent",
    "default_csv_path",
    "load_tier",
    "run_agent_routing",
    "tier_keys",
    "tier_prediction_labels",
]
