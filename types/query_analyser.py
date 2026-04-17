### Core modules ###
from typing import TypedDict


### Type hints ###


### Internal modules ###


class ServicesJsonResponse(TypedDict):
    """
    Represents a service entry returned from the backend API endpoint.

    This class defines the structure of each service object in the Services API
    responses, specifically ``result`` array. Use this for type checking and
    IDE autocompletion when working with service data.

    Attributes:
        id:         Unique numeric identifier for the service.
        name:       URL-friendly service name/slug (e.g., "academic_governance").
        desc:       human-readable service description (used as display text).
        status:     indicate if the service is currently active or not.
        create_on:  8601 timestamp when the service was created (ISO 8601 format).

    Example:
        >>> service = ServicesJsonResponse(
        ...     id=5,
        ...     name="academic_governance",
        ...     desc="Answer question about Academic Governance.",
        ...     status=True,
        ...     create_on="2026-04-11T01:33:23.080388Z"
        ... )
        >>> service["id"]
        5
        >>> service["desc"]
        'Answer question about Academic Governance.'

    Note:
        The API returns 1-based IDs, but the internal representation
        converts these to 0-based indices by subtracting 1 (for legacy purposes).
    """
    id:         int
    desc:       str
    name:       str
    status:     bool
    create_on:  str
