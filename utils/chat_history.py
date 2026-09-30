### Core modules ###


### Type hints ###
from typing import Any


### Internal modules ###


def build_context_from_chat_history(
    details: list[Any],
    num_pairs: int = 5
) -> str:
    """
    Build the LLM conversation context from the completed blocks of a `details`
    list. Returns an empty string when there are no complete pairs.

    Args:
        details (list): completed chat history blocks (attribute access).
        num_pairs (int): number of most recent conversation pairs to include.

    Returns:
        str: formatted conversation history, or "" when none is available.
    """
    pairs: list[tuple[str, str]] = [
        (
            (
                chat_history_field.user_query.split("</files>", 1)[1].strip()
                if   ("</files>" in chat_history_field.user_query)
                else (chat_history_field.user_query)
            ),
            chat_history_field.llm_response
        )
        for chat_history_field in details
        if (
            chat_history_field.user_query
            and
            chat_history_field.llm_response
        )
    ]

    selected_pairs: list[tuple[str, str]] = pairs[-num_pairs:]

    if not selected_pairs:
        return ""

    context_parts: list[str] = []
    for (
        index,
        (user_msg, assistant_msg)
    ) in enumerate(
        iterable=selected_pairs,
        start=1
    ):
        context_parts.append(
            "Previous Conversation Pair {0:d}\n{1:s}\n**User:** {2:s}\n**Assistant:** {3:s}".format(
                index,
                "-" * 30,
                user_msg,
                assistant_msg
            )
        )

    return (
        "{0:s} Conversation History {0:s}\n{1:s}\n{0:s} End of Chat History {0:s}\n".format(
            ("=" * 15),
            "\n\n".join(context_parts)
        )
    )
