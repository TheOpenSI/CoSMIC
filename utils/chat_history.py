def build_context_from_messages(
    messages: list,
    num_pairs: int
) -> str:
    """
    Build a chat history from the given number of conversation pairs.

    Args:
        messages (list): List of messages in the chat history.
        num_pairs (int): Number of conversation pairs to include in the context.

    Returns:
        str: Formatted string representing the chat history.
    """
    if not messages:
        return ""

    pairs: list[tuple[str, str]] = []
    i: int = 0

    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            user_msg: str = messages[i]["content"]
            assistant_msg: str = messages[i + 1]["content"]

            pairs.append((user_msg, assistant_msg))
            i += 2

        else:
            i += 1

    selected_pairs: list[tuple[str ,str]] = pairs[-num_pairs:]

    context_parts: list[str] = []

    for idx, (user_msg, assistant_msg) in enumerate(
        iterable=selected_pairs,
        start=1
    ):
        block: str = "{0:s}\n{1:s}\n{2:s}\n{3:s}".format(
            f"Previous Conversation Pair {idx}",
            f"{'-'*30}",
            f"**User:** {user_msg}",
            f"**Assistant:** {assistant_msg}"
        )
        context_parts.append(block)

    full_context: str = "{0:s}\n\n\n{1:s}\n{2:s}\n".format(
        "Conversation History:",
        f"{context_parts}",
        f"{'='*15} End of Chat History {'='*15}"
    )

    return full_context


if __name__ == "__main__":
    messages: list[dict[str, str]] = [
        {
            'role': 'system',
            'content': 'PDF content from OpenwebUI'
        }
    ]

    print(
        build_context_from_messages(
            messages=messages,
            num_pairs=5
        )
    )
