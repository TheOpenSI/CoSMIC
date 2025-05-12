def build_context_from_messages(messages: list, num_pairs: int) -> str:
    """
    Build a chat hostory from the given number of conversation pairs.

    Args:
        messages (list): List of messages in the chat history.
        num_pairs (int): Number of conversation pairs to include in the context.

    Returns:
        str: Formatted string representing the chat history.
    """
    if messages == []:
        return ""
    
    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            user_msg = messages[i]["content"]
            assistant_msg = messages[i + 1]["content"]
            pairs.append((user_msg, assistant_msg))
            i += 2
        else:
            i += 1

    selected_pairs = pairs[-num_pairs:]

    context_parts = []
    for idx, (user_msg, assistant_msg) in enumerate(selected_pairs, start=1):
        block = (
            f"Conversation Pair {idx}\n"
            f"{'-'*30}\n"
            f"**User:** {user_msg}\n\n"
            f"**Assistant:** {assistant_msg}\n"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts).strip()
