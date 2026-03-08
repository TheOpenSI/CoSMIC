def build_context_from_messages(messages: list, num_pairs: int) -> str:
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
            f"Previous Conversation Pair {idx}\n"
            f"{'-'*30}\n"
            f"**User:** {user_msg}\n"
            f"**Assistant:** {assistant_msg}"
        )
        context_parts.append(block)

    full_context = (
        "Conversation History: \n"
        + "\n\n".join(context_parts)
        + "\n"
        + f"{'='*15} End of Chat History {'='*15}\n"
    )

    return full_context



if __name__ == "__main__":
    messages = [
    {'role': 'system', 'content': 'PDF content from OpenwebUI'},
    {'role': 'user', 'content': 'who is Einstein?'},
    {'role': 'assistant', 'content': 'Albert Einstein (1879-1955) was a renowned German-born physicist'},
    {'role': 'user', 'content': 'is he still alive?'},
    {'role': 'assistant', 'content': 'No, he passed away in 1955.'},
    {'role': 'user', 'content': 'what is the theory of relativity?'}
    ]
    
    print(build_context_from_messages(messages, 5))