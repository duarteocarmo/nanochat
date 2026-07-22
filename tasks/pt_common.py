"""Shared helpers for European Portuguese conversation tasks."""


def render_portuguese_mc(*, question, letters, choices):
    prompt = f"Pergunta de escolha múltipla: {question}\n"
    prompt += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    prompt += "\nResponde apenas com a letra da resposta correta."
    return prompt


def normalize_conversation(messages):
    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
    }
    normalized = []
    for message in messages:
        source_role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        normalized.append({"role": role_map.get(source_role, source_role), "content": content})

    alternating_messages = normalized
    if normalized and normalized[0]["role"] == "system":
        if not isinstance(normalized[0]["content"], str):
            raise ValueError("System message content must be a string")
        alternating_messages = normalized[1:]
    if len(alternating_messages) < 2:
        raise ValueError("Conversation must contain at least one user-assistant turn")
    for index, message in enumerate(alternating_messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        valid_content = isinstance(message["content"], str) if expected_role == "user" else isinstance(message["content"], (str, list))
        if message["role"] != expected_role or not valid_content:
            raise ValueError(f"Invalid message at position {index}: expected {expected_role}")
    return {"messages": normalized}
