def format_prompt(system: str, user: str) -> str:
    """Formats text using correct Llama 3 special tokens."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )