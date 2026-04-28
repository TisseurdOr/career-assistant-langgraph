import os
from datetime import datetime
from typing import List

from langchain_core.messages import trim_messages


def trim_conversation(messages: List) -> List:
    """Keep the latest conversation turns to control prompt size."""
    return trim_messages(
        messages,
        max_tokens=10,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


def save_file(data: str, filename_prefix: str) -> str:
    """Save markdown output under Agent_output with timestamped filename."""
    folder_name = "Agent_output"
    os.makedirs(folder_name, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.md"
    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
    return file_path


def show_md_file(file_path: str) -> None:
    # In notebook mode this can be replaced by Markdown rendering.
    # For script/CLI mode we only print the path.
    print(f"Markdown saved at: {file_path}")

