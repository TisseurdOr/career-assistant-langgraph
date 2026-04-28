import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi


# Load environment variables from .env so we can read API keys safely.
load_dotenv()


def get_llm(model_name: str = "qwen-plus", temperature: float = 0.5) -> ChatTongyi:
    """Create and return a Tongyi (Qwen) chat model instance."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY in environment (.env).")
    return ChatTongyi(model=model_name, api_key=api_key, temperature=temperature)

