from typing import Dict

from graph import build_app


# Compatibility entry module:
# the original notebook used a single agents.py file and called run_user_query().
# We keep the same public function while delegating implementation to split modules.
app = build_app()


def run_user_query(query: str) -> Dict[str, str]:
    """Public entrypoint kept for compatibility with the original notebook script."""
    results = app.invoke({"query": query})
    return {
        "category": results.get("category", ""),
        "response": results.get("response", ""),
    }