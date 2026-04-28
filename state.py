from typing import TypedDict


# State schema shared by all LangGraph nodes.
# - query: incoming user request
# - category: route/category prediction
# - response: final output payload (here we keep file path)
class State(TypedDict, total=False):
    query: str
    category: str
    response: str

