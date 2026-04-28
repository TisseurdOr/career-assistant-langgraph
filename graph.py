from langgraph.graph import END, START, StateGraph

from nodes import (
    ask_query_bot,
    categorize,
    handle_interview_preparation,
    handle_learning_resource,
    handle_resume_making,
    interview_topics_questions,
    job_search,
    mock_interview,
    tutorial_agent,
)
from routes import route_interview, route_learning, route_query
from state import State


def build_app():
    """
    Build and compile the LangGraph workflow.

    Graph shape:
    START -> categorize
      -> handle_learning_resource -> (tutorial_agent | ask_query_bot) -> END
      -> handle_resume_making -> END
      -> handle_interview_preparation -> (mock_interview | interview_topics_questions) -> END
      -> job_search -> END
    """
    workflow = StateGraph(State)

    # Register all processing nodes.
    workflow.add_node("categorize", categorize)
    workflow.add_node("handle_learning_resource", handle_learning_resource)
    workflow.add_node("handle_resume_making", handle_resume_making)
    workflow.add_node("handle_interview_preparation", handle_interview_preparation)
    workflow.add_node("job_search", job_search)
    workflow.add_node("mock_interview", mock_interview)
    workflow.add_node("interview_topics_questions", interview_topics_questions)
    workflow.add_node("tutorial_agent", tutorial_agent)
    workflow.add_node("ask_query_bot", ask_query_bot)

    # Entry edge.
    workflow.add_edge(START, "categorize")

    # Main route by top-level category.
    workflow.add_conditional_edges(
        "categorize",
        route_query,
        {
            "handle_learning_resource": "handle_learning_resource",
            "handle_resume_making": "handle_resume_making",
            "handle_interview_preparation": "handle_interview_preparation",
            "job_search": "job_search",
        },
    )

    # Sub-route for interview branch.
    workflow.add_conditional_edges(
        "handle_interview_preparation",
        route_interview,
        {
            "mock_interview": "mock_interview",
            "interview_topics_questions": "interview_topics_questions",
        },
    )

    # Sub-route for learning branch.
    workflow.add_conditional_edges(
        "handle_learning_resource",
        route_learning,
        {
            "tutorial_agent": "tutorial_agent",
            "ask_query_bot": "ask_query_bot",
        },
    )

    # Terminal nodes.
    workflow.add_edge("handle_resume_making", END)
    workflow.add_edge("job_search", END)
    workflow.add_edge("interview_topics_questions", END)
    workflow.add_edge("mock_interview", END)
    workflow.add_edge("ask_query_bot", END)
    workflow.add_edge("tutorial_agent", END)
    workflow.set_entry_point("categorize")
    return workflow.compile()

