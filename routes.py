from state import State


# ============================ Route Functions =============================
# route_query: top-level branch selection from category id (1~4)
# route_interview: interview sub-route (question bank vs mock)
# route_learning: learning sub-route (tutorial vs Q&A)
# ========================================================================

def route_query(state: State):
    category = (state.get("category") or "").lower()
    if "1" in category:
        return "handle_learning_resource"
    if "2" in category:
        return "handle_resume_making"
    if "3" in category:
        return "handle_interview_preparation"
    if "4" in category:
        return "job_search"
    return "handle_learning_resource"


def route_interview(state: State):
    category = (state.get("category") or "").lower()
    if "question" in category:
        return "interview_topics_questions"
    return "mock_interview"


def route_learning(state: State):
    category = (state.get("category") or "").lower()
    if "tutorial" in category:
        return "tutorial_agent"
    return "ask_query_bot"

