from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import get_llm
from services import InterviewService, JobSearchService, LearningResourceService, ResumeService
from state import State
from utils import show_md_file


# ============================ LangGraph Nodes =============================
# These node functions map user requests into feature handlers.
# Flow:
# 1) categorize -> top-level route
# 2) optional sub-categorize (learning/interview)
# 3) execute terminal handler node that generates markdown output
# ========================================================================

llm = get_llm("qwen-plus")


def categorize(state: State) -> State:
    """Classify query into 1/2/3/4 main categories."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize query into one number only:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n\n"
        "Query: {query}"
    )
    chain = prompt | llm
    return {"category": chain.invoke({"query": state["query"]}).content}


def handle_learning_resource(state: State) -> State:
    """Further classify learning queries into Tutorial vs Question."""
    prompt = ChatPromptTemplate.from_template(
        "Classify as Tutorial or Question.\nQuery: {query}\n"
        "Return one word only: Tutorial or Question."
    )
    chain = prompt | llm
    return {"category": chain.invoke({"query": state["query"]}).content}


def handle_interview_preparation(state: State) -> State:
    """Further classify interview queries into Mock vs Question."""
    prompt = ChatPromptTemplate.from_template(
        "Classify as Mock or Question.\nQuery: {query}\n"
        "Return one word only: Mock or Question."
    )
    chain = prompt | llm
    return {"category": chain.invoke({"query": state["query"]}).content}


def job_search(state: State) -> State:
    """Run job-search workflow and return saved markdown path."""
    prompt = ChatPromptTemplate.from_template(
        "Refactor this job search content into a clear markdown list.\nContent: {result}"
    )
    service = JobSearchService(prompt)
    query = input("Please provide job role and location:\n")
    path = service.find_jobs(query)
    show_md_file(path)
    return {"response": path}


def handle_resume_making(state: State) -> State:
    """Run resume generation workflow."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a resume expert for AI/GenAI roles. Ask details in 4-5 steps and produce markdown resume.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    service = ResumeService(prompt)
    path = service.create_resume(state["query"])
    show_md_file(path)
    return {"response": path}


def ask_query_bot(state: State) -> State:
    """Run interactive Q&A workflow for GenAI learning."""
    prompt = [
        SystemMessage(
            content=(
                "You are an expert Generative AI Engineer helping users through back-and-forth Q&A."
            )
        )
    ]
    service = LearningResourceService(prompt)
    path = service.query_bot(state["query"])
    show_md_file(path)
    return {"response": path}


def tutorial_agent(state: State) -> State:
    """Generate tutorial blog in markdown format."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior GenAI developer and tutorial blogger. Produce clear markdown tutorial with code examples.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    service = LearningResourceService(prompt)
    path = service.tutorial(state["query"])
    show_md_file(path)
    return {"response": path}


def interview_topics_questions(state: State) -> State:
    """Generate interview question bank with references."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a researcher for GenAI interview prep. Provide top interview questions with references.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    service = InterviewService(prompt)
    path = service.interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}


def mock_interview(state: State) -> State:
    """Run interactive mock interview and return transcript file path."""
    prompt = [
        SystemMessage(
            content="You are a Generative AI interviewer. Conduct a mock interview then provide evaluation."
        )
    ]
    service = InterviewService(prompt)
    path = service.mock_interview()
    show_md_file(path)
    return {"response": path}

