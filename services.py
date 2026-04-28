from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from utils import save_file, trim_conversation


# ============================ Service Layer =============================
# This module contains reusable service classes that encapsulate
# the behavior of each feature area:
# - Learning resources (tutorial + Q&A)
# - Interview prep (question bank + mock interview)
# - Resume creation
# - Job search
# ======================================================================

class LearningResourceService:
    def __init__(self, prompt):
        # Shared model and web-search tool used by tutorial/Q&A features.
        self.model = get_llm("qwen-plus")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def tutorial(self, user_input: str) -> str:
        # Build a tool-calling agent and produce a tutorial markdown file.
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})
        return save_file(str(response.get("output")).replace("```markdown", "").strip(), "Tutorial")

    def query_bot(self, user_input: str) -> str:
        # Back-and-forth Q&A session; keeps latest conversation window only.
        print("\nStarting Q&A session. Type 'exit' to end.\n")
        record = [f"User Query: {user_input}\n"]
        self.prompt.append(HumanMessage(content=user_input))
        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            record.append(f"\nExpert Response: {response.content}\n")
            self.prompt.append(AIMessage(content=response.content))
            print("\nEXPERT:", response.content)
            user_input = input("\nYOUR QUERY: ")
            if user_input.lower() == "exit":
                break
            record.append(f"\nUser Query: {user_input}\n")
            self.prompt.append(HumanMessage(content=user_input))
        return save_file("".join(record), "Q&A_Doubt_Session")


class InterviewService:
    def __init__(self, prompt):
        # Interview service supports both interview-topic generation and mock interview.
        self.model = get_llm("qwen-plus")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def interview_questions(self, user_input: str) -> str:
        # Generate interview question bank iteratively and save markdown.
        chat_history = []
        questions_bank = ""
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        while True:
            if user_input.lower() == "exit":
                break
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            questions_bank += str(response.get("output")).replace("```markdown", "").strip() + "\n"
            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            user_input = input("You: ")
        return save_file(questions_bank, "Interview_questions")

    def mock_interview(self) -> str:
        # Conduct interactive mock interview and persist transcript.
        print("\nStarting mock interview. Type 'exit' to end.\n")
        initial_message = "I am ready for the interview."
        interview_record = [f"Candidate: {initial_message}\n"]
        self.prompt.append(HumanMessage(content=initial_message))
        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            self.prompt.append(AIMessage(content=response.content))
            print("\nInterviewer:", response.content)
            interview_record.append(f"\nInterviewer: {response.content}\n")
            user_input = input("\nCandidate: ")
            interview_record.append(f"\nCandidate: {user_input}\n")
            if user_input.lower() == "exit":
                break
            self.prompt.append(HumanMessage(content=user_input))
        return save_file("".join(interview_record), "Mock_Interview")


class ResumeService:
    def __init__(self, prompt):
        # Resume service runs as a tool-calling agent with chat history.
        self.model = get_llm("qwen-plus")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def create_resume(self, user_input: str) -> str:
        # Collect user details step-by-step and save final resume markdown.
        chat_history = []
        response = {"output": ""}
        while True:
            if user_input.lower() == "exit":
                break
            response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            user_input = input("You: ")
        return save_file(str(response.get("output")).replace("```markdown", "").strip(), "Resume")


class JobSearchService:
    def __init__(self, prompt):
        # Job search uses web search results and model post-processing.
        self.model = get_llm("qwen-plus")
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()

    def find_jobs(self, user_input: str) -> str:
        # Query web, refactor output into readable markdown, and save.
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model
        jobs = chain.invoke({"result": results}).content
        return save_file(str(jobs).replace("```markdown", "").strip(), "Job_search")

