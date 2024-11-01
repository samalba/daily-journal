from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition, ToolNode


class Agent:

    SystemPrompt = "You are a helpful assistant."

    def __init__(self, openai_api_key: str):
        self._tools = [

        ]
        model = ChatOpenAI(model="gpt-4o", api_key=SecretStr(openai_api_key))
        self._model = model.bind_tools(self._tools)

    def _call_model(self, state: MessagesState) -> dict:
        messages = state["messages"]
        response = self._model.invoke(messages)
        return {"messages": [response]}

    def _compile_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            # Loop to tools until the last message is not a tool call (in that case, route to END)
            tools_condition,
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def ask(self, prompt: str) -> str:
        inputs = {"messages": [
            SystemMessage(content=Agent.SystemPrompt),
            HumanMessage(content=prompt),
        ]}

        app = self._compile_workflow()
        messages = app.invoke(inputs)

        return messages['messages'][-1].content
