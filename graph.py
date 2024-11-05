from collections import defaultdict
from typing import Annotated, TypedDict, Literal, List, Dict, Optional

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI

from conf import cards

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain


documents = [
    Document(
        page_content=f"Category: {data['category']}, Action: {data['action']}, Target: {data['target']}",
        metadata={"path": path, "category": data["category"], "action": data["action"], "target": data["target"]}
    )
    for path, data in cards.items()
]

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

llm = OpenAI()

db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# query = "I need cards for kids to express yes or no and learn family members."

# docs = new_db.similarity_search_with_score(query)
# res = "\n\n".join([doc[0].page_content for doc in docs])
#
# print(res)


@tool
def retrieve_similar_documents(query: str):
    """
    Retrieve documents from FAISS index that are similar to the query.
    """
    # Perform similarity search
    docs = db.similarity_search_with_score(query, k=10)

    # Combine the content of the top documents
    return [doc[0] for doc in docs]

# Adding the tool to the accessible tools list
tools = [retrieve_similar_documents]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        # specifically for OpenAI we have to set parallel tool call to false
        # because of st primitively visually rendering the tool results
    ).bind_tools(tools, parallel_tool_calls=False)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()