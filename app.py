import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import asyncio

from astream_events_handler import invoke_our_graph
from conf import cards, categories
from util import check_password

load_dotenv()

st.title("AAC ChatBot")
st.markdown("This example allows to fit AAC symbols for your needs")

if not check_password():
    st.stop()

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

category_actions = {}
for path, data in cards.items():
    category = data["category"]
    action = data["keywords"]
    target = data["target"]
    if category not in category_actions:
        category_actions[category] = set()
    category_actions[category].add(action + " " + target)

cats = ""
for category, description in categories.items():
    actions = ", ".join(sorted(category_actions.get(category, [])))
    cats += f" - **{category}**: {description}"
    if actions:
        cats += f", available keywords: {actions}\n"

init = f"""
You are an assistive communication tool that helps users find specific visual communication cards for individuals with communication needs, especially for children and adults with conditions such as autism. Based on the user’s request, your task is to construct a relevant set of cards from predefined .

### Instructions
1. **Categories Available**:
   {cats}

2. **Identify the Target Group**: Determine if the user's query specifies a particular target group, such as "kids" or "adults". If the target group is specified, only suggest cards relevant to that group, otherwise specify "adults" by default.

3. **Use the `retrieve_similar_documents` Tool**: Compile user query based on the known keywords in **Categories Available** and target group (if identified) and send to the `retrieve_similar_documents` tool to retrieve the most relevant cards.

4. **Respond with Relevant Card Groups**: Based on the tool's output, suggest categories and keywords that align with the user's needs. Include examples or specific card groups that might help them with particular activities or communication goals. Display card image as thumbnail {{thumbnail}} under link {{path}} and card to contain only category and keyword.

"""

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content=init), AIMessage(content="How can I help you?")]



# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a placeholder container for streaming and any other events to visually render here
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))
