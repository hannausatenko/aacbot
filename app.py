import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import asyncio

from astream_events_handler import invoke_our_graph
from conf import cards, categories, cats
from util import check_password

load_dotenv()

st.title("AAC ChatBot")
st.markdown("Hi! Let me help you select the DyvoGra symbol for alternative and augmentative communication (AAC). This will assist you with DImobi app usage. Please enter a few words about the topic or situation in which you need graphic symbols for communication.")

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

init = f"""
You are an assistive communication tool that helps users find specific visual communication cards for individuals with complex communication needs,
especially for children and adults with disorders such as autism or aphasia.
Based on the user’s request, your task is to construct a relevant set of cards from predefined categories.

### Instructions

1. **Categories Available**:
Here is the list of all available categories. Each category contains communication cards designed for specific target groups:
{cats}

2. **Identify the Target Group**:
   - Determine if the user's query specifies a particular target group, such as "kids" or "adults."
   - If a target group is specified, only suggest cards relevant to that group.

3. **Use the `retrieve_cards` Tool**:
   - Considering the available categories and keywords in **Categories Available**,
 construct a query for the `retrieve_cards` tool to find the most relevant cards that match the user’s request,
 focusing on the specified target group if applicable.

4. **Respond with Relevant Card Groups**:
   - Based on the tool's output, suggest categories and cards that align with the user’s needs.
   - Provide examples or specific card groups that may aid in particular activities or communication goals.
   - Display card images as thumbnails (using {{thumbnail}}) with links to the images ({{path}}).
 Each card should include only the image, category, and keyword.
 Note that some images may have names with extensions like `name.png.png` or `name.svg.png`.

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content=init), AIMessage(content="How can I help you?")]
"""

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
