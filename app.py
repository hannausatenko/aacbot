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
st.markdown("""Hi! Let me help you select the DyvoGra symbol for alternative and augmentative communication (AAC).
This will assist you with DImobi app usage.
Please enter a few words about the topic or situation in which you need graphic symbols for communication.
""")

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
Based on the user’s request, your task is to construct a relevant set of cards that may help the user to communicate with that person.

### Instructions

1. **Categories Available**:
Here is the list of all available categories. Each category contains communication cards designed for specific target groups:
{cats}

2. **Identify the Target Group**:
   - Determine a particular target group, such as "kids" or "adults".
   - If it is not clear which target group is requested - ask user.
   
3. **Identify the Idea**:
   - Try to identify the main idea, which cards will be suitable to communicate with the person.
   - Think about suitable phrases constructed with several cards, e.g. "I love you", "I do not want milk", "Stay home", "I want to a toilet".

4. **Use the `retrieve_cards` Tool**:
   - Considering the available categories and keywords in **Categories Available**,
 construct one or several consecutive queries for the `retrieve_cards` tool to find the most relevant cards that match the user’s request,
 split complex queries to different consecutive queries,
 each query MUST include the identified target group (see Identify the Target Group).
   - Examples of a query:
      - I, communication, toilet (kids)
      - need, surprise, paint (adults)

5. **Respond with Relevant Card Groups**:
   - Based on the tool's output, suggest categories and card groups that align with the user’s needs.
   - Provide examples or specific card groups that may aid in particular activities or communication goals.
   - Use the only cards that were responded by the tool, do not create your own.

6. **Response format**:
It has to be strictly in JSONL format.
""" + """
{"group": "I want candy", "cards": [
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "I"},
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "want"},
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "candy"}
]}
{"group": "Get calm", "cards": [
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "get calm"}
]}
{"group": "I need rest", "cards": [
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "I"},
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "want"},
    {"url": "{url}", "thumbnail": "{thumbnail}", "name": "rest"}
]}
"""

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content=init), AIMessage(content="""
    How can I help you? Examples:
    Suggest me cards to communicate with my grandmother for her basic needs
    """)]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").html(msg.content)
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
