import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Avex",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------
# TITLE
# --------------------------------
st.title("🤖 Avex")
st.caption("Your AI Assistant")

# --------------------------------
# LOAD API KEY FROM SECRETS
# --------------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]

# --------------------------------
# LLM MODEL
# --------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# --------------------------------
# PROMPT TEMPLATE
# --------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Avex, a smart, friendly and helpful AI assistant."
        ),
        (
            "user",
            "Question: {question}"
        )
    ]
)

# --------------------------------
# OUTPUT PARSER
# --------------------------------
output_parser = StrOutputParser()

# --------------------------------
# CHAIN
# --------------------------------
chain = prompt | llm | output_parser

# --------------------------------
# CHAT HISTORY
# --------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------
# USER INPUT
# --------------------------------
user_input = st.chat_input("Ask me anything...")

# --------------------------------
# RESPONSE
# --------------------------------
if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = chain.invoke(
        {
            "question": user_input
        }
    )

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)