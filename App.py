import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set API key from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Enable LangSmith tracing for debugging (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"  

# Initialize Groq Model with Streaming Enabled
model = ChatGroq(model_name="mixtral-8x7b-32768", streaming=True, max_completion_tokens=1000)

# Streamlit UI
st.title("🤖 Chatbot 🤖")
st.write("Ask me anything!")

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Add user input to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Define prompt template
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a fun, supportive, and witty friend. Talk casually, crack jokes sometimes, and keep the conversation engaging."),
    ("user", "{question}")
    ])

    # Create chain
    chain = prompt_template | model | StrOutputParser()

    # Display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()  # Placeholder for streaming text
        full_response = ""

        # Stream response as it is generated
        for chunk in chain.stream({"question": prompt}):
            full_response += chunk
            response_container.markdown(full_response)  # Update response in real-time

    # Save assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
