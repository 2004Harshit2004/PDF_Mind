from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import os
import getpass
from dotenv import load_dotenv
load_dotenv()

google_api_key=os.getenv('GEMINI_API_KEY')

# Initialization of embedding method
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

# Initialization of vector store
vector_store = Chroma(
    collection_name="sample1",
    embedding_function=embeddings,
    persist_directory="chroma_db",

)


system_prompt = (
    "You are a helpful assistant that answers questions based on the context "
    "provided. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise and to the point."
    "\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key
)

retriever = vector_store.as_retriever()

# Create a "stuff" chain to combine the retrieved documents and the prompt
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the full retrieval chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Initialize the chat message history
msgs = StreamlitChatMessageHistory(key="messages")

# Wrap the retrieval chain with message history
chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="history",
)



st.title("PDF Mind 🧠")
st.write("Ask me anything about your PDF!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What do you want to know?"):
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        # The session ID is not critical for a single-user app, but is required by the Runnable
        config = {"configurable": {"session_id": "any_session"}}
        response = chain_with_history.invoke({"input": prompt}, config=config)

    st.markdown(response["answer"])
    # Add AI response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})