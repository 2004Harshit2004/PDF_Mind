from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from tempfile import NamedTemporaryFile

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

## Setting up UI
import streamlit as st
st.set_page_config(page_title="PDF Mind", page_icon="🧠")
st.title("PDF Mind 🧠")
st.write("Upload a PDF and start asking questions!")



## configuration setup
import os
import getpass
from dotenv import load_dotenv
load_dotenv()

hugging_face_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")

google_api_key=os.getenv('GEMINI_API_KEY')



## File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")



@st.cache_resource(show_spinner=False)
def create_vector_store(file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_path = temp_file.name

    with st.spinner("Processing PDF..."):
        # Load the PDF document from the temporary file
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        # Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)


        # Create the embedding model
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hugging_face_api,
        )

        # Create the vector store from chunks
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="uploaded_pdf_collection",
            persist_directory="chroma_db",
        )
        st.success("PDF processed successfully!")
    return vector_store



# Handle file upload
if uploaded_file:
    # Use the cached function to get the vector store
    vector_store = create_vector_store(uploaded_file)
    retriever = vector_store.as_retriever()

    #3. RAG CHAIN SETUP (moved inside the conditional block)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key
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

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 4. CONVERSATIONAL MEMORY & CHAT UI 
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Display existing chat messages
    for message in msgs.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Get user input
    if prompt := st.chat_input("What do you want to know?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any_session"}}
                response = chain_with_history.invoke({"input": prompt}, config=config)

            st.markdown(response["answer"])


