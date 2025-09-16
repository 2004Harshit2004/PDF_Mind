from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile


# cofiguration setup
import getpass
import os
load_dotenv()

hugging_face_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")

google_api_key = os.getenv("GEMINI_API_KEY")

# file_path = "data\How To Win Friends And Influence People - Carnegie, Dale.pdf"

# loader = PyPDFLoader(file_path)
# docs = loader.load() # list of document which contain page content and meta data

# # splitting the documents in the form of chuncks
# splitter= RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )

# # list of chuncks which contain page content and meta data
# chuncks=splitter.split_documents(docs)


# # Initialization of embedding method by Gemini model
# gemini_embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=google_api_key
# )

# # Initialization of embedding method by hugging face model
# huggingface_embedding = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-mpnet-base-v2",
#     task="feature-extraction",
#     huggingfacehub_api_token=hugging_face_api,
# ) 

# # Initialization of vector store
# vector_store = Chroma(
#     collection_name="sample1",
#     embedding_function=huggingface_embedding,
#     persist_directory="chroma_db",
# )
# vector_store.add_documents(chuncks[10:12])
# print(vector_store.get(include=['embeddings']))


def create_vector_store(file,hugging_face_api_func):
    if not hugging_face_api_func:
        raise ValueError("Hugging Face API token is not set.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_path = temp_file.name

    
    # Load the PDF document from the temporary file
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)


    # Create the embedding model
    embeddings = HuggingFaceEndpointEmbeddings(
        # model="sentence-transformers/all-mpnet-base-v2",
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hugging_face_api_func,
    )

    # Create the vector store from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="uploaded_pdf_collection",
    )
    return vector_store



