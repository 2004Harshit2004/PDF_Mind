from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import getpass
import os
load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")

file_path = "data\How To Win Friends And Influence People - Carnegie, Dale.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() # list of document which contain page content and meta data

# splitting the documents in the form of chuncks
splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# list of chuncks which contain page content and meta data
chuncks=splitter.split_documents(docs)


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

vector_store.add_documents(chuncks[0:10])
# print(vector_store.get(include=['embeddings','documents','metadatas']))

# vector_store.persist()









