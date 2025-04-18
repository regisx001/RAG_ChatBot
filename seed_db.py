from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"


def seed_vector_db():
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create vector store
    vector_store = Chroma(
        collection_name="chatbot",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # Load and split documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Generate UUIDs and add to vector store
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    print(f"Successfully added {len(chunks)} documents to vector store")


if __name__ == "__main__":
    seed_vector_db()
