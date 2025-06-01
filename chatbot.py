from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
import os
from typing import List, Dict, Optional, Any
from uuid import uuid4

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(title="AI Training Chatbot API")

# Add CORS middleware for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Angular app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "API_KEY_HERE")

# Initialize vector store and retrievers
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="chatbot",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

llm_compression = ChatGroq(
    temperature=0,
    api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)
compressor_llm = LLMChainExtractor.from_llm(llm_compression)
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_llm,
    base_retriever=base_retriever
)

# LLM for responses
llm_response = ChatGroq(
    temperature=0,
    api_key=GROQ_API_KEY,
    # model="llama-3.3-70b-versatile"
    model="llama3-70b-8192"
)

# Update prompt template for better formatting
rag_template = """\
Rôle :
Tu es un expert en formation et en formation assistée par l'IA.

Mission :
Aider l'utilisateur à comprendre et à utiliser efficacement une plateforme de formation basée sur l'IA, ainsi qu'à répondre à toute question générale sur la formation assistée par l'IA.

Historique :
{history}

Connaissances :
{context}

Question :
{question}

Consignes :
Répondre uniquement en utilisant les informations disponibles et des donnés general selon la conversation.

En cas de question hors sujet, répondre :
"Je n'ai pas cette information. Voici quelques questions sur lesquelles je peux vous aider :
    -Comment créer une formation assistée par l'IA ?
    -Comment l'IA améliore-t-elle les processus de formation ?
    -Quels sont les avantages de l'apprentissage assisté par l'IA ?
    -Quelles sont les meilleures pratiques pour intégrer l'IA dans la formation ?"_

Adopter un ton professionnel, clair et précis.


Format attendu :
Réponse structurée avec des informations factuelles et bien organisées et adaptable sur la language de conversation.
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

# In-memory conversation store per session
_memories: dict[str, ConversationBufferMemory] = {}

# Pydantic models


class Source(BaseModel):
    id: str
    title: str
    preview: str
    type: str = "Document"
    location: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str  # Keep this as 'response' for Angular
    session_id: str
    sources: Optional[List[Source]] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = None

# Helpers


def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _memories:
        _memories[session_id] = ConversationBufferMemory(
            memory_key="history", return_messages=False)
    return _memories[session_id]


def retrieve_context_and_sources(query: str):
    docs = advanced_retriever.get_relevant_documents(query)
    context = "\n\n".join([f"- {doc.page_content}" for doc in docs])

    # Extract sources
    sources = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        file_name = metadata.get("source", "").split(
            "/")[-1] if "source" in metadata else f"Document {i+1}"
        doc_type = "Document"

        # Try to identify document type from metadata or file extension
        if "type" in metadata:
            doc_type = metadata["type"]
        elif "source" in metadata:
            file_ext = os.path.splitext(metadata["source"])[1].lower()
            if file_ext == ".pdf":
                doc_type = "PDF Document"
            elif file_ext == ".docx":
                doc_type = "Word Document"
            elif file_ext == ".txt":
                doc_type = "Text File"
            elif file_ext in [".csv", ".xlsx"]:
                doc_type = "Data File"

        source = Source(
            id=str(uuid4()),
            title=file_name,
            preview=doc.page_content[:150] +
            "..." if len(doc.page_content) > 150 else doc.page_content,
            type=doc_type,
            location=f"Page {metadata.get('page', 'Unknown')}" if "page" in metadata else "Section"
        )
        sources.append(source)

    return context, sources

# API Endpoints


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Use provided session ID or generate default
    sid = req.session_id or str(uuid4())
    memory = get_memory(sid)
    history = memory.buffer

    try:
        # Get context and sources
        context, sources = retrieve_context_and_sources(req.message)

        # Format prompt
        prompt_text = rag_prompt.format(
            history=history,
            context=context,
            question=req.message
        )

        # Generate response
        res = llm_response([HumanMessage(content=prompt_text)])
        answer = res.content

        # Save conversation context
        memory.save_context({"input": req.message}, {"output": answer})

        # Return response with session ID and sources
        return ChatResponse(
            response=answer,
            session_id=sid,
            sources=sources if sources else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(req: ResetRequest):
    sid = req.session_id or "default"
    _memories.pop(sid, None)
    return {"status": "memory reset", "session_id": sid}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        collection = vector_store._collection
        doc_count = collection.count()
        return {
            "status": "healthy",
            "document_count": doc_count,
            "model": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
def root():
    return {
        "message": "AI Training Chatbot API is running.",
        "status": "online",
        "endpoints": ["/chat", "/reset", "/health"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
