from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(title="AI Training Chatbot API")

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", "API_KEY_HERE")

# Initialize vector store and retrievers
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="chatbot",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

llm_compression = ChatGroq(
    temperature=0,
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
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
    model="llama-3.3-70b-versatile"
)

# Prompt template
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
    _"Je n'ai pas cette information. Voici quelques questions sur lesquelles je peux vous aider :

    Comment créer une formation assistée par l'IA ?

    Comment l'IA améliore-t-elle les processus de formation ?

    Quels sont les avantages de l'apprentissage assisté par l'IA ?

    Quelles sont les meilleures pratiques pour intégrer l'IA dans la formation ?"_

    Adopter un ton professionnel, clair et précis.

Format attendu :
Réponse structurée avec des informations factuelles et bien organisées et adaptable sur la language de conversation.
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

# In-memory conversation store per session
_memories: dict[str, ConversationBufferMemory] = {}

# Pydantic models


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str


class ResetRequest(BaseModel):
    session_id: str | None = None

# Helpers


def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _memories:
        _memories[session_id] = ConversationBufferMemory(
            memory_key="history", return_messages=False)
    return _memories[session_id]


def retrieve_context(query: str) -> str:
    docs = advanced_retriever.get_relevant_documents(query)
    return "\n\n".join([f"- {d.page_content}" for d in docs])

# API Endpoints


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or "default"
    memory = get_memory(sid)
    history = memory.buffer

    context = retrieve_context(req.message)
    prompt_text = rag_prompt.format(
        history=history,
        context=context,
        question=req.message
    )
    try:
        res = llm_response([HumanMessage(content=prompt_text)])
        answer = res.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    memory.save_context({"input": req.message}, {"output": answer})
    return ChatResponse(response=answer)


@app.post("/reset")
def reset(req: ResetRequest):
    sid = req.session_id or "default"
    _memories.pop(sid, None)
    return {"status": "memory reset", "session_id": sid}


@app.get("/")
def root():
    return {"message": "AI Training Chatbot API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
