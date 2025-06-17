from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from uuid import uuid4

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func

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

# Database setup
DB_PATH = os.getenv("DB_PATH", "sqlite:///db.sqlite3")
engine = create_engine(DB_PATH)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    title = Column(String, default="New Conversation")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey(
        "conversations.id", ondelete="CASCADE"))
    role = Column(String)  # "human" or "ai"
    content = Column(Text)
    created_at = Column(DateTime, default=func.now())
    conversation = relationship("Conversation", back_populates="messages")


# Create the tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

# In-memory conversation store per session - keeps recent sessions in memory
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


class MessageModel(BaseModel):
    role: str
    content: str
    created_at: datetime


class ConversationModel(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

# Helpers


def get_memory(session_id: str, db: Session = None) -> ConversationBufferMemory:
    """
    Get conversation memory for a session ID, loading from database if needed
    """
    if session_id not in _memories:
        # Create new memory
        memory = ConversationBufferMemory(
            memory_key="history", return_messages=False)

        # Load existing conversation from database if available
        if db:
            try:
                # Check if conversation exists and load messages
                messages = db.query(Message).join(Conversation).filter(
                    Conversation.id == session_id
                ).order_by(Message.created_at).all()

                # If messages exist, rebuild memory
                for msg in messages:
                    if msg.role == "human":
                        human_msg = msg.content
                        ai_msg = ""

                        # Find the corresponding AI message
                        next_msg = db.query(Message).filter(
                            Message.conversation_id == session_id,
                            Message.created_at > msg.created_at,
                            Message.role == "ai"
                        ).order_by(Message.created_at).first()

                        if next_msg:
                            ai_msg = next_msg.content
                            memory.save_context(
                                {"input": human_msg}, {"output": ai_msg})
            except Exception as e:
                print(f"Error loading conversation history from database: {e}")

        _memories[session_id] = memory

    return _memories[session_id]


def save_message_to_db(db: Session, session_id: str, role: str, content: str):
    """
    Save a message to the database
    """
    try:
        # Check if conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == session_id).first()

        # Create conversation if it doesn't exist
        if not conversation:
            conversation = Conversation(
                id=session_id,
                title=f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            db.add(conversation)
            db.commit()
        else:
            # Update timestamp
            conversation.updated_at = datetime.now()
            db.commit()

        # Add message
        message = Message(
            id=str(uuid4()),
            conversation_id=session_id,
            role=role,
            content=content
        )
        db.add(message)
        db.commit()

        # Update title after first exchange
        message_count = db.query(Message).filter(
            Message.conversation_id == session_id).count()
        if message_count == 2 and role == "ai":
            # Use first user message as conversation title
            first_message = db.query(Message).filter(
                Message.conversation_id == session_id,
                Message.role == "human"
            ).first()
            if first_message:
                title = first_message.content[:50] + \
                    ("..." if len(first_message.content) > 50 else "")
                conversation.title = title
                db.commit()

    except Exception as e:
        db.rollback()
        print(f"Error saving message to database: {e}")


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
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # Use provided session ID or generate default
    sid = req.session_id or str(uuid4())
    memory = get_memory(sid, db)
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

        # Save user message to database
        save_message_to_db(db, sid, "human", req.message)

        # Generate response
        res = llm_response([HumanMessage(content=prompt_text)])
        answer = res.content

        # Save AI response to database
        save_message_to_db(db, sid, "ai", answer)

        # Save conversation context in memory
        memory.save_context({"input": req.message}, {"output": answer})

        # Return response with session ID and sources
        return ChatResponse(
            response=answer,
            session_id=sid,
            sources=sources if sources else None
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(req: ResetRequest, db: Session = Depends(get_db)):
    sid = req.session_id or "default"

    # Clear memory cache
    _memories.pop(sid, None)

    # Don't delete from database by default - just reset the memory
    # To delete completely, use the /conversations/{conversation_id} DELETE endpoint

    return {"status": "memory reset", "session_id": sid}


@app.get("/conversations", response_model=List[ConversationModel])
def list_conversations(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all conversations with pagination"""
    db_conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for conv in db_conversations:
        # Count messages in this conversation
        message_count = db.query(Message).filter(
            Message.conversation_id == conv.id).count()

        result.append(ConversationModel(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=message_count
        ))

    return result


@app.get("/conversations/{conversation_id}", response_model=List[MessageModel])
def get_conversation_messages(
    conversation_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all messages for a specific conversation"""
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        MessageModel(
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at
        )
        for msg in messages
    ]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Clear from memory cache
    _memories.pop(conversation_id, None)

    # Delete from database
    db.delete(conversation)
    db.commit()

    return {"status": "success", "message": f"Conversation {conversation_id} deleted"}


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


@app.get("/conversations/{conversation_id}/html", response_class=HTMLResponse)
def visualize_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Return an HTML visualization of the conversation history"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .all()
    )

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversation: {conversation.title}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
            .message {{ margin-bottom: 20px; padding: 15px; border-radius: 8px; }}
            .human {{ background-color: #e3f2fd; text-align: right; margin-left: 50px; }}
            .ai {{ background-color: #f5f5f5; margin-right: 50px; }}
            .timestamp {{ font-size: 0.8em; color: #666; margin-top: 5px; }}
            h1 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Conversation: {conversation.title}</h1>
        <p>Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    for msg in messages:
        role_class = "human" if msg.role == "human" else "ai"
        role_label = "User" if msg.role == "human" else "AI"
        timestamp = msg.created_at.strftime('%Y-%m-%d %H:%M:%S')

        html_content += f"""
        <div class="message {role_class}">
            <strong>{role_label}:</strong> {msg.content}
            <div class="timestamp">{timestamp}</div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    return html_content


@app.get("/conversations/stats")
def conversation_stats(db: Session = Depends(get_db)):
    """Get statistics about conversations"""
    total_conversations = db.query(Conversation).count()
    total_messages = db.query(Message).count()

    # Get recent activity
    recent_days = 7
    recent_cutoff = datetime.now() - timedelta(days=recent_days)
    recent_conversations = db.query(Conversation).filter(
        Conversation.created_at >= recent_cutoff
    ).count()

    # Get message distribution
    human_messages = db.query(Message).filter(Message.role == "human").count()
    ai_messages = db.query(Message).filter(Message.role == "ai").count()

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "recent_conversations": recent_conversations,
        "message_distribution": {
            "human": human_messages,
            "ai": ai_messages
        }
    }


@app.get("/")
def root():
    return {
        "message": "AI Training Chatbot API is running.",
        "status": "online",
        "endpoints": [
            "/chat",
            "/reset",
            "/health",
            "/conversations",
            "/conversations/{conversation_id}",
            "/conversations/{conversation_id}/html",
            "/conversations/stats"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
