from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = r"chroma_db"
GROQ_API_KEY = ""  # Add your Groq API key here

# Initialize components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="chatbot",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# Set up retriever and compression
num_results = 10
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

llm_compression = ChatGroq(
    temperature=0, api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
compressor_llm = LLMChainExtractor.from_llm(llm_compression)
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_llm,
    base_retriever=retriever
)

# Memory and prompt setup
memory = ConversationBufferMemory(memory_key="history", return_messages=False)
llm_response = ChatGroq(
    temperature=0, model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

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


def retrieve_context(query):
    docs = advanced_retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content)  # Print each document's content
    return "\n\n".join([f"- {doc.page_content}" for doc in docs])


def generate_response(user_message):
    conversation_history = memory.buffer
    context = retrieve_context(user_message)

    prompt = rag_prompt.format(
        history=conversation_history,
        context=context,
        question=user_message
    )

    response = llm_response([HumanMessage(content=prompt)]).content
    memory.save_context({"input": user_message}, {"output": response})
    return response


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ChatBot : Formation assisté par  IA ")
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="Ask your question...",
            container=False
        )
        send_btn = gr.Button("Send")

    def user_interaction(user_message, history):
        response = generate_response(user_message)
        history.append((user_message, response))
        return "", history

    txt_input.submit(user_interaction, [txt_input, state], [
                     txt_input, chatbot])
    send_btn.click(user_interaction, [txt_input, state], [txt_input, chatbot])

if __name__ == "__main__":
    demo.launch()
