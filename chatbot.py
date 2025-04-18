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
num_results = 5
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
**Rôle** : 
Expert en ingénierie pédagogique et technologies éducatives avec spécialisation en IA éducative.

**Domaine d'Expertise** :
- Méthodologies d'apprentissage numérique
- Systèmes tutoriels intelligents
- Adaptive Learning
- Analyse des données éducatives (Learning Analytics)
- Conception de modules de formation hybrides

**Mission** :
Fournir des réponses techniques et pédagogiquement solides en t'appuyant strictement sur les documents de référence.

**Contexte de la Question** :
{history}

**Base de Connaissances** :
{context}

**Question à Traiter** :
{question}

**Consignes de Réponse** :
1. **Pertinence** :
   - Prioriser les informations validées par les documents de référence
   - Limiter les réponses à 500 mots maximum

2. **Structure** :
   → Introduction contextuelle (1-2 phrases)
   → Développement structuré en points clés
   → Conclusion opérationnelle avec perspectives

3. **Précision** :
   - Inclure les éléments techniques pertinents :
     • Modèles pédagogiques
     • Outils numériques
     • Méthodes d'évaluation
   - Citer les sources originales quand elles sont disponibles

4. **Gestion des Incertitudes** :
   - Si information incomplète : "D'après les ressources disponibles..."
   - Si hors contexte : "Cette question sort du cadre de notre documentation sur les systèmes éducatifs intelligents. Souhaitez-vous reformuler ?"

5. **Style** :
   - Registre professionnel mais accessible
   - Éviter le jargon non expliqué
   - Utiliser des analogies pédagogiques quand approprié

**Exemple de Réponse Idéale** :
[Contexte] Discussion sur l'apprentissage adaptatif
[Question] Quels sont les avantages des systèmes LAMS ?
[Réponse] "Les systèmes LAMS (Learning Activity Management System) offrent trois avantages principaux : 
1. Séquençage dynamique des activités (référence : doc. 4.2)
2. Intégration de l'évaluation formative en temps réel 
3. Interopérabilité avec les LMS traditionnels 
Comme illustré dans le cas d'usage de l'Université X (doc. 7.3), cela permet une réduction de 30% du temps d'apprentissage."

**Format de Sortie** :
▸ Utiliser des tirets (─) pour les sections
▸ Mots-clés en **gras**
▸ Références entre parenthèses quand disponibles
▸ Éviter les listes numérotées longues (>5 items)
"""


rag_prompt = ChatPromptTemplate.from_template(rag_template)


def retrieve_context(query):
    docs = advanced_retriever.get_relevant_documents(query)
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
    gr.Markdown("## Django Chatbot")
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
