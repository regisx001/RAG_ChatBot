# Domain-Specific Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a chatbot tailored for a specific domain using the Retrieval-Augmented Generation (RAG) method. The chatbot leverages advanced natural language processing (NLP) techniques and integrates various tools and libraries for document retrieval, conversational memory, and language modeling.

## Features

- **PDF Document Processing**: Load and process domain-specific PDF documents.
- **Text Chunking**: Split large documents into manageable chunks for better context retrieval.
- **Embeddings**: Utilize Ollama embeddings for text representation.
- **Contextual Compression**: Compress and retrieve relevant context using advanced algorithms.
- **Conversational Memory**: Maintain conversation history for context-aware responses.
- **Custom RAG Prompt**: Tailored prompts for generating domain-specific, professional, and informative responses.
- **Web Interface**: Interactive chatbot interface built using Gradio.

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/regisx001/rag_training
cd rag_training
pip install -r requirements.txt
```

---

## Usage

### 1. Load Domain-Specific Documents
Ensure your domain-specific documents, such as PDFs, are available in the project directory. For example, `document01.pdf` is loaded and processed in this project.

### 2. Run the Chatbot
Launch the chatbot interface using the following command:

```bash
python bot.ipynb
```

You can interact with the chatbot via the Gradio web interface. 

- Default URL: `http://127.0.0.1:7861`

---

## Key Components
### Libraries and Tools
The project utilizes the following key libraries and tools:
- **LangChain**: For creating and managing the RAG pipeline.
- **Ollama**: For embedding generation.
- **Gradio**: For building the chatbot frontend.
- **PyPDFLoader**: For loading and parsing PDF documents.
- **InMemoryVectorStore**: For vector-based text storage.
- **ChatGroq**: For generating language model responses.

### RAG Prompt Template
The chatbot uses a custom RAG template to generate responses:
```text
Rôle: Expert in analyzing reports.
Mission: Provide improvements for reports.

Historique:
{history}

Connaissances:
{context}

Question:
{question}

Consignes:
- Respond only with available information.
- If out of scope, reply: "Je n'ai pas cette information. Voulez-vous une question sur les tendances énergétiques marocaines ?"
- Maintain a professional and clear tone.

Format:
Clear structure with factual information.
```

---

## File Structure

- **`bot.ipynb`**: Main Jupyter Notebook containing the chatbot implementation.
- **`requirements.txt`**: List of required Python libraries and dependencies.

---

## Requirements

- **Python 3.8+**
- **API Keys**: Ensure you have valid API keys for ChatGroq and Ollama embeddings.

---

## Example Interaction

1. **User**: "What are the insights from the report?"
2. **Chatbot**: Provides a detailed response based on the retrieved context and conversation history.

---

## Notes

- The chatbot is optimized for a specific domain and may not perform as well outside its intended context.
- To customize the chatbot for another domain, replace the input documents and adjust the RAG prompt accordingly.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Acknowledgments

Special thanks to the developers of LangChain, Gradio, and other libraries used in this project.
