Here is the updated README content with the URL you provided added under the **Vector Store** section.

-----

### Rag LLM: Working with Docs/PDFs in Python

This project demonstrates a basic **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** and a **local LLM (Ollama)**. The workflow processes local PDF documents, creates embeddings, stores them in a vector database, and uses them to answer questions.

-----

### **Project Setup**

First, install the necessary libraries. You will need to have **Ollama** installed and running with your chosen model (`llama3` or `qwen2.5`) on `http://localhost:11434`.

```bash
!pip install langchain-community
!pip install unstructured
!pip install pdfminer
!pip install "unstructured[all-docs]"
!pip install chromadb
!pip install langchain-ollama
```

-----

### **1. Read PDF Documents**

This script reads all PDF files from a specified directory (`./Pdf`) and loads them into a list of documents.

```python
import os
from langchain_community.document_loaders import UnstructuredFileLoader

Docs = "./Pdf"
documents = []
for file in os.listdir(Docs):
    filepath = os.path.join(Docs, file)
    print(filepath)
    loader = UnstructuredFileLoader(filepath)
    documents.extend(loader.load())
```

-----

### **2. Chunking Documents**

Documents are split into smaller, manageable chunks to improve the efficiency and relevance of the RAG process.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}")
    print(chunk.page_content)
    print("-" * 40)
```

-----

### **3. Embedding & Vector Store**

The chunks are converted into numerical embeddings using a local Ollama model (`nomic-embed-text`) and stored in a persistent ChromaDB vector store. This allows for semantic search.

For more information on LangChain's vector store integrations, you can refer to the official documentation: [https://python.langchain.com/docs/integrations/vectorstores/](https://python.langchain.com/docs/integrations/vectorstores/)

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embedding, persist_directory="qa_db")
retriever = db.as_retriever(search_kwargs={"k": 3})
```

-----

### **4. Connect to Local LLM**

This connects to the local Ollama server and initializes a language model to answer questions based on the retrieved context.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3:latest", # Or your chosen model
    temperature=0.5
)
```

-----

### **5. Retrieval QA Chain**

The `RetrievalQA` chain connects the LLM with the vector store's retriever. When a question is asked, it first retrieves the most relevant chunks from the database and then uses the LLM to generate a final answer.

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
question = "Explin PlayOwnAi ID & Name Card?"
print(qa_chain.invoke(question))
```
