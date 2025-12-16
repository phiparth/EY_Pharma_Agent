import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None  # Don't initialize yet!

    def setup(self, api_key: str):
        """Initializes the embedding model with the provided API key."""
        if not api_key:
            raise ValueError("API Key is required to initialize RAG.")
        
        # Initialize now that we have the key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def ingest(self, file_path: str):
        """Loads PDF and creates vector index."""
        if not self.embeddings:
            return "Error: RAG System not initialized. Please provide API Key first."

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Create vector store using the initialized embeddings
        self.vector_store = FAISS.from_documents(pages, self.embeddings)
        return True

    def query(self, query_text: str):
        if not self.embeddings:
            return "RAG System not initialized."
        
        if not self.vector_store:
            return "No internal documents uploaded."
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(query_text, k=3)
        return "\n".join([d.page_content for d in docs])

# Create the instance, but it's "empty" until setup() is called in app.py
rag_system = RAGSystem()
