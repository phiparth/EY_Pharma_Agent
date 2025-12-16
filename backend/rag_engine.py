import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def ingest(self, file_path: str):
        """Loads PDF and creates vector index."""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        self.vector_store = FAISS.from_documents(pages, self.embeddings)
        return True

    def query(self, query_text: str):
        if not self.vector_store:
            return "No internal documents uploaded."
        
        docs = self.vector_store.similarity_search(query_text, k=3)
        return "\n".join([d.page_content for d in docs])

# Singleton instance
rag_system = RAGSystem()
