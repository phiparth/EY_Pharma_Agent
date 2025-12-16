import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None

    def setup(self, api_key: str):
        """Initializes the embedding model with the secret key."""
        if not api_key:
            print("Error: No API Key found in secrets.")
            return
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def load_directory(self, directory_path: str = "data"):
        """Automatically scans and ingests all PDFs in the data folder."""
        if not self.embeddings:
            return # Wait for setup

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return # Nothing to load yet

        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            return

        all_pages = []
        print(f"DEBUG: Found {len(pdf_files)} internal documents. Indexing...")
        
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if all_pages:
            self.vector_store = FAISS.from_documents(all_pages, self.embeddings)
            print("DEBUG: Internal Knowledge Base Built.")

    def query(self, query_text: str):
        if not self.vector_store:
            return "No internal strategy documents found in 'data/' folder."
        
        try:
            docs = self.vector_store.similarity_search(query_text, k=3)
            return "\n".join([d.page_content for d in docs])
        except Exception:
            return "Error querying internal knowledge base."

rag_system = RAGSystem()
