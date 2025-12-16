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
            print("Error: No API Key found.")
            return
        
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception as e:
            print(f"Error initializing embeddings: {e}")

    def load_directory(self, directory_path: str = "data"):
        """Automatically scans and ingests all PDFs in the data folder."""
        # Safety check: if embeddings weren't set up, we can't load.
        if not self.embeddings:
            print("RAG Setup skipped: No embeddings initialized.")
            return 

        # Create data directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            return 

        # Find all PDF files
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            print("No PDFs found in data directory.")
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
            try:
                self.vector_store = FAISS.from_documents(all_pages, self.embeddings)
                print("DEBUG: Internal Knowledge Base Built successfully.")
            except Exception as e:
                print(f"Error building vector store: {e}")

    def query(self, query_text: str):
        if not self.vector_store:
            return "No internal strategy documents loaded."
        
        try:
            docs = self.vector_store.similarity_search(query_text, k=3)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Error querying internal knowledge base: {e}"

# Create the instance
rag_system = RAGSystem()
