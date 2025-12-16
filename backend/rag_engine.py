import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None

    def setup(self, api_key: str):
        if not api_key: return
        try:
            # Using standard small embeddings model
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {e}")

    def load_directory(self, directory_path: str = "data"):
        if not self.embeddings: return 
        if not os.path.exists(directory_path): os.makedirs(directory_path); return 

        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files: return

        all_pages = []
        print(f"DEBUG: Indexing {len(pdf_files)} PDF files...")
        
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if all_pages:
            try:
                self.vector_store = FAISS.from_documents(all_pages, self.embeddings)
                print("DEBUG: Internal Knowledge Base Built.")
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

rag_system = RAGSystem()
