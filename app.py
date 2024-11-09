import streamlit as st
from chromadb import Client, Settings
import os

class SimpleRAG:
    def __init__(self):
        """Initialize the RAG system using ChromaDB with default embeddings"""
        self.client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db"
        ))
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection("documents")
        except ValueError:
            self.collection = self.client.get_collection("documents")
        
    def add_documents(self, documents: list[str]) -> None:
        """Add documents to the collection"""
        # Reset collection
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection("documents")
        except ValueError:
            pass
        
        # Add documents with simple IDs
        if documents:
            self.collection.add(
                documents=documents,
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
    
    def query(self, question: str, n_results: int = 3) -> list[str]:
        """Query the documents"""
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=min(n_results, len(self.collection.get()['documents']))
            )
            return results['documents'][0]
        except Exception:
            return []

# Set page config
st.set_page_config(page_title="Simple RAG System", page_icon="📚")

# Initialize RAG system
@st.cache_resource
def get_rag():
    return SimpleRAG()

rag = get_rag()

# Main interface
st.title("📚 Simple RAG System")

# Document input
st.header("Add Documents")
doc_input = st.text_area(
    "Enter your documents (separate paragraphs with blank lines):",
    height=200
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Process Documents"):
        if doc_input.strip():
            with st.spinner("Processing..."):
                docs = [d.strip() for d in doc_input.split('\n\n') if d.strip()]
                rag.add_documents(docs)
                st.success("Documents processed!")
        else:
            st.error("Please enter some documents first")

with col2:
    if st.button("Load Sample Data"):
        sample_docs = [
            "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and released in 1991.",
            "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
            "Python is widely used in data science, machine learning, web development, and automation.",
            "Python's package manager pip makes it easy to install and manage third-party packages."
        ]
        rag.add_documents(sample_docs)
        st.success("Sample data loaded!")

# Query interface
st.header("Ask Questions")
query = st.text_input("Enter your question:")
n_results = st.slider("Number of results:", 1, 5, 3)

if st.button("Search"):
    if query.strip():
        results = rag.query(query, n_results)
        if results:
            st.subheader("Results:")
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}"):
                    st.write(result)
        else:
            st.info("No results found. Try adding some documents first.")
    else:
        st.error("Please enter a question")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. **Add documents:**
        - Paste text and click "Process Documents", or
        - Click "Load Sample Data" for examples
    2. **Ask questions:**
        - Type your question
        - Click "Search"
    3. **View results:**
        - Click each result to expand
    """)
