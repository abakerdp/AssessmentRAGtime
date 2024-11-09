import streamlit as st
import chromadb
import os
import tempfile

# Force chromadb to use HTTP client
os.environ["CHROMADB_CLIENT"] = "rest"

class SimpleRAG:
    def __init__(self):
        """Initialize the RAG system using ChromaDB with default embeddings"""
        # Create a temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        # Initialize client with persistent directory
        self.client = chromadb.PersistentClient(path=self.temp_dir)
        
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
                n_results=min(n_results, self.collection.count())
            )
            return results['documents'][0]
        except Exception as e:
            if "Found no documents" in str(e):
                return []
            raise e

# Set page config
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize RAG system
@st.cache_resource
def get_rag():
    return SimpleRAG()

rag = get_rag()

# Add custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 2])

# Sidebar (Document Input)
with col1:
    st.header("📝 Add Documents")
    doc_input = st.text_area(
        "Enter your documents here:",
        height=300,
        placeholder="Paste your text here... Separate different documents with blank lines."
    )
    
    if st.button("🔄 Process Documents"):
        if doc_input.strip():
            with st.spinner("Processing documents..."):
                # Split into paragraphs and filter empty ones
                docs = [d.strip() for d in doc_input.split('\n\n') if d.strip()]
                rag.add_documents(docs)
                st.success("✅ Documents processed successfully!")
        else:
            st.error("⚠️ Please enter some documents first")

    # Sample data button
    if st.button("📚 Load Sample Data"):
        sample_docs = [
            "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and released in 1991.",
            "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a large standard library.",
            "Python is widely used in data science, machine learning, web development, and automation. Many popular frameworks like Django and Flask are written in Python.",
            "Python's package manager pip makes it easy to install and manage third-party packages. The Python Package Index (PyPI) hosts millions of projects."
        ]
        with st.spinner("Loading sample data..."):
            rag.add_documents(sample_docs)
            st.success("✅ Sample data loaded!")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Add documents:**
           - Paste text in the box above, or
           - Click "Load Sample Data"
        2. **Process:**
           - Click "Process Documents"
        3. **Query:**
           - Type your question
           - Adjust number of results
           - Click "Search"
        """)

# Main area (Query Interface)
with col2:
    st.header("🔍 Ask Questions")
    query = st.text_input("Enter your question:", placeholder="What would you like to know?")
    n_results = st.slider("Number of results:", 1, 5, 3)

    if st.button("🔎 Search"):
        if not query.strip():
            st.error("⚠️ Please enter a question")
        else:
            with st.spinner("Searching..."):
                results = rag.query(query, n_results)
                
            if results:
                st.subheader("📊 Results:")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}"):
                        st.markdown(f"```\n{result}\n```")
            else:
                st.warning("ℹ️ No results found. Try adding some documents first or rephrase your question.")

    # Show some example questions
    with st.expander("💡 Example questions (with sample data)"):
        st.markdown("""
        - When was Python created?
        - What is Python used for?
        - How does Python package management work?
        - What programming paradigms does Python support?
        """)
