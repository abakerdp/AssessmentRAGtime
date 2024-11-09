import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self):
        """Initialize the RAG system with sentence transformer"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        
    def add_documents(self, documents: list[str]) -> None:
        """Add documents and compute their embeddings"""
        self.documents = documents
        if documents:
            self.embeddings = self.model.encode(documents)
        else:
            self.embeddings = None
            
    def query(self, question: str, n_results: int = 3) -> list[tuple[str, float]]:
        """Query documents and return results with scores"""
        if not self.documents or not self.embeddings:
            return []
        
        # Get question embedding
        q_embedding = self.model.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(q_embedding, self.embeddings)[0]
        
        # Get top results
        n_results = min(n_results, len(self.documents))
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        # Return documents and scores
        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results

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

# Title
st.title("📚 Simple RAG System")

# Create two columns for layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("📝 Add Documents")
    doc_input = st.text_area(
        "Enter your documents here:",
        height=300,
        placeholder="Paste your text here... Separate different documents with blank lines."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process Documents", use_container_width=True):
            if doc_input.strip():
                with st.spinner("Processing..."):
                    docs = [d.strip() for d in doc_input.split('\n\n') if d.strip()]
                    rag.add_documents(docs)
                    st.success("✅ Documents processed!")
            else:
                st.error("Please enter some documents first")
                
    with col2:
        if st.button("Load Sample Data", use_container_width=True):
            sample_docs = [
                "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and released in 1991.",
                "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
                "Python is widely used in data science, machine learning, web development, and automation.",
                "Python's package manager pip makes it easy to install and manage third-party packages."
            ]
            rag.add_documents(sample_docs)
            st.success("✅ Sample data loaded!")

with right_col:
    st.header("🔍 Ask Questions")
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know?"
    )
    n_results = st.slider("Number of results:", 1, 5, 3)
    
    if st.button("Search", use_container_width=True):
        if not query.strip():
            st.error("Please enter a question")
        else:
            with st.spinner("Searching..."):
                results = rag.query(query, n_results)
                
            if results:
                st.subheader("📊 Results:")
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"Result {i} (Relevance: {score:.2f})"):
                        st.markdown(f"```\n{doc}\n```")
            else:
                st.info("No results found. Try adding some documents first.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Add your documents:**
       - Paste text in the left panel and click "Process Documents", or
       - Click "Load Sample Data" to try with example content
    
    2. **Ask questions:**
       - Type your question in the right panel
       - Adjust the number of results you want to see
       - Click "Search"
    
    3. **View results:**
       - Results are shown with relevance scores
       - Click each result to expand and view the full text
    """)
