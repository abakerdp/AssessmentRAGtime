import streamlit as st
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import textwrap

class SimpleRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system with a specified embedding model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def load_documents(self, documents: List[str]) -> None:
        """Load and chunk documents."""
        self.chunks = []
        for doc in documents:
            # Simple paragraph-based chunking
            paragraphs = doc.split('\n\n')
            for para in paragraphs:
                # Clean and wrap text
                cleaned = ' '.join(para.split())
                if cleaned:
                    # Split into smaller chunks if paragraph is too long
                    if len(cleaned) > 500:
                        wrapped = textwrap.wrap(cleaned, 500)
                        self.chunks.extend(wrapped)
                    else:
                        self.chunks.append(cleaned)
    
    def build_index(self) -> None:
        """Create FAISS index from document chunks."""
        if not self.chunks:
            return
            
        # Generate embeddings
        embeddings = self.model.encode(self.chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(np.array(embeddings).astype('float32'))
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """Retrieve relevant chunks for a query."""
        if not self.index:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(dist),
                'rank': i + 1
            })
            
        return results

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    return SimpleRAG()

# Streamlit interface
st.title("📚 Simple RAG System")

# Sidebar for document input
st.sidebar.header("Add Documents")
doc_input = st.sidebar.text_area(
    "Enter your documents here (one or more paragraphs):",
    height=300,
    help="Paste your text documents here. Separate different documents with blank lines."
)

# Initialize or get RAG system
rag = get_rag_system()

# Process documents when submitted
if st.sidebar.button("Process Documents"):
    if doc_input.strip():
        with st.spinner("Processing documents..."):
            rag.load_documents([doc_input])
            rag.build_index()
        st.sidebar.success("Documents processed successfully!")
    else:
        st.sidebar.error("Please enter some documents first.")

# Main area for queries
st.header("Ask Questions")
query = st.text_input("Enter your question:", key="query")
k = st.slider("Number of results to return:", min_value=1, max_value=5, value=3)

# Process query
if st.button("Search"):
    if not rag.index:
        st.error("Please add and process some documents first!")
    elif not query.strip():
        st.error("Please enter a question!")
    else:
        with st.spinner("Searching..."):
            results = rag.retrieve(query, k=k)
            
        if results:
            st.subheader("Results:")
            for result in results:
                with st.expander(f"Result {result['rank']} (Score: {result['score']:.4f})"):
                    st.write(result['chunk'])
        else:
            st.warning("No results found. Try a different query or add more documents.")

# Add sample data button
if st.sidebar.button("Load Sample Data"):
    sample_data = """ChatGPT is an artificial intelligence chatbot developed by OpenAI. 
    It was launched in November 2022 and has gained significant popularity.
    The chatbot uses natural language processing to generate human-like responses.
    
    ChatGPT is built on top of OpenAI's GPT family of large language models.
    It can engage in conversations, answer questions, and assist with various tasks.
    
    OpenAI was founded in 2015 with the goal of ensuring artificial 
    intelligence benefits humanity as a whole. The company has developed
    several influential AI models and technologies.
    
    In 2019, OpenAI transitioned from a non-profit to a "capped-profit"
    model to attract more funding while maintaining its mission."""
    
    with st.spinner("Loading sample data..."):
        rag.load_documents([sample_data])
        rag.build_index()
    st.sidebar.success("Sample data loaded!")

# Add instructions
with st.sidebar.expander("How to use"):
    st.markdown("""
    1. **Add documents**: Paste your text in the sidebar text area and click "Process Documents"
    2. **Or use sample data**: Click "Load Sample Data" to try with example content
    3. **Ask questions**: Type your question in the main area and click "Search"
    4. **Adjust results**: Use the slider to control how many results you see
    """)
