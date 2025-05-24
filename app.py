import streamlit as st
import numpy as np

# --- MEMORY-EFFICIENT MODEL LOAD ---
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")  # lightweight model

model = load_model()

# --- SIMULATED DOCUMENT LOADING (Replace with real logic) ---
@st.cache_data
def load_my_documents():
    # Simulate a few short sample docs (TEMP FIX: Limit to 5)
    return [
        "Streamlit is an open-source Python app framework for ML and data science.",
        "Qdrant is a vector similarity search engine written in Rust.",
        "SentenceTransformers make it easy to compute embeddings.",
        "Render is a cloud provider for hosting full stack apps.",
        "Vector search helps you find similar documents semantically.",
        "This document won't be loaded due to limit."  # gets excluded
    ][:5]  # Limit number of documents to stay under memory

# --- EMBEDDING FUNCTION ---
@st.cache_data
def embed_documents(docs):
    return model.encode(docs, show_progress_bar=False)

# --- APP UI ---
st.set_page_config(page_title="Semantic Search", layout="centered")
st.title("🔍 Lightweight Semantic Search App (Render Optimized)")

docs = load_my_documents()
doc_embeddings = embed_documents(docs)

query = st.text_input("Enter your search query:")
if query:
    query_embedding = model.encode([query])[0]

    # Compute cosine similarity
    scores = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-9
    )

    # Show top 3 results
    top_k = 3
    top_indices = np.argsort(scores)[::-1][:top_k]
    st.subheader("Top Matches:")
    for i in top_indices:
        st.markdown(f"**Score:** {scores[i]:.2f}")
        st.write(docs[i])
