import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance
import numpy as np

# Load model with caching
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional

model = load_model()

# Initialize in-memory Qdrant DB with 384-dim vectors
@st.cache_resource
def init_qdrant():
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return client

qdrant = init_qdrant()

# UI
st.title("📚 Lightweight Semantic Search")
st.markdown("Add your own documents and run semantic queries in real-time.")

# Document input
docs = st.text_area("Enter documents (one per line):")
if st.button("Embed & Store"):
    if docs.strip():
        lines = docs.strip().split("\n")
        embeddings = model.encode(lines).tolist()
        points = [models.PointStruct(id=i, vector=vec, payload={"text": line}) for i, (line, vec) in enumerate(zip(lines, embeddings))]
        qdrant.upsert(collection_name="docs", points=points)
        st.success(f"Added {len(lines)} documents to vector DB.")
    else:
        st.warning("No documents entered.")

# Search
query = st.text_input("🔍 Enter search query:")
if query:
    qvec = model.encode(query).tolist()
    results = qdrant.search(collection_name="docs", query_vector=qvec, limit=5)
    st.subheader("Top Matches")
    for res in results:
        st.write(f"- **{res.payload['text']}** (score: {res.score:.2f})")

