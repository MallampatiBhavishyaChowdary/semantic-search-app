import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchParams
import numpy as np
import matplotlib.pyplot as plt
import gc

# ------------------------ CONFIG ------------------------ #
st.set_page_config(page_title="Semantic Search App", layout="wide")
st.title("🔍 Real-Time Semantic Search (Streamlit + Qdrant)")

# ------------------------ CACHED RESOURCES ------------------------ #
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(":memory:")  # In-memory DB, no external API
    client.recreate_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return client

model = load_model()
qdrant = get_qdrant_client()

# ------------------------ ADD DOCUMENTS ------------------------ #
st.subheader("📄 Add Your Documents")
docs = st.text_area("Enter documents (one per line):", height=150)

if st.button("Embed Documents"):
    if docs.strip():
        lines = [d.strip() for d in docs.strip().split("\n") if d.strip()]
        embeddings = model.encode(lines).tolist()
        points = [PointStruct(id=i, vector=vec, payload={"text": lines[i]}) for i, vec in enumerate(embeddings)]
        qdrant.upsert(collection_name="my_collection", points=points)
        st.success("✅ Documents embedded and stored in memory!")
        gc.collect()

# ------------------------ QUERY SEARCH ------------------------ #
st.subheader("🔎 Semantic Search")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query.strip():
        query_embedding = model.encode(query).tolist()
        hits = qdrant.search(
            collection_name="my_collection",
            query_vector=query_embedding,
            limit=5,
            search_params=SearchParams(hnsw_ef=64, exact=False)
        )

        if hits:
            st.markdown("### 🔍 Top Matches")
            for hit in hits:
                st.markdown(f"**Score:** {hit.score:.4f} — `{hit.payload['text']}`")

            # Optional: show similarity chart
            st.subheader("📊 Similarity Scores")
            scores = [hit.score for hit in hits]
            labels = [f"Doc {i+1}" for i in range(len(hits))]
            fig, ax = plt.subplots()
            ax.barh(labels, scores, color="skyblue")
            ax.invert_yaxis()
            ax.set_xlabel("Cosine Similarity")
            st.pyplot(fig)

        else:
            st.warning("❗ No matches found.")
        gc.collect()
