import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# ---------- MODEL LOADING ---------- #
@st.cache_resource(show_spinner="🔄 Loading the model. Please wait...")
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')



# ---------- EMBEDDING FUNCTION ---------- #
def embed_documents(model, documents):
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings


# ---------- MAIN APP ---------- #
def main():
    st.set_page_config(page_title="Semantic Search App", layout="centered")
    st.title("📚 Semantic Text Search")

    model = load_model()
    st.success("✅ Model loaded successfully!")

    # Text input for documents
    st.subheader("Add Documents:")
    document_input = st.text_area("Enter your documents (one per line):", height=200)
    documents = [doc.strip() for doc in document_input.split('\n') if doc.strip()]

    # Query input
    st.subheader("Search Query:")
    query = st.text_input("Enter your query:")

    if st.button("Search") and query and documents:
        with st.spinner("🔍 Searching..."):
            doc_embeddings = embed_documents(model, documents)
            query_embedding = model.encode(query, convert_to_tensor=True)

            scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            top_k = min(5, len(documents))
            top_results = torch.topk(scores, k=top_k)

            st.subheader("🔎 Top Matches:")
            for score, idx in zip(top_results[0], top_results[1]):
                st.markdown(f"**{documents[idx]}** — Similarity: `{score.item():.4f}`")


# ---------- MAIN CALL ---------- #
if __name__ == '__main__':
    main()
