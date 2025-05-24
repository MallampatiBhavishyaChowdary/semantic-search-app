%pip install transformers sentence-transformers torch             
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer                                                                       client = QdrantClient(":memory:") 

collection_name = "text_search"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")


client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f"✅ Collection '{collection_name}' created successfully.")
                             model = SentenceTransformer('all-MiniLM-L6-v2')  

texts = [
    "Who is German and likes bread?",
    "Everyone in Germany.",
    "French people love baguettes.",
    "Italy is famous for pizza."
]


embeddings = model.encode(texts, convert_to_numpy=True)                                                                      
for i, (text, vector) in enumerate(zip(texts, embeddings)):
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=vector.tolist(),
                payload={"text": text}
            )
        ]
    )

print("✅ Embeddings inserted successfully!")                                                                                             
query = "Who likes bread in Europe?"


query_embedding = model.encode([query], convert_to_numpy=True)[0]


results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=3
)


print(f"\n🔍 Query: {query}")
for res in results:
    print(f"Text: {res.payload['text']} | Score: {res.score:.4f}")  this our project code and we developed an app as import streamlit as st
import uuid
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Page Config (Must be first) ---
st.set_page_config(
    page_title="Semantic Search Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {background-color: #f4f9fd;}
    .block-container {padding: 2rem 3rem;}
    h1, h2, h3 {color: #003262;}
    .stTextInput > div > div > input {
        border: 1px solid #003262;
        border-radius: 10px;
        padding: 8px;
    }
    .stApp {
        background-image: url('https://designimages.appypie.com/allimages/appbackground60.webp');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 3rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Sentence Transformer Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

model = load_model()

# --- Initialize Qdrant In-Memory DB ---
@st.cache_resource
def init_qdrant():
    client = QdrantClient(":memory:")
    collection_name = "text_search"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    return client, collection_name

client, collection_name = init_qdrant()

# --- Insert Default Texts ---
default_texts = [
    "Who is German and likes bread?",
    "Everyone in Germany.",
    "French people love baguettes.",
    "Italy is famous for pizza."
]

def insert_texts(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    for text, vector in zip(texts, embeddings):
        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=vector.tolist(),
                payload={"text": text}
            )]
        )

insert_texts(default_texts)

# --- Sidebar Info ---
with st.sidebar:
    st.title("⚙ Settings")
    st.markdown("This app uses [Qdrant](https://qdrant.tech) and [Sentence Transformers](https://www.sbert.net/) to perform semantic similarity search.")
    st.markdown("🔄 Add custom documents and search similar texts.")
    st.divider()
    st.info("Built by Bhavishya for SDP Viva")

# --- Main Interface ---
st.title("🔍 Semantic Search App")
st.subheader("🔎 Real-time search with semantic understanding")

# --- Session State for Query History ---
if 'queries' not in st.session_state:
    st.session_state.queries = []

# --- Custom Document Input ---
with st.expander("📄 Add Custom Document"):
    document = st.text_area("Enter your custom document here:")
    if st.button("➕ Add to Knowledge Base"):
        if document:
            doc_embedding = model.encode([document], convert_to_numpy=True)[0]
            client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=uuid.uuid4().int >> 64,
                    vector=doc_embedding.tolist(),
                    payload={"text": document}
                )]
            )
            st.success("✅ Document added successfully!")

# --- Query Search ---
query = st.text_input("Type your search query here 👇")

if query:
    st.session_state.queries.append(query)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=5
    )

    st.markdown("### 🔍 Top Matches:")
    similarity_scores = [res.score for res in results]
    documents = [res.payload["text"] for res in results]

    for res in results:
        st.markdown(f"• {res.payload['text']}")
        st.caption(f"Similarity: {res.score:.4f}")
        st.markdown("---")

    # --- Chart Visualization ---
    st.markdown("### 📊 Similarity Scores Visualization")
    fig, ax = plt.subplots()
    ax.barh(documents[::-1], similarity_scores[::-1], color="#5dade2")
    ax.set_xlabel("Similarity Score")
    ax.set_xlim(0, 1)
    ax.set_title("Top Matching Documents")
    st.pyplot(fig)

# --- Query Evolution ---
if st.session_state.queries:
    with st.expander("🧠 Query History"):
        for q in reversed(st.session_state.queries[-5:]):
            st.write(f"• {q}")
