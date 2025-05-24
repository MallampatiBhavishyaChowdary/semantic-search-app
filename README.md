# 🔍 Semantic Search Using Qdrant & Sentence Transformers

This project demonstrates a real-time **semantic search engine** built using [Streamlit](https://streamlit.io/), [Qdrant](https://qdrant.tech/), and [Sentence Transformers](https://www.sbert.net/). Users can input custom documents, enter semantic search queries, and visualize the top matches with similarity scores.

---

## 🚀 Live Demo

> ⚠️ To deploy this project on [Render](https://render.com/) or [Streamlit Cloud](https://streamlit.io/cloud), see the [Deployment](#deployment) section.

---

## 🧩 Features

- ✅ Add custom documents to your knowledge base
- 🔎 Perform real-time semantic search
- 📊 Visualize similarity scores with an interactive chart
- 🧠 Query history tracking
- 🌐 Modern UI with background styling and sidebar
- 🗂 Powered by Qdrant vector search and Sentence Transformers

---

## 🛠️ Tech Stack

| Tool/Library                | Purpose                                      |
|----------------------------|----------------------------------------------|
| [Streamlit](https://streamlit.io/)         | Interactive web frontend                      |
| [Qdrant](https://qdrant.tech/)             | In-memory vector database                     |
| [Sentence Transformers](https://www.sbert.net/) | Embedding model (`mxbai-embed-large-v1`)      |
| [Matplotlib](https://matplotlib.org/)      | Visualization of similarity scores            |
| Python                       | Programming language                        |

---

## ⚙️ How It Works

1. **Document Upload**  
   Users add texts (documents) through the UI, which are converted into high-dimensional **embeddings** using the Sentence Transformer model.

2. **Storage in Qdrant**  
   The vector embeddings are stored in an **in-memory Qdrant collection** with cosine similarity as the distance metric.

3. **Semantic Querying**  
   A user inputs a search query, which is embedded and compared to existing document vectors in Qdrant.

4. **Result Display**  
   Top matching documents are shown, ranked by similarity score, along with a horizontal bar chart.

---

## 📸 UI Preview

| Home Interface | Similarity Chart |
|----------------|------------------|
| ![Home UI](https://designimages.appypie.com/allimages/appbackground60.webp) | ![Bar Chart](https://via.placeholder.com/500x300.png?text=Similarity+Score+Chart) |

---

## 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.commallampatiBhavishyaChowdary/semantic-search-app.git
cd semantic-search-app

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🧪 Sample Inputs
Example Documents:
"Who is German and likes bread?"

"Everyone in Germany."

"French people love baguettes."

"Italy is famous for pizza."

Sample Query:
"Who likes bread in Europe?"

Returns:

"Who is German and likes bread?"

"Everyone in Germany."

"French people love baguettes."

📦 Deployment
You can deploy this project on:

🌐 Streamlit Cloud

🚀 Render (add render.yaml for config)

Ensure you include:

requirements.txt

app.py

render.yaml (optional for Render)

📚 References
Qdrant Docs

Sentence Transformers

Mixedbread AI Embedding Model

Streamlit Documentation

👨‍💻 Author
Mallampati Bhavishya
