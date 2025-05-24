# 🔍✨ Semantic Search Web App — Powered by AI & Vectors!

Welcome to the **smartest way to search** through text using meaning — not just keywords! This app uses **Sentence Transformers**, **Qdrant Vector DB**, and a beautiful **Streamlit UI** to help you find semantically similar documents 🔥

---

## 🧠 What Can This App Do?

🎯 **Understand Your Queries**  
Say goodbye to boring keyword matches! We use **semantic embeddings** to understand the *real meaning* of your search.

📄 **Add Your Own Docs**  
Have your own knowledge base? Paste it in and search through it — instantly!

📊 **Get Visual**  
Enjoy a cute bar chart showing how close your search was to the top results 💙

🕵️ **Track Your Curiosity**  
Every query you try is stored so you can peek back at your search history like a detective 🕵️‍♂️

---

## 🚀 Tech Behind the Magic

- 🧠 `mixedbread-ai/mxbai-embed-large-v1` – our brain for understanding text  
- 🧰 `Qdrant` – the memory palace that stores all document embeddings  
- 🎨 `Streamlit` – the sleek, modern interface  
- 📊 `Matplotlib` – our go-to for visualizing similarity scores  
- 💬 `Python` – the one ring to glue them all together!

---

## 🕹️ How to Use It (Locally)

```bash
# Clone this magical repo
git clone https://github.com/your-username/semantic-search-app.git
cd semantic-search-app

# Start a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # use venv\Scripts\activate on Windows

# Install the spellbook 🧪
pip install -r requirements.txt

# Run the magic 🧙‍♂️
streamlit run app.py

## 🧪 Test It Out!

Try adding documents like:

- "Who is German and likes bread?"
- "Everyone in Germany."
- "French people love baguettes."
- "Italy is famous for pizza."

Then search:

> ✨ **"Who likes bread in Europe?"**

And watch the magic unfold 💫

---

## 🌍 Deploy It Like a Pro

This app works beautifully on:

- ☁️ **Streamlit Cloud** – super fast & free hosting!
- 🚀 **Render** – with a `render.yaml` for config

Just make sure to include:

- `app.py`
- `requirements.txt`
- *(Optional)* `render.yaml`

---

## 🧾 Requirements

Create a file named `requirements.txt` and add the following:

streamlit
sentence-transformers
qdrant-client
matplotlib
torch


Paste that in `requirements.txt`✨

---


Built with love by **Mallampati Bhavishya** 💙  
📍 Final Year CSE Student @ VIT-AP  


