# 📄 DocChat — RAG-Based Document Chatbot

An AI-powered chatbot that lets you upload any PDF and have a conversation with it.
Built with LangChain, OpenAI, FAISS, and Streamlit.

---

## 🧠 How RAG Works

```
PDF → Extract Text → Split Chunks → Embed (OpenAI) → FAISS Vector Store
                                                              ↓
User Question → Embed Question → Find Similar Chunks → GPT → Answer
```

---

## 🚀 Setup (Step by Step)

### Step 1 — Clone / download project
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your OpenAI API key
```bash
cp .env.example .env
# Open .env and paste your key:
# OPENAI_API_KEY=sk-...
```
Get a free API key at: https://platform.openai.com/api-keys

### Step 5 — Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py            # Streamlit UI
├── rag_engine.py     # Core RAG pipeline
├── requirements.txt  # Dependencies
├── .env.example      # API key template
└── README.md
```

---

## 🌐 Deploy Free on HuggingFace Spaces

1. Create account at https://huggingface.co
2. New Space → Streamlit → upload all files
3. Add secret: `OPENAI_API_KEY` = your key
4. Share the live link on your resume!

---

## ✨ Features

- Upload any PDF (textbooks, papers, contracts, reports)
- Multi-turn conversation with memory
- Shows source chunks used for each answer
- Fast similarity search with FAISS
- Clean, minimal UI

---

## 🛠️ Built With

- [LangChain](https://langchain.com) — RAG pipeline
- [OpenAI](https://openai.com) — Embeddings + GPT-3.5
- [FAISS](https://faiss.ai) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web UI
- [PyPDF2](https://pypdf2.readthedocs.io) — PDF parsing
