"""
rag_engine.py
-------------
Core RAG pipeline:
  1. Load & parse PDF
  2. Split into chunks
  3. Embed chunks → FAISS vector store
  4. Retrieve relevant chunks for a query
  5. Generate answer with GPT
"""

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()


# ── Step 1: Extract text from PDF ────────────────────────────────────────────

def load_pdf(uploaded_file) -> str:
    """Read all pages of a PDF and return raw text."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ── Step 2: Split text into overlapping chunks ────────────────────────────────

def split_text(raw_text: str) -> list[str]:
    """
    Split long text into smaller overlapping chunks.
    - chunk_size=1000   → ~1000 characters per chunk
    - chunk_overlap=200 → 200 chars shared between adjacent chunks
      (prevents losing context at chunk boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    return chunks


# ── Step 3: Embed chunks → FAISS vector store ─────────────────────────────────

def build_vector_store(chunks: list[str]) -> FAISS:
    """
    Convert each text chunk into an embedding vector and store in FAISS.
    FAISS enables fast similarity search (nearest-neighbour lookup).
    """
    embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# ── Step 4 + 5: Build conversational QA chain ────────────────────────────────

def build_qa_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Combine:
      - Retriever  → finds the top-k most relevant chunks
      - LLM        → generates an answer using those chunks as context
      - Memory     → keeps track of the conversation history
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
)


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},     # retrieve top 4 chunks
        ),
        memory=memory,
        return_source_documents=True,   # show which chunks were used
        verbose=False,
    )
    return qa_chain


# ── Public helper ─────────────────────────────────────────────────────────────

def ask(qa_chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question and return:
      - answer      : the AI-generated response
      - sources     : list of text snippets used as context
    """
    result = qa_chain.invoke({"question": question})
    sources = [doc.page_content for doc in result.get("source_documents", [])]
    return {
        "answer": result["answer"],
        "sources": sources,
    }
