"""
app.py
------
Streamlit UI for the RAG Document Chatbot.
Run with:  streamlit run app.py
"""

import streamlit as st
from rag_engine import load_pdf, split_text, build_vector_store, build_qa_chain, ask

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocChat · RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f0f9f4;
        border-left: 3px solid #1D9E75;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        font-size: 12px;
        color: #444;
        margin-top: 6px;
    }
    .step-badge {
        display: inline-block;
        background: #1D9E75;
        color: white;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        text-align: center;
        line-height: 22px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ── Sidebar: upload + process ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 DocChat")
    st.markdown("Upload any PDF and chat with it using AI.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Max ~50 pages works best"
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Reading PDF..."):
                raw_text = load_pdf(uploaded_file)

            with st.spinner("Splitting into chunks..."):
                chunks = split_text(raw_text)

            with st.spinner(f"Embedding {len(chunks)} chunks into FAISS..."):
                vector_store = build_vector_store(chunks)

            with st.spinner("Building QA chain..."):
                st.session_state.qa_chain = build_qa_chain(vector_store)
                st.session_state.chat_history = []
                st.session_state.doc_name = uploaded_file.name

            st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

    st.divider()

    if st.session_state.doc_name:
        st.markdown(f"**Active doc:** `{st.session_state.doc_name}`")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("""
**How RAG works:**

<span class="step-badge">1</span> PDF → raw text  
<span class="step-badge">2</span> Text → chunks  
<span class="step-badge">3</span> Chunks → embeddings  
<span class="step-badge">4</span> Query → find similar chunks  
<span class="step-badge">5</span> Chunks + query → GPT answer  
""", unsafe_allow_html=True)


# ── Main chat area ────────────────────────────────────────────────────────────

st.markdown("## 💬 Chat with your document")

if not st.session_state.qa_chain:
    st.info("👈 Upload a PDF in the sidebar and click **Process Document** to get started.")

    # Show example use cases
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📚 Study notes**  
        Upload lecture PDFs and ask questions to prepare for exams.
        """)
    with col2:
        st.markdown("""
        **📋 Legal / Policy docs**  
        Paste a contract or HR policy and ask what it says about specific clauses.
        """)
    with col3:
        st.markdown("""
        **🔬 Research papers**  
        Upload a paper and ask for methodology, findings, or limitations.
        """)

else:
    # Render full chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Source chunks used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-box"><b>Chunk {i}:</b> {src[:300]}...</div>',
                            unsafe_allow_html=True
                        )

    # Chat input
    if user_question := st.chat_input("Ask anything about your document..."):
        # Show user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
        })
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get AI answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                result = ask(st.session_state.qa_chain, user_question)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📎 Source chunks used", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f'<div class="source-box"><b>Chunk {i}:</b> {src[:300]}...</div>',
                            unsafe_allow_html=True
                        )

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
