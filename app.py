import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage, SystemMessage


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, local, no API key needed

MODEL_OPTIONS = {
    "openai/gpt-oss-120b": "Best quality (recommended)",
    "openai/gpt-oss-20b": "Fastest",
    "qwen/qwen3.6-27b": "Balanced",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
HISTORY_WINDOW = 6  

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant that answers questions about the "
    "user's uploaded PDF documents.\n"
    "Rules:\n"
    "- Answer ONLY using the document excerpts given to you below the question.\n"
    "- If the excerpts don't contain the answer, say so clearly instead of guessing.\n"
    "- When useful, mention which document/page the information came from.\n"
    "- Be clear and concise. Use short paragraphs or bullet points when helpful."
)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Loaded once per server process and reused across reruns/users."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_documents(pdf_files):
    """Extract text per page as Document objects with source/page metadata."""
    documents = []
    unreadable_files = []

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
        except Exception:
            unreadable_files.append(pdf_file.name)
            continue

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                unreadable_files.append(f"{pdf_file.name} (password protected)")
                continue

        found_text = False
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                found_text = True
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_file.name, "page": page_number},
                    )
                )

        if not found_text:
            unreadable_files.append(f"{pdf_file.name} (no extractable text — likely scanned)")

    return documents, unreadable_files


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def format_context(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_llm(model_name, api_key):
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_tokens=1024,
        streaming=True,
    )


def retrieve_context(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=TOP_K)
    context = format_context(docs)
    return docs, context


def build_messages(question, context, chat_history):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history[-HISTORY_WINDOW:])
    messages.append(
        HumanMessage(content=f"Document excerpts:\n\n{context}\n\nQuestion: {question}")
    )
    return messages


def stream_llm_tokens(llm, messages):
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def init_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],       
        "display_messages": [],   
        "processed_files": [],
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "model_name": DEFAULT_MODEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



CUSTOM_CSS = """
<style>
.main .block-container { padding-top: 2rem; max-width: 880px; }

[data-testid="stChatMessage"] { padding: 0.25rem 0; }

.app-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.app-header h1 {
    font-size: 1.7rem;
    margin: 0;
    background: linear-gradient(90deg, #F26B3A, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle { color: #8a8f98; margin-bottom: 1.5rem; font-size: 0.95rem; }

.status-pill {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.status-ok { background: #e3f6e9; color: #1c8a4b; }
.status-warn { background: #fdecea; color: #c0392b; }

section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 8px;
}
</style>
"""


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Setup")

        # API key
        if not st.session_state.groq_api_key:
            st.markdown('<span class="status-pill status-warn">No Groq API key</span>', unsafe_allow_html=True)
            key_input = st.text_input(
                "Groq API key",
                type="password",
                placeholder="gsk_...",
                help="Get a free key at console.groq.com. It's kept only in this browser session, never written to disk.",
            )
            if key_input:
                st.session_state.groq_api_key = key_input
                st.rerun()
        else:
            st.markdown('<span class="status-pill status-ok">Groq connected</span>', unsafe_allow_html=True)
            if st.button("Change API key"):
                st.session_state.groq_api_key = ""
                st.rerun()

        st.session_state.model_name = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_name),
            format_func=lambda m: f"{m}  ·  {MODEL_OPTIONS[m]}",
        )

        st.divider()
        st.markdown("### 📄 Documents")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Process documents", disabled=not pdf_docs, type="primary"):
            process_documents(pdf_docs)

        if st.session_state.processed_files:
            st.caption("Indexed:")
            for name in st.session_state.processed_files:
                st.caption(f"✅ {name}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.session_state.display_messages = []
                st.rerun()
        with col2:
            if st.button("♻️ Reset all"):
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.session_state.display_messages = []
                st.rerun()


def process_documents(pdf_docs):
    with st.spinner("Reading PDFs..."):
        documents, unreadable = extract_documents(pdf_docs)

    if unreadable:
        st.warning("Couldn't read: " + ", ".join(unreadable))

    if not documents:
        st.error("No readable text found in the uploaded file(s).")
        return

    with st.spinner(f"Splitting {len(documents)} page(s) into chunks..."):
        chunks = chunk_documents(documents)

    with st.spinner(f"Embedding {len(chunks)} chunk(s) (first run downloads a small local model)..."):
        try:
            st.session_state.vectorstore = build_vectorstore(chunks)
        except Exception as e:
            st.error(f"Failed to build the document index: {e}")
            return

    st.session_state.processed_files = [f.name for f in pdf_docs]
    st.session_state.chat_history = []
    st.session_state.display_messages = []
    st.success(f"Indexed {len(chunks)} chunks from {len(pdf_docs)} file(s). Ask away!")


def render_chat():
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources used"):
                    for doc in msg["sources"]:
                        source = doc.metadata.get("source", "document")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:220].strip()
                        st.markdown(f"**{source} — page {page}**")
                        st.caption(snippet + ("…" if len(doc.page_content) > 220 else ""))

    question = st.chat_input("Ask a question about your documents...")
    if question:
        handle_question(question)


def handle_question(question):
    if st.session_state.vectorstore is None:
        st.session_state.display_messages.append(
            {"role": "assistant", "content": "Please upload and process a PDF first (sidebar)."}
        )
        st.rerun()
        return

    if not st.session_state.groq_api_key:
        st.session_state.display_messages.append(
            {"role": "assistant", "content": "Please add your Groq API key in the sidebar first."}
        )
        st.rerun()
        return

    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            docs, context = retrieve_context(question, st.session_state.vectorstore)
            messages = build_messages(question, context, st.session_state.chat_history)
            llm = get_llm(st.session_state.model_name, st.session_state.groq_api_key)
            full_response = st.write_stream(stream_llm_tokens(llm, messages))
            sources = docs
        except Exception as e:
            message = str(e)
            if "401" in message or "invalid_api_key" in message.lower() or "unauthorized" in message.lower():
                full_response = "Your Groq API key looks invalid. Please double-check it in the sidebar."
            elif "rate" in message.lower() and "limit" in message.lower():
                full_response = "Groq's rate limit was hit. Please wait a moment and try again."
            else:
                full_response = f"Something went wrong while generating the answer: {message}"
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
            st.markdown(full_response)
            sources = []

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=full_response))
    st.session_state.display_messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_state()
    render_sidebar()

    st.markdown('<div class="app-header"><h1>Chat with your PDFs</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload documents, then ask questions — answered instantly by Groq.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.vectorstore is None:
        st.info("👈 Upload one or more PDFs in the sidebar and click **Process documents** to get started.")

    render_chat()


if __name__ == "__main__":
    main()
