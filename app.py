"""
app.py — Streamlit UI for the Self-Correcting RAG Pipeline

Features:
  - Chat interface with per-answer confidence badges
  - Sidebar file uploader (judges can drop in their own docs)
  - Retrieval trace expander per answer showing chunks + scores + judge reasoning
  - Visual badges: ✅ Grounded / ⚠️ Low Confidence / 🔀 Contradiction
"""

import os
import tempfile
import shutil
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from Streamlit secrets (Streamlit Cloud) or environment variable
try:
    key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
except Exception:
    pass  # No secrets file or empty — rely on environment variable

from ingest import (
    run_ingestion,
    get_chroma_collection,
    load_documents_from_folder,
    chunk_documents,
    get_embedding_function,
    COLLECTION_NAME,
)
from rag_pipeline import run_pipeline, DecisionType, PipelineResult

DATA_FOLDER = "./data"
CHROMA_DIR = "./chroma_db"


# ── Chroma collection loader (cached across reruns) ───────────────────────────
@st.cache_resource(show_spinner="Building knowledge base index…")
def load_collection():
    """Load existing Chroma collection, or build it from data/ on first run."""
    try:
        col = get_chroma_collection(CHROMA_DIR)
        if col.count() == 0:
            raise ValueError("Empty collection — needs rebuild")
        return col
    except Exception:
        run_ingestion(DATA_FOLDER, CHROMA_DIR)
        return get_chroma_collection(CHROMA_DIR)


# ── Rendering helper — must be defined before it is called ───────────────────
def render_assistant_message(result: PipelineResult) -> None:
    """Render confidence badge, answer text, and collapsible retrieval trace."""

    # Badge
    if result.decision == DecisionType.GROUNDED_ANSWER:
        st.success("✅ **Grounded** — answer fully supported by retrieved sources")
    elif result.decision == DecisionType.CONTRADICTORY:
        st.warning("🔀 **Contradiction Detected** — sources disagree on this topic")
    else:
        st.info("⚠️ **Low Confidence** — clarification requested")

    # Answer
    st.markdown(result.answer)

    # Retrieval trace (collapsed by default)
    with st.expander("🔍 Show Retrieval Trace", expanded=False):
        left, right = st.columns([3, 2])

        with left:
            st.markdown("**Retrieved Chunks:**")
            if result.chunks_used:
                for i, chunk in enumerate(result.chunks_used):
                    filled = int(chunk.score * 20)
                    bar = "█" * filled + "░" * (20 - filled)
                    st.markdown(
                        f"**Chunk {i + 1}** | `{chunk.source}` p.{chunk.page} | "
                        f"Score: `{chunk.score:.3f}` `{bar}`"
                    )
                    preview = chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
                    st.text(preview)
                    st.divider()
            else:
                st.markdown("_No chunks retrieved._")

        with right:
            st.markdown("**LLM Judge Result:**")
            if result.judge_result:
                j = result.judge_result
                st.markdown(f"- Sufficient: `{j.sufficient}`")
                st.markdown(f"- Contradictory: `{j.contradictory}`")
                st.markdown(f"- Reasoning: _{j.reasoning}_")
                if j.conflicting_claims:
                    st.markdown("**Conflicting claims:**")
                    for claim in j.conflicting_claims:
                        st.markdown(f"> {claim}")
            else:
                st.markdown("_Judge not called (score below threshold — re-query path)_")

            st.markdown("**Pipeline Decision Log:**")
            for entry in result.log_entries:
                st.code(entry, language=None)

            if result.requery_attempted and result.reformulated_query:
                st.markdown(f"**Reformulated query:** _{result.reformulated_query}_")

            st.markdown(
                f"**Top similarity score:** `{result.confidence_score:.4f}` | "
                f"**Decision:** `{result.decision.value}`"
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Self-Correcting RAG")
    st.markdown("---")

    st.subheader("📁 Knowledge Base")
    st.markdown(
        "Pre-loaded sample documents:\n\n"
        "- **Company Policy Handbook 2024**\n"
        "- **HR Policy Update Memo** *(March 2024)*\n"
        "- **Project Phoenix Technical Spec**\n\n"
        "_The memo deliberately contradicts the handbook on several policies — "
        "use the contradiction demo buttons to see this caught live._"
    )

    st.markdown("---")
    st.subheader("📤 Upload Your Own Documents")
    uploaded_files = st.file_uploader(
        "Add PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Files are ingested into the same ChromaDB collection.",
    )

    if uploaded_files and st.button("📥 Ingest Uploaded Files", type="primary"):
        with st.spinner("Ingesting…"):
            tmp_dir = tempfile.mkdtemp()
            try:
                for f in uploaded_files:
                    (Path(tmp_dir) / f.name).write_bytes(f.read())

                docs = load_documents_from_folder(tmp_dir)
                if docs:
                    chunks = chunk_documents(docs)
                    import chromadb as _chromadb
                    _ef = get_embedding_function()
                    _client = _chromadb.PersistentClient(path=CHROMA_DIR)
                    _col = _client.get_collection(name=COLLECTION_NAME, embedding_function=_ef)
                    _col.upsert(
                        ids=[c.metadata["chunk_id"] + "_upload" for c in chunks],
                        documents=[c.page_content for c in chunks],
                        metadatas=[c.metadata for c in chunks],
                    )
                    st.success(f"✅ Ingested {len(chunks)} chunks from {len(uploaded_files)} file(s)")
                    st.cache_resource.clear()
                else:
                    st.warning("No text could be extracted from the uploaded files.")
            except Exception as exc:
                st.error(f"Ingestion error: {exc}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    st.markdown("---")
    st.subheader("⚙️ About")
    st.markdown(
        "**Self-Correcting RAG Pipeline**\n\n"
        "Detects insufficient or contradictory context *before* generating an answer.\n\n"
        "**Confidence badges:**\n"
        "- ✅ **Grounded** — fully supported by sources\n"
        "- ⚠️ **Low Confidence** — clarification requested\n"
        "- 🔀 **Contradiction** — sources disagree, conflict surfaced\n\n"
        "Stack: ChromaDB · sentence-transformers · Claude 3.5 Haiku · Streamlit"
    )


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🧠 Self-Correcting RAG Pipeline")
st.caption(
    "Ask questions about the loaded documents. "
    "The system checks retrieval confidence and detects source contradictions "
    "before generating any answer."
)

# Load (or build) the vector index
collection = load_collection()
st.info(f"📚 Knowledge base ready — **{collection.count()} chunks** indexed across 3 documents.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "result" in msg:
            render_assistant_message(msg["result"])
        else:
            st.markdown(msg["content"])

# Live chat input
if prompt := st.chat_input("Ask a question about the documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                result = run_pipeline(prompt, collection)
                render_assistant_message(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer, "result": result}
                )
            except Exception as exc:
                err_msg = (
                    f"❌ **Pipeline error:** `{exc}`\n\n"
                    "Make sure `ANTHROPIC_API_KEY` is set in your environment or Streamlit secrets."
                )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})


# ── One-click demo buttons ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**💡 Click a demo question to try it instantly:**")

demo_questions = [
    ("✅ Answerable",    "What is the gym membership subsidy amount?"),
    ("🔀 Contradiction", "How many days of annual leave do full-time employees get?"),
    ("⚠️ No Answer",     "What is the company's policy on cryptocurrency investments?"),
    ("✅ Answerable",    "What database does Project Phoenix use?"),
    ("🔀 Contradiction", "Does the company reimburse home internet costs?"),
    ("⚠️ No Answer",     "What is the pet insurance allowance for employees?"),
    ("✅ Answerable",    "What is the Recovery Time Objective for Project Phoenix?"),
    ("🔀 Contradiction", "What is the meal allowance for international travel?"),
    ("⚠️ No Answer",     "What is the company's stance on social media during work hours?"),
]

cols = st.columns(3)
for i, (label, question) in enumerate(demo_questions):
    with cols[i % 3]:
        short = question if len(question) <= 58 else question[:55] + "…"
        if st.button(f"{label}  \n_{short}_", key=f"demo_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
