import streamlit as st
import anthropic
from rank_bm25 import BM25Okapi
import pdfplumber
import re

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocRAG — Chat with your documents",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit default header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #0d1117; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* Pipeline steps */
    .pipeline-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #8b949e;
    }

    /* Source card */
    .source-card {
        background: #1c1600;
        border-left: 3px solid #9e6a03;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 12px;
        color: #e3b341;
    }
    .source-text {
        color: #8b949e;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 6px;
    }

    /* Chat input */
    .stChatInput { border-radius: 10px; }

    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = set([
    'the','and','for','are','but','not','you','all','can','had','her','was','one',
    'our','out','has','him','his','how','its','may','new','now','old','see','two',
    'way','who','did','get','let','put','say','she','too','use','that','this','with',
    'have','from','they','will','been','were','said','each','than','then','them',
    'more','also','into','over','such','when','where','which','while','their','there',
    'would','could','should','about','after','before','other','some','very','just',
    'like','what','know','take','only','even','back','much','come','here','most',
    'make','does','both','used','being','those','through','during','between','another','without'
])


# ─── Text Processing ──────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text.strip(), num_pages


def chunk_document(text, max_words=300, overlap=60):
    """
    Chunk text into overlapping passages.
    Strategy: paragraph-aware first, sliding window as fallback.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 40]
    chunks = []

    if len(paragraphs) > 2:
        current, word_count = [], 0
        for para in paragraphs:
            para_words = len(para.split())
            if word_count + para_words > max_words and current:
                chunks.append('\n\n'.join(current))
                last = current[-1]
                last_wc = len(last.split())
                current = [last] if last_wc <= overlap else []
                word_count = last_wc if current else 0
            current.append(para)
            word_count += para_words
        if current:
            chunks.append('\n\n'.join(current))

    # Fallback: sliding window on words
    if not chunks:
        words = text.split()
        step = max_words - overlap
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + max_words])
            if len(chunk) > 60:
                chunks.append(chunk)

    return chunks


def tokenize(text):
    """Tokenize text — lowercase, strip punctuation, remove stopwords."""
    tokens = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


# ─── BM25 Retrieval ───────────────────────────────────────────────────────────
def build_bm25_index(chunks):
    """Build a BM25Okapi index from a list of text chunks."""
    tokenized = [tokenize(c) for c in chunks]
    return BM25Okapi(tokenized)


def retrieve_chunks(bm25_index, chunks, query, top_k=5):
    """
    Score all chunks against the query using BM25.
    Returns list of (chunk_text, score) tuples.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return [(chunks[i], 0.0) for i in range(min(top_k, len(chunks)))]

    scores = bm25_index.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[idx], float(scores[idx])) for idx in ranked]


# ─── Claude Generation ────────────────────────────────────────────────────────
def generate_answer(api_key, doc_name, query, retrieved):
    """Call Claude Sonnet with retrieved chunks as grounded context."""
    context = '\n\n---\n\n'.join([
        f"[Source {i+1}]:\n{chunk}"
        for i, (chunk, _) in enumerate(retrieved)
    ])

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=f"""You are a precise document Q&A assistant for the file: "{doc_name}".

Strict rules:
- Answer ONLY from the provided source excerpts below
- If the answer is not in the excerpts, say: "This information is not found in the document"
- Reference [Source N] when citing specific content
- Be accurate and concise — no hallucination

Retrieved excerpts ({len(retrieved)} most relevant chunks):
{context}""",
        messages=[{"role": "user", "content": query}]
    )
    return response.content[0].text


# ─── Session State Init ───────────────────────────────────────────────────────
if "documents"   not in st.session_state: st.session_state.documents   = {}
if "messages"    not in st.session_state: st.session_state.messages    = []
if "active_doc"  not in st.session_state: st.session_state.active_doc  = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 DocRAG")
    st.caption("Chat with your documents — grounded answers, no hallucination")
    st.divider()

    # ── API Key ──────────────────────────────────────────────────────────────
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=default_key,
        placeholder="sk-ant-api03-…",
        help="Get your key at console.anthropic.com. On Streamlit Cloud, set ANTHROPIC_API_KEY in Secrets."
    )

    if api_key:
        st.success("API key configured ✓", icon="🔑")
    else:
        st.warning("Enter your API key to enable chat", icon="⚠️")

    st.divider()

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown("**Upload document**")
    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Text-based PDFs only. Scanned documents won't work."
    )

    if uploaded:
        if uploaded.name not in st.session_state.documents:
            with st.spinner(f"Indexing "{uploaded.name}"…"):
                try:
                    text, num_pages = extract_text_from_pdf(uploaded)
                    if len(text.strip()) < 50:
                        st.error("No readable text found. This may be a scanned PDF.")
                    else:
                        chunks     = chunk_document(text)
                        bm25_index = build_bm25_index(chunks)

                        st.session_state.documents[uploaded.name] = {
                            "name":     uploaded.name,
                            "pages":    num_pages,
                            "chunks":   chunks,
                            "bm25":     bm25_index,
                            "size_kb":  round(uploaded.size / 1024),
                        }
                        st.session_state.active_doc = uploaded.name

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f'✅ **"{uploaded.name}"** indexed!\n\n'
                                f'{num_pages} pages → **{len(chunks)} chunks** ready for BM25 retrieval\n\n'
                                f'Try asking:\n'
                                f'- "Summarize the key topics in this document"\n'
                                f'- "What does it say about [topic]?"\n'
                                f'- "List all the main sections"'
                            ),
                            "sources": []
                        })
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ── Document Selector ────────────────────────────────────────────────────
    if st.session_state.documents:
        st.divider()
        st.markdown("**Indexed documents**")
        for doc_name, doc in st.session_state.documents.items():
            is_active  = st.session_state.active_doc == doc_name
            short_name = doc_name[:26] + "…" if len(doc_name) > 26 else doc_name
            label      = f"{'● ' if is_active else '○ '}{short_name}"
            if st.button(label, key=f"select_{doc_name}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_doc = doc_name
                st.rerun()
            st.caption(f"{doc['pages']}p · {len(doc['chunks'])} chunks · {doc['size_kb']}KB")

    # ── Pipeline Info ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**RAG pipeline**")
    for step in [
        ("📤", "PDF extraction",   "pdfplumber"),
        ("✂️", "Text chunking",    "sliding window + overlap"),
        ("🔍", "BM25 retrieval",   "rank-bm25 library"),
        ("🤖", "Generation",       "Claude Sonnet 4.6"),
    ]:
        st.markdown(
            f'<div class="pipeline-box">{step[0]} <b>{step[1]}</b><br>'
            f'<span style="font-size:10px;color:#484f58">{step[2]}</span></div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 DocRAG")

if st.session_state.active_doc:
    doc = st.session_state.documents[st.session_state.active_doc]
    st.caption(
        f"Active: **{st.session_state.active_doc}** · "
        f"{len(doc['chunks'])} chunks · BM25 retrieval · Claude Sonnet 4.6"
    )
else:
    st.info("👈 Upload a PDF from the sidebar to get started.", icon="📄")

st.divider()

# ── Render conversation ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🔍" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # Source chunks expandable
        sources = msg.get("sources", [])
        if sources:
            max_score = max((s["score"] for s in sources), default=0.001) or 0.001
            with st.expander(f"📑 {len(sources)} chunks retrieved"):
                for j, src in enumerate(sources):
                    score = src["score"]
                    pct   = int((score / max_score) * 100) if max_score > 0 else 0
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Source {j+1}**")
                    with col2:
                        st.markdown(
                            f'<span style="font-family:monospace;font-size:11px;'
                            f'color:#e3b341">BM25: {score:.2f}</span>',
                            unsafe_allow_html=True
                        )
                    st.progress(pct / 100)
                    preview = src["text"][:280] + ("…" if len(src["text"]) > 280 else "")
                    st.markdown(
                        f'<div class="source-text">{preview}</div>',
                        unsafe_allow_html=True
                    )
                    if j < len(sources) - 1:
                        st.divider()

# ── Chat Input ────────────────────────────────────────────────────────────────
placeholder = (
    f'Ask about "{st.session_state.active_doc[:30]}…"…'
    if st.session_state.active_doc and len(st.session_state.active_doc) > 30
    else f'Ask about "{st.session_state.active_doc}"…'
    if st.session_state.active_doc
    else "Upload a PDF first…"
)

if user_query := st.chat_input(placeholder, disabled=not st.session_state.active_doc):

    # Validate
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar first.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # Run RAG pipeline
    with st.chat_message("assistant", avatar="🔍"):
        active_doc = st.session_state.documents[st.session_state.active_doc]

        with st.status("Running RAG pipeline…", expanded=True) as status:
            # Step 1: Retrieve
            st.write("🔍 BM25 retrieval…")
            results  = retrieve_chunks(active_doc["bm25"], active_doc["chunks"], user_query, top_k=5)
            relevant = [(c, s) for c, s in results if s > 0] or results[:3]
            st.write(f"✅ {len(relevant)} relevant chunks found")

            # Step 2: Generate
            st.write("🤖 Generating answer with Claude…")
            try:
                answer = generate_answer(
                    api_key,
                    st.session_state.active_doc,
                    user_query,
                    relevant
                )
                status.update(label="Done!", state="complete", expanded=False)
            except anthropic.AuthenticationError:
                answer = "❌ Invalid API key. Please check your Anthropic API key in the sidebar."
                status.update(label="Authentication error", state="error")
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                status.update(label="Error", state="error")

        # Display answer
        st.markdown(answer)

        # Display sources
        sources_data = [{"text": c, "score": s} for c, s in relevant]
        max_score    = max((s["score"] for s in sources_data), default=0.001) or 0.001

        with st.expander(f"📑 {len(sources_data)} chunks retrieved"):
            for j, src in enumerate(sources_data):
                score = src["score"]
                pct   = int((score / max_score) * 100) if max_score > 0 else 0
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Source {j+1}**")
                with col2:
                    st.markdown(
                        f'<span style="font-family:monospace;font-size:11px;'
                        f'color:#e3b341">BM25: {score:.2f}</span>',
                        unsafe_allow_html=True
                    )
                st.progress(pct / 100)
                preview = src["text"][:280] + ("…" if len(src["text"]) > 280 else "")
                st.markdown(f'<div class="source-text">{preview}</div>', unsafe_allow_html=True)
                if j < len(sources_data) - 1:
                    st.divider()

        # Save to session state
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources_data
        })
