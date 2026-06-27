# DocRAG — Streamlit Version

A Retrieval-Augmented Generation (RAG) assistant built with **Python + Streamlit**. Upload any PDF and chat with it — answers are grounded entirely in the document with zero hallucination.

## Live Demo

> Deploy your own on Streamlit Cloud in 5 minutes — see below ↓

---

## RAG Pipeline

```
PDF Upload  ──▶  Text Extraction (pdfplumber)
                       │
                       ▼
              Text Chunking (sliding window + overlap)
                       │
                       ▼
              BM25 Indexing (rank-bm25)
                       │
            User Question
                       │
                       ▼
              BM25 Retrieval (top-5 relevant chunks)
                       │
                       ▼
              Claude Sonnet 4.6 (grounded generation)
                       │
                       ▼
                  Answer + Source viewer
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/docrag-streamlit.git
cd docrag-streamlit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create the secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open `.streamlit/secrets.toml` and add your key:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-key"
```

> Get your key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deploy on Streamlit Cloud (Free)

1. **Push this repo to GitHub** (public or private)

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. **Click "New app"** → select your repo → set main file as `app.py`

4. **Add your API key as a Secret:**
   - In the deployment settings, go to **"Advanced settings" → "Secrets"**
   - Add this:
     ```toml
     ANTHROPIC_API_KEY = "sk-ant-api03-your-key"
     ```

5. **Click Deploy** — your app will be live in ~2 minutes at:
   `https://yourusername-docrag-streamlit-app-xxxx.streamlit.app`

---

## Project Structure

```
docrag-streamlit/
├── app.py                          ← Main Streamlit application
├── requirements.txt                ← Python dependencies
├── .gitignore                      ← Excludes secrets from git
├── .streamlit/
│   ├── config.toml                 ← Dark theme config
│   └── secrets.toml.example        ← API key template (copy to secrets.toml)
└── README.md
```

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| UI | `streamlit` | Web app framework |
| PDF Parsing | `pdfplumber` | Extract text from PDFs |
| Retrieval | `rank-bm25` | BM25Okapi keyword ranking |
| Generation | `anthropic` | Claude Sonnet 4.6 |

---

## Features

- Upload multiple PDFs and switch between them
- Paragraph-aware chunking with word overlap
- BM25Okapi retrieval (better than TF-IDF)
- Source chunk viewer with relevance score bars
- Dark theme auto-configured
- API key via Streamlit Secrets (for deployment) or sidebar input (local)

---

## Limitations

- **Scanned PDFs** won't work — use OCR first (e.g. Adobe Acrobat, pytesseract)
- **BM25 is keyword-based** — won't catch semantic similarity. Upgrade to vector embeddings (ChromaDB + OpenAI embeddings) for production
- **No memory across sessions** — documents reset on page refresh

---

## License

MIT
