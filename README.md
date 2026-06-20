# Chat with PDF

Upload PDFs, ask questions about them, get answers streamed back instantly by Groq.

## Setup

```bash
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key to `.env` (already done if you kept the file from before):

```
GROQ_API_KEY=your_key_here
```

Don't have a key? Get one free at [console.groq.com](https://console.groq.com/keys).
You can also paste it directly into the app sidebar instead — it's never written to disk that way.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), upload a PDF, click **Process documents**, and start chatting.

## What changed from the original

The original `app.py` had several bugs that prevented it from running at all, plus a very slow embedding model. Everything below was fixed:

| Issue | Fix |
|---|---|
| `from langchain_memory import ...` — package doesn't exist | Removed; chat history is now plain Python state, no deprecated memory class needed |
| `from langchain_llms import Groq` — package doesn't exist | `from langchain_groq import ChatGroq` (the real, current integration) |
| `from langchain_chains import ...` — package doesn't exist | Replaced with a small custom retrieve → prompt → stream function (the old `langchain.chains` module was removed entirely in LangChain 1.0) |
| `Groq(model="gpt-3.5-turbo")` — wrong class and a model Groq doesn't serve | `ChatGroq(model="openai/gpt-oss-120b", ...)` — a real, current Groq model |
| `user_question` used but never defined (`st.text_input(...)` return value was discarded) | Proper `st.chat_input` + session state wiring |
| `hkunlp/instructor-xl` embedding model — multi-GB download, very slow on CPU | `sentence-transformers/all-MiniLM-L6-v2` — ~90MB, loads once and is cached, dramatically faster |
| `PyPDF2` (merged into / superseded by `pypdf`) | `pypdf` |
| `page.extract_text()` could return `None` and crash concatenation | Guarded extraction with empty/None handling, plus graceful handling of scanned/encrypted PDFs |
| No streaming, single-shot Q&A box, no chat history shown | Real chat UI (`st.chat_message` / `st.chat_input`), streamed answers, multi-turn history, source citations per answer |
| All libraries outdated/mismatched | Pinned to current versions (LangChain 1.3, LangChain-Groq 1.1, Streamlit 1.58, pypdf 6.13, etc.) |

## Notes

- **Only Groq needs an API key.** Embeddings run locally via a small open-source model (first run downloads it automatically, then it's cached).
- Groq retired `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` in June 2026. The app uses their recommended replacements (`openai/gpt-oss-120b` by default, with `openai/gpt-oss-20b` and `qwen/qwen3.6-27b` as alternatives in the sidebar).
- Scanned/image-only PDFs won't extract text (no OCR included) — the app will tell you which files it couldn't read.
- **Security note:** the Groq key in this `.env` was visible in the file you uploaded. Since it passed through a third-party chat, treat it as exposed and regenerate it at console.groq.com, then paste the new one in.
