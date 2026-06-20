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

as alternatives in the sidebar).
- Scanned/image-only PDFs won't extract text (no OCR included) — the app will tell you which files it couldn't read.
- **Security note:** the Groq key in this `.env` was visible in the file you uploaded. Since it passed through a third-party chat, treat it as exposed and regenerate it at console.groq.com, then paste the new one in.
