import streamlit as st
import streamlit.components.v1 as components
import os
import re

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import requests  # for Jina reranker API

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="▶️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

GROQ_API_KEY    = _secret("GROQ_API_KEY")
JINA_API_KEY    = _secret("JINA_API_KEY")
FAISS_CACHE_DIR = "faiss_cache"
TOP_K_RETRIEVE  = 15
TOP_K_RERANK    = 5

# ── Proxy config (Streamlit Cloud only) ──────────────────────────────────────
# Streamlit Cloud servers are in the US. Some YouTube videos are geo-blocked
# there even if they work fine locally (India).
# To fix: Settings → Secrets → add:  PROXY_URL = "http://user:pass@host:port"
# Recommended: webshare.io (10 free proxies, works well from India videos)
# Leave absent / empty to use no proxy (default for local runs).
# ─────────────────────────────────────────────────────────────────────────────
PROXY_URL = _secret("PROXY_URL")   # empty string when not set

SYSTEM_PROMPT = """
You are an intelligent YouTube video assistant. You have watched the video
and have access to the most relevant transcript excerpts as context.

Answer using this priority order:

TIER 1 — VIDEO CONTEXT (highest priority):
  If the transcript context directly answers the question, use it.
  Reference timestamps naturally (e.g. "Around 3:56 the speaker explains...").
  For summary questions ("what is the video about"), synthesise across ALL sources.

TIER 2 — GENERAL KNOWLEDGE (fill the gaps):
  If the question asks for a definition, background concept, or explanation
  of a term that the transcript mentions but does not fully explain —
  answer from your general knowledge AND connect it to the video context.
  Example: video is about RAG and user asks "what is an embedding?" →
  explain embeddings from general knowledge, then tie it to how the video uses them.

TIER 3 — HONEST LIMITATION:
  Only say you cannot answer if the question is completely unrelated to both
  the video topic AND general knowledge (e.g. asking about a private detail
  that only the speaker would know).

Style rules:
- Be detailed, clear, and helpful.
- Always prefer answering over refusing.
- Never say "the transcript does not provide" as a full answer — if the
  transcript is thin, supplement with general knowledge.
- Keep answers focused on what would genuinely help someone who just watched
  this video.
""".strip()

# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return JinaEmbeddings(
        jina_api_key=JINA_API_KEY,
        model_name="jina-embeddings-v3",
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    # llama3-70b gives much better answers than 8b.
    # If you hit Groq rate limits, switch to "llama3-8b-8192".
    return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# No reranker resource to load — Jina Reranker v2 is a pure API call.
# Zero download. Zero cold-start delay. Better quality than BGE-base.

# ─────────────────────────────────────────────
# RERANKING  (BGE or lexical fallback)
# ─────────────────────────────────────────────

def rerank(query: str, docs: list, top_n: int) -> list:
    """
    Jina Reranker v2 — pure API call, zero local model download.
    Model: jina-reranker-v2-base-multilingual
    - Supports 100+ languages including Hindi, Telugu, English
    - Better quality than BGE-base
    - Works identically locally and on Streamlit Cloud
    Falls back to lexical scoring if the API call fails.
    """
    if not docs:
        return docs

    try:
        resp = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":     "jina-reranker-v2-base-multilingual",
                "query":     query,
                "documents": [d.page_content for d in docs],
                "top_n":     top_n,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            # Each result: {"index": int, "relevance_score": float, ...}
            ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
            return [docs[r["index"]] for r in ranked[:top_n]]
    except Exception:
        pass

    # Lexical fallback — never breaks the app
    q_words = set(re.findall(r"\w+", query.lower()))
    scored  = []
    for doc in docs:
        words       = re.findall(r"\w+", doc.page_content.lower())
        hits        = sum(1 for w in words if w in q_words)
        unique_hits = len(set(words) & q_words)
        score       = (hits / max(len(words), 1)) + unique_hits * 0.1
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def create_timestamp_link(video_id: str, seconds: int) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

def seconds_to_mmss(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# ─────────────────────────────────────────────
# YOUTUBE PLAYER  (player + seek buttons in one iframe)
# ─────────────────────────────────────────────

def render_player_with_buttons(video_id: str, sources: list, start_seconds: int = 0):
    buttons_html = ""
    for src in sources:
        s     = int(src["start"])
        label = seconds_to_mmss(s)
        buttons_html += f"""
        <button onclick="seekTo({s})"
          style="background:#1e293b;border:1.5px solid #3b82f6;color:#93c5fd;
                 border-radius:8px;padding:5px 16px;font-size:13px;font-weight:600;
                 cursor:pointer;margin:4px 6px 4px 0;"
          onmouseover="this.style.background='#1d4ed8'"
          onmouseout="this.style.background='#1e293b'">
          ▶ {label}
        </button>"""

    html = f"""<!DOCTYPE html><html><head>
    <style>
      body {{ margin:0; padding:0; background:#0e1117; font-family:sans-serif; }}
      #player-wrap {{ width:100%; aspect-ratio:16/9; border-radius:10px; overflow:hidden; background:#000; }}
      #btn-row {{ padding:8px 0 4px; }}
      .btn-label {{ font-size:11px; color:#64748b; margin-bottom:4px; }}
    </style>
    </head><body>
      <div id="player-wrap"><div id="yt-player"></div></div>
      <div id="btn-row">
        <div class="btn-label">⏩ Jump to timestamp:</div>
        {buttons_html if buttons_html else '<span style="color:#64748b;font-size:12px">Ask a question to see timestamps</span>'}
      </div>
      <script>
        var player;
        var tag = document.createElement('script');
        tag.src = 'https://www.youtube.com/iframe_api';
        document.head.appendChild(tag);

        function onYouTubeIframeAPIReady() {{
          player = new YT.Player('yt-player', {{
            videoId: '{video_id}',
            playerVars: {{ autoplay:0, rel:0, modestbranding:1, start:{start_seconds} }},
            events: {{
              onReady: function(e) {{
                if ({start_seconds} > 0) e.target.seekTo({start_seconds}, true);
              }}
            }}
          }});
        }}

        function seekTo(seconds) {{
          if (player && player.seekTo) {{
            player.seekTo(seconds, true);
            player.playVideo();
          }}
        }}
      </script>
    </body></html>"""
    components.html(html, height=500, scrolling=False)

# ─────────────────────────────────────────────
# TRANSCRIPT PIPELINE
# ─────────────────────────────────────────────

def _segments_from_raw(raw) -> list[dict]:
    """Normalise raw transcript snippets to plain dicts."""
    return [
        {
            "text":     s.text     if hasattr(s, "text")     else s["text"],
            "start":    s.start    if hasattr(s, "start")    else s["start"],
            "duration": s.duration if hasattr(s, "duration") else s.get("duration", 0),
        }
        for s in raw
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT FETCHING  — 3-source fallback chain
#
# Source 1: youtube-transcript-api direct
#   Works perfectly locally (your Indian IP is not blocked).
#   On Streamlit Cloud (US IP) YouTube often blocks it.
#
# Source 2: Supadata API  (free, no API key, no signup)
#   https://supadata.ai  — fetches transcripts server-side from their own
#   infrastructure, bypassing the IP block problem entirely.
#   Returns plain text with timestamps.
#
# Source 3: YouTubeTranscript.io API  (free, no API key)
#   Second free fallback if Supadata is down.
#
# This chain means: works locally always, works on Streamlit Cloud via
# free external APIs — zero cost, zero proxy needed.
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_via_yt_api(video_id: str) -> tuple[list[dict], str]:
    """Source 1: direct youtube-transcript-api."""
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
    )
    api             = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        t    = transcript_list.find_transcript(["en"])
        lang = "en"
    except NoTranscriptFound:
        available = [x.language_code for x in transcript_list]
        t         = transcript_list.find_transcript(available)
        lang      = t.language_code
        if lang != "en":
            t    = t.translate("en")
            lang = f"{lang}→en"

    raw      = t.fetch()
    segments = _segments_from_raw(raw)
    if not segments:
        raise ValueError("Empty transcript")
    return segments, lang


def _fetch_via_supadata(video_id: str) -> tuple[list[dict], str]:
    """
    Source 2: Supadata free transcript API.
    Endpoint: https://api.supadata.ai/v1/youtube/transcript
    No API key required. Returns segments with timestamps.
    """
    url  = "https://api.supadata.ai/v1/youtube/transcript"
    resp = requests.get(
        url,
        params={"videoId": video_id, "text": "false"},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    if resp.status_code != 200:
        raise ValueError(f"Supadata returned {resp.status_code}")

    data = resp.json()

    # Supadata response: {"content": [{"text":..,"offset":..,"duration":..},...], "lang":...}
    content = data.get("content", [])
    if not content:
        raise ValueError("Supadata returned empty transcript")

    segments = [
        {
            "text":     item.get("text", ""),
            "start":    item.get("offset", 0) / 1000,   # ms → seconds
            "duration": item.get("duration", 0) / 1000,
        }
        for item in content
        if item.get("text", "").strip()
    ]
    lang = data.get("lang", "en")
    return segments, lang


def _fetch_via_youtubetranscript(video_id: str) -> tuple[list[dict], str]:
    """
    Source 3: youtubetranscript.com free API.
    Endpoint: https://youtubetranscript.com/?server_vid2=VIDEO_ID
    Returns XML-like transcript, parsed manually.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    url  = f"https://youtubetranscript.com/?server_vid2={video_id}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        raise ValueError(f"youtubetranscript.com returned {resp.status_code}")

    # Response is XML: <transcript><text start="0.5" dur="2.3">hello</text></transcript>
    try:
        root     = ET.fromstring(resp.text)
        segments = []
        for el in root.findall("text"):
            text  = el.text or ""
            start = float(el.get("start", 0))
            dur   = float(el.get("dur", 0))
            if text.strip():
                segments.append({"text": text.strip(), "start": start, "duration": dur})
        if not segments:
            raise ValueError("Empty XML transcript")
        return segments, "en"
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}")


def fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    """
    Fetch transcript using a 3-source fallback chain.
    Tries each source in order and returns the first that succeeds.
    """
    sources = [
        ("youtube-transcript-api", _fetch_via_yt_api),
        ("Supadata API",           _fetch_via_supadata),
        ("YouTubeTranscript.com",  _fetch_via_youtubetranscript),
    ]

    last_error = None
    for name, fn in sources:
        try:
            segments, lang = fn(video_id)
            # Tag which source succeeded (shown in sidebar)
            return segments, f"{lang} (via {name})"
        except Exception as e:
            last_error = f"{name}: {e}"
            continue

    raise ValueError(
        f"❌ Could not fetch transcript from any source.\n\n"
        f"Last error: {last_error}\n\n"
        f"This video may have captions disabled, be private, or be deleted."
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+",     " ", text)
    return text.strip()


def merge_windows(segments: list[dict], window_size: int = 20) -> list[dict]:
    step, merged = window_size // 2, []
    for i in range(0, len(segments), step):
        chunk = segments[i : i + window_size]
        if not chunk:
            break
        merged.append({
            "text":  " ".join(clean_text(s["text"]) for s in chunk),
            "start": chunk[0]["start"],
            "end":   chunk[-1]["start"] + chunk[-1].get("duration", 0),
        })
    return merged


def semantic_split(windows: list[dict], embeddings) -> list[dict]:
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    result = []
    for w in windows:
        try:
            docs = chunker.create_documents([w["text"]])
            for d in docs:
                result.append({"text": d.page_content, "start": w["start"], "end": w["end"]})
        except Exception:
            result.append(w)
    return result


def build_documents(chunks: list[dict], video_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for c in chunks:
        for text in splitter.split_text(c["text"]):
            docs.append(Document(
                page_content=text,
                metadata={"start": c["start"], "end": c["end"], "video_id": video_id},
            ))
    return docs


def build_or_load_vectorstore(docs: list[Document], video_id: str):
    cache_path = os.path.join(FAISS_CACHE_DIR, video_id)
    embeddings = load_embeddings()
    if os.path.exists(cache_path):
        vs = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        return vs, True
    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(cache_path, exist_ok=True)
    vs.save_local(cache_path)
    return vs, False

# ─────────────────────────────────────────────
# FULL INGESTION PIPELINE
# ─────────────────────────────────────────────

def process_video(url: str, progress_bar, status_text):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a video ID from that URL.")

    embeddings = load_embeddings()
    cache_path = os.path.join(FAISS_CACHE_DIR, video_id)

    if os.path.exists(cache_path):
        status_text.text("💾 Loading from cache…")
        progress_bar.progress(70)
        vs = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_RETRIEVE, "fetch_k": 40, "lambda_mult": 0.7},
        )
        progress_bar.progress(100)
        status_text.text("✅ Loaded from cache!")
        return retriever, video_id, vs.index.ntotal, True, "en (cached)"

    status_text.text("📥 Step 1/5 — Fetching transcript…")
    progress_bar.progress(5)
    segments, lang = fetch_transcript(video_id)

    status_text.text("🔗 Step 2/5 — Merging subtitle windows…")
    progress_bar.progress(20)
    windows = merge_windows(segments, window_size=20)

    status_text.text("🧠 Step 3/5 — Semantic chunking…")
    progress_bar.progress(38)
    sem_chunks = semantic_split(windows, embeddings)

    status_text.text("✂️ Step 4/5 — Recursive splitting…")
    progress_bar.progress(55)
    docs = build_documents(sem_chunks, video_id)

    status_text.text(f"🔢 Step 5/5 — Embedding {len(docs)} chunks & building FAISS…")
    progress_bar.progress(72)
    vs, _ = build_or_load_vectorstore(docs, video_id)

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVE, "fetch_k": 40, "lambda_mult": 0.7},
    )
    progress_bar.progress(100)
    status_text.text("✅ Done! Ready to chat.")
    return retriever, video_id, len(docs), False, lang

# ─────────────────────────────────────────────
# RAG QUERY PIPELINE
# ─────────────────────────────────────────────

# ── Query type classifier ─────────────────────────────────────────────────────
# Detects whether a query is asking about the video specifically or is a
# general/definitional question. Used to tune retrieval aggressiveness.

_SUMMARY_PATTERNS = re.compile(
    r"\b(what is (the |this )?video|summarize|overview|about this video|"
    r"what does (the |this )?video|explain (the |this )?video|"
    r"what (topics?|subjects?) (does|is)|tell me about (the |this )?video)\b",
    re.IGNORECASE,
)

_DEFINITION_PATTERNS = re.compile(
    r"\b(what is|what are|define|explain|meaning of|definition of|"
    r"how does .+ work|what do you mean by|what does .+ mean)\b",
    re.IGNORECASE,
)

def _classify_query(query: str) -> str:
    """Returns 'summary', 'definition', or 'specific'."""
    if _SUMMARY_PATTERNS.search(query):
        return "summary"
    if _DEFINITION_PATTERNS.search(query):
        return "definition"
    return "specific"


def answer_question(query: str, retriever, video_id: str) -> dict:
    llm        = load_llm()
    query_type = _classify_query(query)

    # For summary questions retrieve more chunks for broader coverage
    k = TOP_K_RERANK + 3 if query_type == "summary" else TOP_K_RERANK

    retrieved_docs = retriever.invoke(query)
    top_docs       = rerank(query, retrieved_docs, top_n=k)

    context = ""
    for i, doc in enumerate(top_docs):
        start   = int(doc.metadata["start"])
        ts_mmss = seconds_to_mmss(start)
        link    = create_timestamp_link(doc.metadata["video_id"], start)
        context += (
            f"\n--- SOURCE {i+1} | Timestamp: {ts_mmss} ({start}s) | {link} ---\n"
            f"{doc.page_content}\n"
        )

    # Add a query-type hint so the LLM knows which tier to use
    type_hint = {
        "summary":    "\n[This is a SUMMARY question — synthesise across all sources.]",
        "definition": "\n[This is a DEFINITION/CONCEPT question — use video context first, then general knowledge if needed.]",
        "specific":   "",
    }[query_type]

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== TRANSCRIPT CONTEXT ===\n{context}\n=== END CONTEXT ==={type_hint}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = llm.invoke(prompt)

    sources = [
        {
            "start": int(d.metadata["start"]),
            "link":  create_timestamp_link(d.metadata["video_id"], int(d.metadata["start"])),
        }
        for d in top_docs
    ]
    return {"answer": response.content, "sources": sources}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "retriever":     None,
    "video_id":      None,
    "n_chunks":      0,
    "from_cache":    False,
    "detected_lang": "—",
    "messages":      [],
    "show_player":   False,
    "last_sources":  [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("▶️ YouTube RAG")
    st.caption("Ask questions about any YouTube video.")
    st.divider()

    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    process_clicked = st.button("Process video", use_container_width=True, type="primary")
    prog_bar    = st.empty()
    status_text = st.empty()

    if process_clicked:
        if not url_input.strip():
            st.error("Please paste a YouTube URL.")
        else:
            new_id = extract_video_id(url_input)
            if new_id == st.session_state.video_id:
                st.info("This video is already loaded.")
            else:
                st.session_state.messages     = []
                st.session_state.retriever    = None
                st.session_state.video_id     = None
                st.session_state.show_player  = False
                st.session_state.last_sources = []

                _bar  = prog_bar.progress(0)
                _text = status_text.empty()

                try:
                    retriever, vid, n, cached, lang = process_video(
                        url_input, _bar, _text
                    )
                    st.session_state.retriever     = retriever
                    st.session_state.video_id      = vid
                    st.session_state.n_chunks      = n
                    st.session_state.from_cache    = cached
                    st.session_state.detected_lang = lang
                    st.session_state.show_player   = True
                    st.rerun()
                except ValueError as e:
                    prog_bar.empty()
                    status_text.empty()
                    st.error(str(e))
                except Exception as e:
                    prog_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Unexpected error: {e}")

    st.divider()

    if st.session_state.video_id:
        st.markdown(f"**Video ID:** `{st.session_state.video_id}`")
        st.markdown(f"**Chunks:** {st.session_state.n_chunks}")
        st.markdown(f"**Language:** {st.session_state.detected_lang}")
        st.markdown(
            "💾 **Cache:** hit" if st.session_state.from_cache
            else "🆕 **Cache:** saved"
        )
        st.divider()
        if st.button(
            "Hide player" if st.session_state.show_player else "Show player",
            use_container_width=True,
        ):
            st.session_state.show_player = not st.session_state.show_player
            st.rerun()
    else:
        st.caption("No video loaded yet.")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("YouTube Video Chatbot")

if not st.session_state.retriever:
    st.info("👈 Paste a YouTube URL in the sidebar and click **Process video** to begin.")
    st.stop()

if st.session_state.show_player:
    render_player_with_buttons(
        st.session_state.video_id,
        st.session_state.last_sources,
        start_seconds=0,
    )
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            cols = st.columns(len(msg["sources"]))
            for col, src in zip(cols, msg["sources"]):
                col.markdown(f"[⏱ {seconds_to_mmss(src['start'])}]({src['link']})")

query = st.chat_input("Ask a question about the video…")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer_question(
                query,
                st.session_state.retriever,
                st.session_state.video_id,
            )
        st.markdown(result["answer"])
        if result["sources"]:
            cols = st.columns(len(result["sources"]))
            for col, src in zip(cols, result["sources"]):
                col.markdown(f"[⏱ {seconds_to_mmss(src['start'])}]({src['link']})")

    st.session_state.last_sources = result["sources"]
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.rerun()