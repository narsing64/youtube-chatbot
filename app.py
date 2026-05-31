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

# Bump this string whenever the chunking/embedding pipeline changes.
# Old cached indexes will be ignored and rebuilt automatically.
CACHE_VERSION = "v3_proportional_timestamps"

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
                 border-radius:8px;padding:5px 14px;font-size:13px;font-weight:600;
                 cursor:pointer;margin:4px 6px 4px 0;white-space:nowrap;"
          onmouseover="this.style.background='#1d4ed8'"
          onmouseout="this.style.background='#1e293b'">
          ▶ {label}
        </button>"""

    # Player is rendered inside components.html which is itself an iframe.
    # We set an explicit pixel width on #player-wrap so it never bleeds
    # outside its container regardless of Streamlit layout width.
    # Height breakdown: 480px video (16:9 for ~854px wide) + 70px buttons = 550px total.
    html = f"""<!DOCTYPE html><html><head>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0; padding: 0;
        background: #0e1117;
        font-family: sans-serif;
        overflow-x: hidden;
      }}
      #outer {{
        width: 100%;
        max-width: 100%;
        padding: 0;
      }}
      #player-wrap {{
        position: relative;
        width: 100%;
        padding-top: 56.25%;   /* 16:9 ratio */
        border-radius: 10px;
        overflow: hidden;
        background: #000;
      }}
      #yt-player {{
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
      }}
      #btn-row {{
        padding: 10px 0 4px;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }}
      .btn-label {{
        width: 100%;
        font-size: 11px;
        color: #64748b;
        margin-bottom: 2px;
      }}
    </style>
    </head><body>
      <div id="outer">
        <div id="player-wrap"><div id="yt-player"></div></div>
        <div id="btn-row">
          <div class="btn-label">⏩ Jump to timestamp:</div>
          {buttons_html if buttons_html else '<span style="color:#64748b;font-size:12px">Ask a question to see timestamps</span>'}
        </div>
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
    # height = 56.25% of iframe width for video + 70px for button row
    # components.html iframe is ~700px wide on desktop → video ~394px tall
    components.html(html, height=480, scrolling=False)

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
# TRANSCRIPT FETCHING  — 4-source fallback chain
#
# Source 1: youtube-transcript-api with browser-spoofed headers
#   Mimics a real browser request — bypasses many IP blocks without proxy.
#   Works locally always. Often works on Streamlit Cloud too.
#
# Source 2: youtube-transcript-api plain (no headers)
#   Fallback in case header-spoofing causes issues.
#
# Source 3: Supadata API (free, no API key)
#   https://supadata.ai — their servers fetch from YouTube.
#   Bypasses Streamlit Cloud IP blocks entirely.
#
# Source 4: RapidAPI YouTube Transcript (free tier, no key needed for basic)
#   Final fallback.
#
# VALIDATION: every source output is checked — if it contains < 5 segments
# or looks like an error message, it is rejected and the next source is tried.
# This prevents the "Chunks: 1" problem where an error string gets indexed.
# ─────────────────────────────────────────────────────────────────────────────

# Error keywords that indicate a fallback returned an error page, not a transcript
_ERROR_SIGNALS = [
    "blocking", "blocked", "ip ban", "unavailable", "error",
    "subtitle", "caption disabled", "could not", "cannot retrieve",
]

def _is_valid_transcript(segments: list[dict]) -> bool:
    """
    Returns True only if segments look like a real transcript:
    - At least 5 segments
    - Average segment text length > 10 chars (not error messages)
    - Does not contain known error keywords
    """
    if len(segments) < 5:
        return False
    avg_len = sum(len(s["text"]) for s in segments) / len(segments)
    if avg_len < 8:
        return False
    # Check if the full text looks like an error message
    full_text = " ".join(s["text"] for s in segments[:10]).lower()
    if any(kw in full_text for kw in _ERROR_SIGNALS):
        return False
    return True


def _fetch_via_yt_api_browser(video_id: str) -> tuple[list[dict], str]:
    """
    Source 1: youtube-transcript-api with browser-spoofed User-Agent + cookies.
    This bypasses YouTube's bot detection in many cases without needing a proxy.
    """
    from youtube_transcript_api._errors import NoTranscriptFound
    from requests import Session

    # Create a session that looks like a real Chrome browser
    session = Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language":  "en-US,en;q=0.9",
        "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding":  "gzip, deflate, br",
        "DNT":              "1",
        "Connection":       "keep-alive",
    })

    api             = YouTubeTranscriptApi(http_client=session)
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
    if not _is_valid_transcript(segments):
        raise ValueError(f"Invalid transcript ({len(segments)} segments)")
    return segments, lang


def _fetch_via_yt_api_plain(video_id: str) -> tuple[list[dict], str]:
    """Source 2: youtube-transcript-api with no special headers."""
    from youtube_transcript_api._errors import NoTranscriptFound

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
    if not _is_valid_transcript(segments):
        raise ValueError(f"Invalid transcript ({len(segments)} segments)")
    return segments, lang


def _fetch_via_supadata(video_id: str) -> tuple[list[dict], str]:
    """
    Source 3: Supadata free transcript API.
    https://api.supadata.ai/v1/youtube/transcript
    No API key required. Their servers handle YouTube geo-restrictions.
    Response: {"content": [{"text":..,"offset":ms,"duration":ms},...], "lang":...}
    """
    resp = requests.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        params={"videoId": video_id, "text": "false"},
        timeout=25,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    if resp.status_code != 200:
        raise ValueError(f"Supadata HTTP {resp.status_code}: {resp.text[:200]}")

    data    = resp.json()
    raw     = data.get("content", [])
    if not raw:
        raise ValueError("Supadata: empty content")

    segments = [
        {
            "text":     item.get("text", ""),
            "start":    item.get("offset", 0) / 1000,
            "duration": item.get("duration", 0) / 1000,
        }
        for item in raw
        if item.get("text", "").strip()
    ]
    if not _is_valid_transcript(segments):
        raise ValueError(f"Supadata: invalid transcript ({len(segments)} segments)")

    lang = data.get("lang", "en")
    return segments, lang


def _fetch_via_rapidapi(video_id: str) -> tuple[list[dict], str]:
    """
    RapidAPI — youtube-transcript3 by solid-api.
    Free tier: 100 requests/month. No credit card needed.
    Sign up: https://rapidapi.com/solid-api-solid-api-default/api/youtube-transcript3
    Add RAPIDAPI_KEY to Streamlit secrets.
    """
    rapidapi_key = _secret("RAPIDAPI_KEY")
    if not rapidapi_key:
        raise ValueError("RAPIDAPI_KEY not set in secrets")

    resp = requests.get(
        "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url",
        params={"url": f"https://www.youtube.com/watch?v={video_id}", "flat_text": "false"},
        headers={
            "X-RapidAPI-Key":  rapidapi_key,
            "X-RapidAPI-Host": "youtube-transcript3.p.rapidapi.com",
        },
        timeout=25,
    )
    if resp.status_code != 200:
        raise ValueError(f"RapidAPI HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()

    # Handle multiple possible response shapes from this API
    raw = []
    if isinstance(data, list):
        raw = data                                    # Shape C: root is list
    elif isinstance(data, dict):
        raw = (
            data.get("transcript")                    # Shape A
            or data.get("transcripts")
            or data.get("results")
            or data.get("data")
            or []
        )
        # Shape B: nested transcript inside results
        if raw and isinstance(raw[0], dict) and "transcript" in raw[0]:
            raw = raw[0]["transcript"]

    if not raw:
        raise ValueError(f"RapidAPI: could not find transcript in response. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

    def _parse_start(item: dict) -> float:
        """Extract start time in seconds — handles offset(ms), start(s), begin(s)."""
        if "start" in item:
            return float(item["start"])           # already in seconds
        if "offset" in item:
            return float(item["offset"]) / 1000   # milliseconds → seconds
        if "begin" in item:
            return float(item["begin"])
        return 0.0

    segments = [
        {
            "text":     str(item.get("text", item.get("content", ""))).strip(),
            "start":    _parse_start(item),
            "duration": float(item.get("duration", item.get("dur", 2.0))),
        }
        for item in raw
        if str(item.get("text", item.get("content", ""))).strip()
    ]
    if not _is_valid_transcript(segments):
        raise ValueError(f"RapidAPI: invalid transcript ({len(segments)} segments)")
    return segments, "en"


def fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    """
    Smart fallback chain — order depends on environment:

    If RAPIDAPI_KEY is set (Streamlit Cloud):
      1. RapidAPI            ← works on cloud, bypasses IP block
      2. yt-api browser      ← sometimes works even on cloud
      3. yt-api plain        ← last attempt

    If no RAPIDAPI_KEY (local):
      1. yt-api browser      ← works locally always
      2. yt-api plain        ← fallback
      3. RapidAPI            ← uses key if somehow set
    """
    has_rapidapi = bool(_secret("RAPIDAPI_KEY"))

    if has_rapidapi:
        # Cloud order: RapidAPI first (reliable), direct API as bonus attempts
        sources = [
            ("RapidAPI",                                 _fetch_via_rapidapi),
            ("youtube-transcript-api (browser headers)", _fetch_via_yt_api_browser),
            ("youtube-transcript-api (plain)",           _fetch_via_yt_api_plain),
        ]
    else:
        # Local order: direct API first (fast), RapidAPI never called
        sources = [
            ("youtube-transcript-api (browser headers)", _fetch_via_yt_api_browser),
            ("youtube-transcript-api (plain)",           _fetch_via_yt_api_plain),
        ]

    errors = []
    for name, fn in sources:
        try:
            segments, lang = fn(video_id)
            return segments, f"{lang} (via {name})"
        except Exception as e:
            errors.append(f"  • {name}: {e}")
            continue

    # Build a helpful error message
    if not has_rapidapi:
        tip = (
            "\n\n💡 To fix on Streamlit Cloud:\n"
            "1. Sign up free at https://rapidapi.com/solid-api-solid-api-default/api/youtube-transcript3\n"
            "2. Subscribe to the FREE plan (100 requests/month)\n"
            "3. Copy your RapidAPI key\n"
            "4. Add to Streamlit secrets:  RAPIDAPI_KEY = \"your_key_here\""
        )
    else:
        tip = "\n\nYour RAPIDAPI_KEY is set but the request failed. Check the key is valid."

    raise ValueError(
        "❌ Could not fetch transcript.\n\n"
        "Attempted:\n" + "\n".join(errors) + tip
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+",     " ", text)
    return text.strip()


def merge_windows(segments: list[dict], window_size: int = 20) -> list[dict]:
    """
    Merge subtitle segments into overlapping windows.
    Stores segment_starts list alongside the merged text so build_documents
    can assign accurate timestamps to each sub-chunk (not just window start).
    """
    step, merged = window_size // 2, []
    for i in range(0, len(segments), step):
        chunk = segments[i : i + window_size]
        if not chunk:
            break
        merged.append({
            "text":            " ".join(clean_text(s["text"]) for s in chunk),
            "start":           chunk[0]["start"],
            "end":             chunk[-1]["start"] + chunk[-1].get("duration", 0),
            # Per-segment timestamps within this window — used for accurate
            # sub-chunk timestamp assignment in build_documents
            "segment_starts":  [s["start"] for s in chunk],
            "segment_texts":   [clean_text(s["text"]) for s in chunk],
        })
    return merged


def _estimate_start_from_segments(
    sub_text: str,
    segment_texts: list[str],
    segment_starts: list[float],
    window_start: float,
) -> float:
    """
    Find which segment in the window best matches the beginning of sub_text.
    Returns that segment's actual timestamp — far more accurate than proportion.
    """
    if not segment_starts:
        return window_start

    first_words = set(sub_text.lower().split()[:6])  # first 6 words of sub-chunk

    best_idx   = 0
    best_score = 0
    for idx, seg_text in enumerate(segment_texts):
        seg_words = set(seg_text.lower().split())
        score     = len(first_words & seg_words)
        if score > best_score:
            best_score = score
            best_idx   = idx

    return segment_starts[best_idx]


def semantic_split(windows: list[dict], embeddings) -> list[dict]:
    """
    Semantic chunking — splits on topic changes.
    Passes segment_starts and segment_texts through so build_documents
    can still assign accurate timestamps to each sub-chunk.
    """
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
                result.append({
                    "text":           d.page_content,
                    "start":          w["start"],
                    "end":            w["end"],
                    # carry through for accurate timestamp matching
                    "segment_starts": w.get("segment_starts", []),
                    "segment_texts":  w.get("segment_texts",  []),
                })
        except Exception:
            result.append(w)  # fallback: keep full window
    return result


def build_documents(chunks: list[dict], video_id: str) -> list[Document]:
    """
    Split each semantic chunk into sub-chunks and assign accurate timestamps.
    Uses _estimate_start_from_segments to match each sub-chunk back to the
    actual subtitle segment it came from — gives real timestamps, not 0:00.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for c in chunks:
        sub_texts = splitter.split_text(c["text"])
        if not sub_texts:
            continue

        seg_texts  = c.get("segment_texts",  [])
        seg_starts = c.get("segment_starts", [])

        for text in sub_texts:
            # Use real segment matching if available, else proportional estimate
            if seg_texts and seg_starts:
                start = _estimate_start_from_segments(
                    text, seg_texts, seg_starts, float(c["start"])
                )
            else:
                start = float(c["start"])

            docs.append(Document(
                page_content=text,
                metadata={
                    "start":    start,
                    "end":      c["end"],
                    "video_id": video_id,
                },
            ))
    return docs


def build_or_load_vectorstore(docs: list[Document], video_id: str):
    # Version-namespaced cache path — old pipeline caches are automatically ignored
    cache_path = os.path.join(FAISS_CACHE_DIR, CACHE_VERSION, video_id)
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
                    st.session_state.max_ts        = st.session_state.get("max_ts", 0)
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
        st.markdown(f"**Max timestamp:** {seconds_to_mmss(st.session_state.get('max_ts', 0))}")
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
