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

# ── Retrieval settings ────────────────────────
# TOP_K_RETRIEVE: how many chunks MMR fetches before reranking
# TOP_K_RERANK:   how many of those we keep for the LLM context
# Increasing TOP_K_RETRIEVE gives the reranker more to choose from.
TOP_K_RETRIEVE = 15
TOP_K_RERANK   = 5     # more context → better answers for broad questions

SYSTEM_PROMPT = """
You are a helpful YouTube video assistant.

The user will ask questions about a YouTube video. You are given transcript excerpts from the video, each with timestamps.

Rules:

1. First determine whether the user's question can be answered from the provided transcript context.

2. If the answer is present in the transcript:

   * Answer using ONLY the transcript information.
   * Reference relevant timestamps naturally.
   * Do not add outside knowledge.
   * Start the response with:
     "[From Video]"

3. If the transcript is insufficient to answer the question:

   * State that the information is not discussed in the provided video excerpts.
   * Then answer using general knowledge if you know the answer.
   * Clearly separate this information by starting with:
     "[General Knowledge]"

4. If the transcript is insufficient and you do not know the answer:

   * Respond:
     "I could not find enough information in the video to answer this, and I do not have sufficient general knowledge to answer reliably."

5. Never claim information came from the video unless it is supported by the provided transcript.

6. When answering from the video:

   * Be detailed and clear.
   * Cite timestamps when relevant.
   * Combine information from multiple transcript excerpts if needed.

Examples:

Question: "What does the speaker say about coalition governments?"
Response:
[From Video]
Around 12:15 the speaker explains that coalition governments...

Question: "Where is India located?"
Response:
The provided video excerpts do not discuss India's geographic location.

[General Knowledge]
India is located in South Asia and shares borders with Pakistan, China, Nepal, Bhutan, Bangladesh, and Myanmar.

""".strip()

# ─────────────────────────────────────────────
# CACHED RESOURCES  (API clients only — no downloads)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return JinaEmbeddings(
        jina_api_key=JINA_API_KEY,
        model_name="jina-embeddings-v3",
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    # Using llama3-70b for significantly better answer quality.
    # If you hit rate limits switch back to llama3-8b-8192.

# ─────────────────────────────────────────────
# RERANKER  — pure Python, no model download
# Scores each (query, chunk) pair by counting query-word overlaps
# weighted by chunk length. Good enough to improve ordering with
# zero latency and zero external calls.
# ─────────────────────────────────────────────
def simple_rerank(query: str, docs: list, top_n: int = 5) -> list:
    """
    Lightweight lexical reranker.
    Scores each doc by how many unique query words appear in it,
    normalised by doc length so shorter noisy chunks don't win.
    Falls back to original order on any error.
    """
    if not docs:
        return docs
    try:
        q_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc in docs:
            text  = doc.page_content.lower()
            words = re.findall(r"\w+", text)
            if not words:
                scored.append((0.0, doc))
                continue
            # TF-style: hits / total_words, boosted by unique hits
            hits        = sum(1 for w in words if w in q_words)
            unique_hits = len(set(words) & q_words)
            score = (hits / len(words)) + (unique_hits * 0.1)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
    except Exception:
        return docs[:top_n]

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
# YOUTUBE PLAYER
# The player and the seek buttons live in THE SAME components.html()
# call so postMessage works within one iframe — no cross-frame issue.
# We rebuild this single block every time seek_to changes.
# ─────────────────────────────────────────────
def render_player_with_buttons(video_id: str, sources: list, start_seconds: int = 0):
    """
    Renders the YouTube player AND all timestamp buttons inside one
    self-contained iframe. Because everything shares the same window,
    postMessage works perfectly without any cross-frame permission issues.
    """
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

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; padding:0; background:#0e1117; font-family:sans-serif; }}
      #player-wrap {{ width:100%; aspect-ratio:16/9; border-radius:10px; overflow:hidden; background:#000; }}
      #yt-player {{ width:100%; height:100%; }}
      #btn-row {{ padding:8px 0 4px; }}
      .btn-label {{ font-size:11px; color:#64748b; margin-bottom:4px; }}
    </style>
    </head>
    <body>
      <div id="player-wrap"><div id="yt-player"></div></div>
      <div id="btn-row">
        <div class="btn-label">⏩ Jump to timestamp:</div>
        {buttons_html if buttons_html else '<span style="color:#64748b;font-size:12px">No timestamps yet</span>'}
      </div>

      <script>
        var player;
        var pendingSeek = {start_seconds};

        var tag = document.createElement('script');
        tag.src = 'https://www.youtube.com/iframe_api';
        document.head.appendChild(tag);

        function onYouTubeIframeAPIReady() {{
          player = new YT.Player('yt-player', {{
            videoId: '{video_id}',
            playerVars: {{
              autoplay: 0,
              rel: 0,
              modestbranding: 1,
              start: {start_seconds}
            }},
            events: {{
              onReady: function(e) {{
                if (pendingSeek > 0) {{
                  e.target.seekTo(pendingSeek, true);
                }}
              }}
            }}
          }});
        }}

        function seekTo(seconds) {{
          if (player && player.seekTo) {{
            player.seekTo(seconds, true);
            player.playVideo();
            document.getElementById('player-wrap').scrollIntoView({{behavior:'smooth'}});
          }}
        }}
      </script>
    </body>
    </html>
    """
    # height: 56.25% of width for 16:9 + ~70px for button row
    components.html(html, height=500, scrolling=False)

# ─────────────────────────────────────────────
# TRANSCRIPT PIPELINE
# ─────────────────────────────────────────────
def fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    api             = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    try:
        t    = transcript_list.find_transcript(["en"])
        lang = "en"
    except Exception:
        t    = transcript_list.find_transcript(
            [x.language_code for x in transcript_list]
        )
        lang = t.language_code
        if lang != "en":
            t = t.translate("en")
    raw = t.fetch()
    segments = [
        {
            "text":     s.text     if hasattr(s, "text")     else s["text"],
            "start":    s.start    if hasattr(s, "start")    else s["start"],
            "duration": s.duration if hasattr(s, "duration") else s.get("duration", 0),
        }
        for s in raw
    ]
    return segments, lang


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+",     " ", text)
    return text.strip()


def merge_windows(segments: list[dict], window_size: int = 20) -> list[dict]:
    """
    Merge subtitle lines into overlapping paragraph-like windows.
    Larger window_size = more context per chunk = better semantic retrieval.
    """
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
        chunk_size=700,
        chunk_overlap=150,   # higher overlap = better context continuity
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
        vs = FAISS.load_local(
            cache_path, embeddings, allow_dangerous_deserialization=True
        )
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

    # ── Fast path: FAISS cache hit ────────────
    if os.path.exists(cache_path):
        status_text.text("💾 Loading from cache…")
        progress_bar.progress(70)
        vs = FAISS.load_local(
            cache_path, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_RETRIEVE, "fetch_k": 40, "lambda_mult": 0.7},
        )
        progress_bar.progress(100)
        status_text.text("✅ Loaded from cache!")
        return retriever, video_id, vs.index.ntotal, True, "en (cached)"

    # ── Slow path: fresh processing ───────────
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
def answer_question(query: str, retriever, video_id: str) -> dict:
    llm = load_llm()

    # Step 1: MMR retrieval — diverse set of candidate chunks
    retrieved_docs = retriever.invoke(query)

    # Step 2: Lightweight rerank — push most relevant chunks to top
    top_docs = simple_rerank(query, retrieved_docs, top_n=TOP_K_RERANK)

    # Step 3: Build rich context block for the LLM
    context = ""
    for i, doc in enumerate(top_docs):
        start   = int(doc.metadata["start"])
        ts_mmss = seconds_to_mmss(start)
        link    = create_timestamp_link(doc.metadata["video_id"], start)
        context += (
            f"\n--- SOURCE {i+1} | Timestamp: {ts_mmss} ({start}s) | {link} ---\n"
            f"{doc.page_content}\n"
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== TRANSCRIPT CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
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
    "last_sources":  [],     # sources from the most recent answer
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
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")

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

# ── Player (self-contained iframe with seek buttons) ──
if st.session_state.show_player:
    render_player_with_buttons(
        st.session_state.video_id,
        st.session_state.last_sources,   # buttons for most recent answer
        start_seconds=0,
    )
    st.divider()

# ── Chat history (text only, no buttons — buttons live in player) ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show small timestamp links as plain text under assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            cols = st.columns(len(msg["sources"]))
            for col, src in zip(cols, msg["sources"]):
                ts = seconds_to_mmss(src["start"])
                col.markdown(f"[⏱ {ts}]({src['link']})")

# ── Chat input ────────────────────────────────
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
                ts = seconds_to_mmss(src["start"])
                col.markdown(f"[⏱ {ts}]({src['link']})")

    # Update last_sources so the player shows the new seek buttons
    st.session_state.last_sources = result["sources"]

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    # Rerun so the player iframe rebuilds with updated seek buttons
    st.rerun()