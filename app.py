import os
import sqlite3
from datetime import datetime
from typing import Optional
import hashlib

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ------------------------
# Setup & Config
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DB_PATH = "journal.db"

# Audio models (can be overridden in .env)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")        # speech-to-text
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")          # text-to-speech

st.set_page_config(page_title="AI Journal Companion", page_icon="🧠", layout="centered")

# --- Streamlit audio_input compatibility (1.39 uses experimental name) ---
try:
    audio_input_fn = getattr(st, "audio_input")
except Exception:
    audio_input_fn = None
if audio_input_fn is None:
    audio_input_fn = getattr(st, "experimental_audio_input", None)

# ------------------------
# DB helpers
# ------------------------
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )

    # Entries table (per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            mood INTEGER,
            tags TEXT,
            entry TEXT NOT NULL,
            response_mode TEXT NOT NULL,
            response TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


def save_entry(user_id: int, ts: str, mood: Optional[int], tags: str,
               entry: str, response_mode: str, response: str):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO entries (user_id, ts, mood, tags, entry, response_mode, response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, ts, mood, tags, entry, response_mode, response),
    )
    conn.commit()
    conn.close()


def load_entries(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM entries WHERE user_id = ? ORDER BY ts DESC",
        conn,
        params=(user_id,),
    )
    conn.close()
    return df

# ------------------------
# User auth helpers
# ------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> tuple[bool, str]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "Username already taken."
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id, stored_hash = row
    if stored_hash == hash_password(password):
        return {"id": user_id, "username": username}
    return None

# ------------------------
# LLM helpers
# ------------------------
SYSTEM_PROMPT = (
    "You are an empathetic, supportive journaling companion. You are NOT a clinician. "
    "Tone: warm, validating, non-judgmental. Use short paragraphs and bullet points. "
    "Follow the selected mode strictly:\n"
    "- Reflect: Paraphrase feelings, name emotions, highlight values/needs.\n"
    "- Reframe: Gentle CBT-style reframe, spot common thinking traps, offer 1–2 alternative thoughts.\n"
    "- Action Plan: Suggest 2–4 tiny next steps (10-minute tasks), plus a 1-sentence mantra.\n"
    "Always include: a 1-line grounding exercise suggestion.\n"
    "Never offer diagnoses or crisis advice — instead show the safety note."
)


def call_openai(user_text: str, mode: str) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ Missing API key. Add OPENAI_API_KEY to your .env file."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Mode: {mode}\nJournal: {user_text}"},
        ],
        "temperature": 0.7,
    }

    try:
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ API error: {e}"

# ------------------------
# Audio helpers (STT & TTS)
# ------------------------
def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Send microphone recording to Whisper for transcription."""
    if not OPENAI_API_KEY:
        return ""
    try:
        files = {
            "file": (filename, audio_bytes, "audio/webm"),
        }
        data = {
            "model": WHISPER_MODEL,
            "response_format": "text",
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def synthesize_speech(text: str, voice: str = "alloy", format_ext: str = "mp3") -> bytes:
    """Convert text to speech using OpenAI TTS; return audio bytes (mp3)."""
    if not OPENAI_API_KEY:
        return b""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice": voice,
            "format": format_ext,
        }
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")
        return b""

# ------------------------
# App UI
# ------------------------
init_db()  # ensure tables exist

if "user" not in st.session_state:
    st.session_state.user = None

# Safety banner (always visible)
st.info(
    "**Safety note:** This app is for reflection, not medical care. "
    "If you’re in crisis, call local emergency services or a crisis hotline (e.g., 988 in the U.S.).",
    icon="🛟",
)

# Login / signup gate
st.title("💬 AI Journal Companion")

if st.session_state.user is None:
    st.subheader("Login or Sign Up")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in"):
            user = authenticate_user(login_username, login_password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome back, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        signup_username = st.text_input("New username", key="signup_username")
        signup_password = st.text_input("New password", type="password", key="signup_password")
        if st.button("Create account"):
            ok, msg = create_user(signup_username, signup_password)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()  # Don't show journal UI until logged in

# If we reach here, user is logged in
user = st.session_state.user
user_id = user["id"]

st.caption(f"Logged in as **{user['username']}**")
if st.button("Log out"):
    st.session_state.user = None
    st.rerun()

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Response mode", ["Reflect", "Reframe", "Action Plan"], index=0)
    mood = st.slider("Today's mood", 1, 5, 3)
    tags = st.text_input("Tags (comma-separated)", placeholder="work, family, focus")
    st.divider()
    use_mic = st.checkbox("🎙️ Use microphone input", value=False, help="Record instead of typing")
    read_aloud = st.checkbox("🔊 Read reply aloud", value=False)
    voice_choice = st.selectbox("Voice", ["alloy", "verse", "aria", "bright"], index=0, help="Pick a TTS voice")
    st.write("Storage: local SQLite (journal.db)")
    if st.button("Export CSV"):
        df = load_entries(user_id)
        st.download_button(
            label="Download journal.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="journal.csv",
            mime="text/csv",
        )

transcribed_text = ""
journal_text = ""

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if use_mic:
    st.markdown("#### 🎙️ Record a short note")
    if audio_input_fn is None:
        st.warning(
            "Your Streamlit version doesn't include an audio recorder widget. "
            "Upgrade Streamlit (>=1.40) or install a component like `audio-recorder-streamlit`."
        )
        audio_file = None
    else:
        audio_file = audio_input_fn("Press to record", help="Record up to ~60s, then release")

    if audio_file is not None:
        st.session_state.audio_bytes = audio_file.getvalue()
        with st.status("Transcribing…", expanded=False):
            transcribed_text = transcribe_audio(
                st.session_state.audio_bytes,
                filename=(getattr(audio_file, "name", None) or "audio.webm"),
            )
        preview = transcribed_text if len(transcribed_text) <= 120 else (transcribed_text[:120] + "…")
        st.caption(f"Transcribed: {preview}")
        journal_text = transcribed_text
else:
    journal_text = st.text_area(
        "What's on your mind today?",
        height=200,
        placeholder="Free-write for a few minutes. You can talk about your day, a challenge, or a feeling...",
    )

if st.button("Get Companion Reply", type="primary"):
    if not (journal_text and journal_text.strip()):
        st.warning("Please write something or record audio first.")
    else:
        with st.status("Thinking...", expanded=False):
            reply = call_openai(journal_text.strip(), mode)
        ts = datetime.utcnow().isoformat()
        save_entry(user_id, ts, int(mood), tags.strip(), journal_text.strip(), mode, reply)
        st.success("Saved to your journal.")
        st.markdown("### Companion Reply")
        st.write(reply)

        if read_aloud and reply and not reply.startswith("❌"):
            st.markdown("#### 🔊 Audio Reply")
            audio_mp3 = synthesize_speech(reply, voice=voice_choice)
            if audio_mp3:
                st.audio(audio_mp3, format="audio/mp3")

st.divider()
st.subheader("📘 Recent Entries")

df = load_entries(user_id)
if df.empty:
    st.caption("Your journal is empty. Your first entry will appear here.")
else:
    # Simple mood chart
    st.markdown("**Mood over time (last 30 entries)**")
    chart_df = df.head(30).iloc[::-1][["ts", "mood"]].set_index("ts")
    st.line_chart(chart_df)

    # Table view
    st.dataframe(
        df[["ts", "mood", "tags", "response_mode", "entry", "response"]],
        use_container_width=True,
        hide_index=True,
    )
