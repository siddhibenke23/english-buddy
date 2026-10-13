import json
import os
import sqlite3
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from passlib.hash import bcrypt
from itsdangerous import URLSafeSerializer, BadSignature
import httpx

import nltk
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()
client = Groq()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

FRIEND_SYSTEM = """You are a friendly English speaking partner for a learner.
The learner's messages come from speech recognition, so they will have
no punctuation and no capital letters. Do NOT treat missing punctuation
or capitalization as a mistake — ignore that completely.
Never use emoji in your replies.

Every time you reply, follow this exact structure:
1. Respond naturally to what the learner said, 1-2 sentences, staying in the conversation.
2. Ask one follow-up question.
3. Only if the learner's LAST message had a real grammar mistake (wrong verb tense, wrong word, sentence structure — NOT punctuation), add a new line starting with "Correction:" that shows ONLY the learner's sentence fixed. Never correct your own sentences. If there was no mistake, skip this line entirely.
"""

def interviewer_system(interview_type: str, role: str) -> str:
    style_map = {
        "formal": "Be formal, professional, and businesslike. Address the candidate respectfully.",
        "informal": "Be warm, conversational, and relaxed, while still asking real interview questions.",
        "communication": "Focus questions on communication skills, teamwork, conflict resolution, and soft skills.",
        "technical": "Focus questions on technical knowledge relevant to the role, with follow-ups that probe depth of understanding.",
    }
    style = style_map.get(interview_type, style_map["formal"])
    return f"""You are a professional interviewer conducting a mock {interview_type} interview
for a candidate applying for the role of {role}. {style}

The candidate's messages come from speech recognition, so they have no
punctuation and no capital letters — ignore that completely, it is not a mistake.
Never use emoji in your replies.

Rules:
- Ask ONE question at a time.
- Ask natural follow-up questions based on the candidate's actual answer.
- Do NOT correct grammar or give feedback during the interview — save that for the final report.
- Keep your own responses brief and professional, like a real interviewer.
- Never repeat a question you already asked.
- Start by briefly greeting the candidate and asking your first question.
"""

messages = [{"role": "system", "content": FRIEND_SYSTEM}]
current_mode = "friend"
current_interview_type = "formal"
current_role = "the position"


def init_db():
    conn = sqlite3.connect("sessions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            mode TEXT,
            grammar REAL, vocabulary REAL, answer_relevance REAL,
            sentence_quality REAL, clarity REAL, keyword_usage REAL,
            overall REAL, sentiment TEXT,
            grammar_mistakes TEXT, strengths TEXT, improvements TEXT,
            type_token_ratio REAL, flesch_reading_ease REAL,
            avg_sentence_length REAL, total_words INTEGER, unique_words INTEGER
        )
    """)
    conn.commit()
    conn.close()


init_db()


def init_users_table():
    conn = sqlite3.connect("sessions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_users_table()


def ensure_columns():
    conn = sqlite3.connect("sessions.db")
    for col, coltype in [
        ("user_id", "INTEGER"), ("mode", "TEXT"),
        ("type_token_ratio", "REAL"), ("flesch_reading_ease", "REAL"),
        ("avg_sentence_length", "REAL"), ("total_words", "INTEGER"), ("unique_words", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {coltype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()


ensure_columns()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this")
serializer = URLSafeSerializer(SECRET_KEY)


def get_current_user(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return serializer.loads(token)
    except BadSignature:
        return None


class AuthData(BaseModel):
    email: str
    password: str


@app.post("/signup")
def signup(data: AuthData, response: Response):
    conn = sqlite3.connect("sessions.db")
    existing = conn.execute("SELECT id FROM users WHERE email=?", (data.email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = bcrypt.hash(data.password)
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?,?,?)",
        (data.email, hashed, datetime.now().isoformat()),
    )
    user_id = cur.lastrowid
    conn.commit()
    conn.close()
    token = serializer.dumps(user_id)
    response.set_cookie("session", token, httponly=True, max_age=60 * 60 * 24 * 30)
    return {"ok": True, "email": data.email}


@app.post("/login")
def login(data: AuthData, response: Response):
    conn = sqlite3.connect("sessions.db")
    row = conn.execute("SELECT id, password_hash FROM users WHERE email=?", (data.email,)).fetchone()
    conn.close()
    if not row or not bcrypt.verify(data.password, row[1]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = serializer.dumps(row[0])
    response.set_cookie("session", token, httponly=True, max_age=60 * 60 * 24 * 30)
    return {"ok": True, "email": data.email}


@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/me")
def me(request: Request):
    user_id = get_current_user(request)
    if user_id is None:
        return {"logged_in": False}
    return {"logged_in": True, "user_id": user_id}


class StartSession(BaseModel):
    mode: str
    interview_type: str | None = "formal"
    role: str | None = "the position"


@app.post("/start")
def start_session(data: StartSession):
    global messages, current_mode, current_interview_type, current_role
    current_mode = data.mode
    current_interview_type = data.interview_type or "formal"
    current_role = data.role or "the position"
    if data.mode == "interviewer":
        messages = [{"role": "system", "content": interviewer_system(current_interview_type, current_role)}]
    else:
        messages = [{"role": "system", "content": FRIEND_SYSTEM}]
    return {"ok": True, "mode": current_mode}


class UserMessage(BaseModel):
    text: str


@app.post("/chat")
def chat(msg: UserMessage):
    messages.append({"role": "user", "content": msg.text})
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        reasoning_effort="low",
        stream=False,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return {"reply": reply}


def compute_nlp_metrics(learner_text: str) -> dict:
    if not learner_text.strip():
        return {"type_token_ratio": 0, "flesch_reading_ease": 0, "avg_sentence_length": 0, "total_words": 0, "unique_words": 0}
    words = [w.lower() for w in word_tokenize(learner_text) if w.isalpha()]
    sentences = sent_tokenize(learner_text) or [learner_text]
    total_words = len(words)
    unique_words = len(set(words))
    ttr = round(unique_words / total_words, 3) if total_words else 0
    avg_sentence_len = round(total_words / len(sentences), 2) if sentences else 0
    try:
        flesch = round(textstat.flesch_reading_ease(learner_text), 1)
    except Exception:
        flesch = 0
    return {"type_token_ratio": ttr, "flesch_reading_ease": flesch, "avg_sentence_length": avg_sentence_len, "total_words": total_words, "unique_words": unique_words}


REPORT_PROMPT = """You are analyzing an English speaking-practice or interview
conversation between an AI and a learner. The learner's messages come from
speech recognition, so they have no punctuation and no capital letters. This
is NOT a mistake — never mention missing punctuation, missing capitalization,
or missing articles caused by casual speech. Only flag real grammar issues:
wrong verb tense, wrong word choice, subject-verb agreement, sentence
structure that would still be wrong even if spoken casually.

Base your analysis only on the learner's messages, not the AI's.

Respond with ONLY valid JSON, no other text, in exactly this shape:
{
  "grammar": <0-10>,
  "vocabulary": <0-10>,
  "answer_relevance": <0-10>,
  "sentence_quality": <0-10>,
  "clarity": <0-10>,
  "keyword_usage": <0-10>,
  "overall": <0-10, average of the above>,
  "sentiment": "<Positive/Neutral/Negative>",
  "grammar_mistakes": [<list of short strings, each "wrong -> correct". ONLY include real mistakes. If a message has no real mistake, do NOT add an entry for it at all>],
  "strengths": [<list of 2-3 short strings>],
  "improvements": [<list of 2-3 short strings, about real grammar/vocabulary issues only, never about punctuation or capitalization>]
}
"""


@app.get("/report")
def report(request: Request):
    user_id = get_current_user(request)
    learner_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system")

    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": REPORT_PROMPT}, {"role": "user", "content": convo}],
        reasoning_effort="medium",
        stream=False,
        response_format={"type": "json_object"},
    )
    raw = result.choices[0].message.content
    r = json.loads(raw)
    nlp = compute_nlp_metrics(learner_text)
    r["nlp_metrics"] = nlp

    conn = sqlite3.connect("sessions.db")
    conn.execute(
        """INSERT INTO sessions
           (created_at, mode, grammar, vocabulary, answer_relevance, sentence_quality,
            clarity, keyword_usage, overall, sentiment, grammar_mistakes, strengths, improvements,
            user_id, type_token_ratio, flesch_reading_ease, avg_sentence_length, total_words, unique_words)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(), current_mode,
            r["grammar"], r["vocabulary"], r["answer_relevance"], r["sentence_quality"],
            r["clarity"], r["keyword_usage"], r["overall"], r["sentiment"],
            json.dumps(r["grammar_mistakes"]), json.dumps(r["strengths"]), json.dumps(r["improvements"]),
            user_id, nlp["type_token_ratio"], nlp["flesch_reading_ease"],
            nlp["avg_sentence_length"], nlp["total_words"], nlp["unique_words"],
        ),
    )
    conn.commit()
    conn.close()
    return {"raw": json.dumps(r)}


@app.get("/history")
def history(request: Request):
    user_id = get_current_user(request)
    conn = sqlite3.connect("sessions.db")
    conn.row_factory = sqlite3.Row
    if user_id is None:
        rows = conn.execute("SELECT * FROM sessions WHERE user_id IS NULL ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post("/speak")
async def speak(msg: UserMessage):
    async def audio_stream():
        async with httpx.AsyncClient() as client_http:
            async with client_http.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
                headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"},
                json={"text": msg.text, "model_id": "eleven_turbo_v2_5", "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


@app.get("/export-pdf")
def export_pdf(request: Request):
    user_id = get_current_user(request)
    conn = sqlite3.connect("sessions.db")
    conn.row_factory = sqlite3.Row
    if user_id is None:
        row = conn.execute("SELECT * FROM sessions WHERE user_id IS NULL ORDER BY created_at DESC LIMIT 1").fetchone()
    else:
        row = conn.execute("SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No session found")

    s = dict(row)
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("English Buddy — Session Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Mode: {s.get('mode','friend')} &nbsp;&nbsp; Date: {s['created_at']}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Scores", styles["Heading2"]),
        Paragraph(f"Grammar: {s['grammar']}/10", styles["Normal"]),
        Paragraph(f"Vocabulary: {s['vocabulary']}/10", styles["Normal"]),
        Paragraph(f"Answer Relevance: {s['answer_relevance']}/10", styles["Normal"]),
        Paragraph(f"Sentence Quality: {s['sentence_quality']}/10", styles["Normal"]),
        Paragraph(f"Clarity: {s['clarity']}/10", styles["Normal"]),
        Paragraph(f"Keyword Usage: {s['keyword_usage']}/10", styles["Normal"]),
        Paragraph(f"<b>Overall: {s['overall']}/10</b>", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("NLP Metrics (computed)", styles["Heading2"]),
        Paragraph(f"Vocabulary Diversity (Type-Token Ratio): {s.get('type_token_ratio')}", styles["Normal"]),
        Paragraph(f"Flesch Reading Ease: {s.get('flesch_reading_ease')}", styles["Normal"]),
        Paragraph(f"Avg. Sentence Length: {s.get('avg_sentence_length')} words", styles["Normal"]),
        Paragraph(f"Total Words Spoken: {s.get('total_words')}", styles["Normal"]),
        Paragraph(f"Unique Words Used: {s.get('unique_words')}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Mistakes", styles["Heading2"]),
    ]
    for m in json.loads(s["grammar_mistakes"] or "[]"):
        story.append(Paragraph(f"• {m}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Strengths", styles["Heading2"]))
    for st in json.loads(s["strengths"] or "[]"):
        story.append(Paragraph(f"✓ {st}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("To Improve", styles["Heading2"]))
    for imp in json.loads(s["improvements"] or "[]"):
        story.append(Paragraph(f"⚠ {imp}", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=english_buddy_report.pdf"})


app.mount("/", StaticFiles(directory="static", html=True), name="static")