
import json
import os
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
import nltk
import textstat
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from groq import Groq
from itsdangerous import URLSafeSerializer, BadSignature
from nltk.tokenize import word_tokenize, sent_tokenize
from passlib.hash import bcrypt
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

client = Groq()
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
DATABASE_FILE = BASE_DIR / "sessions.db"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "pNInz6obpgDQGcFmaJgB"

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this")
serializer = URLSafeSerializer(SECRET_KEY)


# --------------------------------------------------
# HOMEPAGE
# Loads index.html from the same folder as new.py
# --------------------------------------------------

@app.get("/")
def home():
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Homepage not found: {INDEX_FILE}"
        )
    return FileResponse(INDEX_FILE)


# --------------------------------------------------
# AI PROMPTS
# --------------------------------------------------

FRIEND_SYSTEM = """You are a friendly English speaking partner for a learner.

The learner's messages come from speech recognition, so they will have
no punctuation and no capital letters. Do NOT treat missing punctuation
or capitalization as a mistake — ignore that completely.

Never use emoji in your replies.

Every time you reply, follow this exact structure:

1. Respond naturally to what the learner said, in 1-2 sentences.
2. Ask one follow-up question.
3. Only if the learner's last message had a real grammar mistake,
add a new line starting with "Correction:" showing the corrected sentence.
Do not correct punctuation or capitalization.
"""


def interviewer_system(interview_type: str, role: str) -> str:
    style_map = {
        "formal": "Be formal, professional, and businesslike.",
        "informal": "Be warm, conversational, and relaxed.",
        "communication": (
            "Focus on communication, teamwork, conflict resolution, "
            "and soft skills."
        ),
        "technical": "Focus on technical knowledge relevant to the role.",
    }

    style = style_map.get(interview_type, style_map["formal"])

    return f"""You are a professional interviewer conducting a mock
{interview_type} interview for the role of {role}.

{style}

The candidate's messages come from speech recognition. Ignore missing
punctuation and capitalization.

Never use emoji.

Rules:
- Ask ONE question at a time.
- Ask natural follow-up questions.
- Do not correct grammar during the interview.
- Keep responses brief and professional.
- Never repeat a question already asked.
- Start with a brief greeting and your first question.
"""


messages = [{"role": "system", "content": FRIEND_SYSTEM}]
current_mode = "friend"
current_interview_type = "formal"
current_role = "the position"


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            mode TEXT,
            grammar REAL,
            vocabulary REAL,
            answer_relevance REAL,
            sentence_quality REAL,
            clarity REAL,
            keyword_usage REAL,
            overall REAL,
            sentiment TEXT,
            grammar_mistakes TEXT,
            strengths TEXT,
            improvements TEXT,
            type_token_ratio REAL,
            flesch_reading_ease REAL,
            avg_sentence_length REAL,
            total_words INTEGER,
            unique_words INTEGER
        )
    """)

    conn.commit()
    conn.close()


def init_users_table():
    conn = sqlite3.connect(DATABASE_FILE)

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


def ensure_columns():
    conn = sqlite3.connect(DATABASE_FILE)

    columns = [
        ("user_id", "INTEGER"),
        ("mode", "TEXT"),
        ("type_token_ratio", "REAL"),
        ("flesch_reading_ease", "REAL"),
        ("avg_sentence_length", "REAL"),
        ("total_words", "INTEGER"),
        ("unique_words", "INTEGER"),
    ]

    for col, coltype in columns:
        try:
            conn.execute(
                f"ALTER TABLE sessions ADD COLUMN {col} {coltype}"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

    conn.close()


init_db()
init_users_table()
ensure_columns()


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

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
    conn = sqlite3.connect(DATABASE_FILE)

    existing = conn.execute(
        "SELECT id FROM users WHERE email=?",
        (data.email,)
    ).fetchone()

    if existing:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    hashed = bcrypt.hash(data.password)

    cur = conn.execute(
        """
        INSERT INTO users (email, password_hash, created_at)
        VALUES (?, ?, ?)
        """,
        (data.email, hashed, datetime.now().isoformat())
    )

    user_id = cur.lastrowid

    conn.commit()
    conn.close()

    response.set_cookie(
        "session",
        serializer.dumps(user_id),
        httponly=True,
        max_age=60 * 60 * 24 * 30,
        samesite="lax",
    )

    return {"ok": True, "email": data.email}


@app.post("/login")
def login(data: AuthData, response: Response):
    conn = sqlite3.connect(DATABASE_FILE)

    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE email=?",
        (data.email,)
    ).fetchone()

    conn.close()

    if not row or not bcrypt.verify(data.password, row[1]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    response.set_cookie(
        "session",
        serializer.dumps(row[0]),
        httponly=True,
        max_age=60 * 60 * 24 * 30,
        samesite="lax",
    )

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


# --------------------------------------------------
# SPEAKING / INTERVIEW SESSION
# --------------------------------------------------

class StartSession(BaseModel):
    mode: str
    interview_type: str | None = "formal"
    role: str | None = "the position"


@app.post("/start")
def start_session(data: StartSession):
    global messages, current_mode
    global current_interview_type, current_role

    current_mode = data.mode
    current_interview_type = data.interview_type or "formal"
    current_role = data.role or "the position"

    if data.mode == "interviewer":
        messages = [{
            "role": "system",
            "content": interviewer_system(
                current_interview_type,
                current_role,
            ),
        }]
    else:
        messages = [{
            "role": "system",
            "content": FRIEND_SYSTEM,
        }]

    return {"ok": True, "mode": current_mode}


class UserMessage(BaseModel):
    text: str


@app.post("/chat")
def chat(msg: UserMessage):
    messages.append({
        "role": "user",
        "content": msg.text,
    })

    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        reasoning_effort="low",
        stream=False,
    )

    reply = result.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": reply,
    })

    return {"reply": reply}


# --------------------------------------------------
# NLP METRICS
# --------------------------------------------------

def compute_nlp_metrics(learner_text: str) -> dict:
    if not learner_text.strip():
        return {
            "type_token_ratio": 0,
            "flesch_reading_ease": 0,
            "avg_sentence_length": 0,
            "total_words": 0,
            "unique_words": 0,
        }

    try:
        words = [
            word.lower()
            for word in word_tokenize(learner_text)
            if word.isalpha()
        ]

        sentences = sent_tokenize(learner_text) or [learner_text]

    except LookupError:
        return {
            "type_token_ratio": 0,
            "flesch_reading_ease": 0,
            "avg_sentence_length": 0,
            "total_words": 0,
            "unique_words": 0,
            "error": "NLTK tokenizer data is missing.",
        }

    total_words = len(words)
    unique_words = len(set(words))

    ttr = (
        round(unique_words / total_words, 3)
        if total_words else 0
    )

    avg_sentence_len = (
        round(total_words / len(sentences), 2)
        if sentences else 0
    )

    try:
        flesch = round(
            textstat.flesch_reading_ease(learner_text),
            1,
        )
    except Exception:
        flesch = 0

    return {
        "type_token_ratio": ttr,
        "flesch_reading_ease": flesch,
        "avg_sentence_length": avg_sentence_len,
        "total_words": total_words,
        "unique_words": unique_words,
    }


# --------------------------------------------------
# SESSION REPORT
# --------------------------------------------------

REPORT_PROMPT = """Analyze an English speaking-practice or interview
conversation between an AI and a learner.

The learner's messages come from speech recognition. Missing punctuation
and capitalization are NOT mistakes.

Analyze only the learner's messages.

Return ONLY valid JSON in this structure:
{
  "grammar": 0,
  "vocabulary": 0,
  "answer_relevance": 0,
  "sentence_quality": 0,
  "clarity": 0,
  "keyword_usage": 0,
  "overall": 0,
  "sentiment": "Neutral",
  "grammar_mistakes": [],
  "strengths": [],
  "improvements": []
}

Scores must be from 0 to 10. Overall should be the average of the scores.
Include only genuine grammar mistakes.
"""


@app.get("/report")
def report(request: Request):
    learner_text = " ".join(
        item["content"]
        for item in messages
        if item["role"] == "user"
    )

    conversation = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in messages
        if item["role"] != "system"
    )

    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": conversation},
        ],
        reasoning_effort="medium",
        stream=False,
        response_format={"type": "json_object"},
    )

    raw = result.choices[0].message.content
    report_data = json.loads(raw)
    report_data["nlp_metrics"] = compute_nlp_metrics(learner_text)

    # Keep the existing behavior: return the report without saving it.
    return {"raw": json.dumps(report_data)}


# --------------------------------------------------
# SESSION HISTORY
# --------------------------------------------------

@app.get("/history")
def history(request: Request):
    user_id = get_current_user(request)

    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row

    if user_id is None:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE user_id IS NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE user_id=?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# --------------------------------------------------
# AI VOICE
# --------------------------------------------------

@app.post("/speak")
async def speak(msg: UserMessage):
    if not ELEVEN_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY is missing from your .env file.",
        )

    async def audio_stream():
        async with httpx.AsyncClient() as http_client:
            async with http_client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
                headers={
                    "xi-api-key": ELEVEN_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": msg.text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                    },
                },
            ) as api_response:
                if api_response.status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail="The voice service returned an error.",
                    )

                async for chunk in api_response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
    )


# --------------------------------------------------
# EXPORT SESSION REPORT AS PDF
# --------------------------------------------------

@app.post("/export-pdf")
def export_pdf(payload: dict):
    report_data = payload.get("report") or {}
    nlp = report_data.get("nlp_metrics") or {}

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "English Buddy - Session Report",
            styles["Title"],
        ),
        Spacer(1, 10),
        Paragraph(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"],
        ),
        Spacer(1, 14),
        Paragraph("Scores", styles["Heading2"]),
    ]

    score_fields = [
        ("Grammar", "grammar"),
        ("Vocabulary", "vocabulary"),
        ("Answer Relevance", "answer_relevance"),
        ("Sentence Quality", "sentence_quality"),
        ("Clarity", "clarity"),
        ("Keyword Usage", "keyword_usage"),
    ]

    for label, key in score_fields:
        story.append(
            Paragraph(
                f"{label}: {report_data.get(key, 0)}/10",
                styles["Normal"],
            )
        )

    story.extend([
        Paragraph(
            f"<b>Overall: {report_data.get('overall', 0)}/10</b>",
            styles["Normal"],
        ),
        Spacer(1, 14),
        Paragraph("NLP Metrics", styles["Heading2"]),
        Paragraph(
            f"Vocabulary Diversity (TTR): "
            f"{nlp.get('type_token_ratio', 'N/A')}",
            styles["Normal"],
        ),
        Paragraph(
            f"Flesch Reading Ease: "
            f"{nlp.get('flesch_reading_ease', 'N/A')}",
            styles["Normal"],
        ),
        Paragraph(
            f"Average Sentence Length: "
            f"{nlp.get('avg_sentence_length', 'N/A')} words",
            styles["Normal"],
        ),
        Paragraph(
            f"Total Words Spoken: {nlp.get('total_words', 'N/A')}",
            styles["Normal"],
        ),
        Paragraph(
            f"Unique Words Used: {nlp.get('unique_words', 'N/A')}",
            styles["Normal"],
        ),
        Spacer(1, 14),
        Paragraph("Mistakes", styles["Heading2"]),
    ])

    for item in report_data.get("grammar_mistakes", []):
        story.append(
            Paragraph(str(item), styles["Normal"])
        )

    story.extend([
        Spacer(1, 10),
        Paragraph("Strengths", styles["Heading2"]),
    ])

    for item in report_data.get("strengths", []):
        story.append(
            Paragraph(str(item), styles["Normal"])
        )

    story.extend([
        Spacer(1, 10),
        Paragraph("To Improve", styles["Heading2"]),
    ])

    for item in report_data.get("improvements", []):
        story.append(
            Paragraph(str(item), styles["Normal"])
        )

    document.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                "attachment; filename=english_buddy_report.pdf"
        },
    )