import json
import os
import sqlite3
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from passlib.hash import bcrypt
from itsdangerous import URLSafeSerializer, BadSignature

load_dotenv()
client = Groq()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

SYSTEM = """You are a friendly English speaking partner for a learner.
The learner's messages come from speech recognition, so they will have
no punctuation and no capital letters. Do NOT treat missing punctuation
or capitalization as a mistake — ignore that completely.
Never use emoji in your replies.

Every time you reply, follow this exact structure:
1. Respond naturally to what the learner said, 1-2 sentences, staying in the conversation.
2. Ask one follow-up question.
3. Only if the learner's LAST message had a real grammar mistake (wrong verb tense, wrong word, sentence structure — NOT punctuation), add a new line starting with "Correction:" that shows ONLY the learner's sentence fixed. Never correct your own sentences. If there was no mistake, skip this line entirely.
"""

messages = [{"role": "system", "content": SYSTEM}]


def init_db():
    conn = sqlite3.connect("sessions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            grammar REAL, vocabulary REAL, answer_relevance REAL,
            sentence_quality REAL, clarity REAL, keyword_usage REAL,
            overall REAL, sentiment TEXT,
            grammar_mistakes TEXT, strengths TEXT, improvements TEXT
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


def ensure_user_id_column():
    conn = sqlite3.connect("sessions.db")
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.close()


ensure_user_id_column()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this")
serializer = URLSafeSerializer(SECRET_KEY)


def get_current_user(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return serializer.loads(token)  # returns user_id
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


REPORT_PROMPT = """You are analyzing an English speaking-practice conversation
between an AI and a learner. The learner's messages come from speech
recognition, so they have no punctuation and no capital letters. This is
NOT a mistake — never mention missing punctuation, missing capitalization,
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

    convo = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"
    )
    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": convo},
        ],
        reasoning_effort="medium",
        stream=False,
        response_format={"type": "json_object"},
    )
    raw = result.choices[0].message.content

    r = json.loads(raw)
    conn = sqlite3.connect("sessions.db")
    conn.execute(
        """INSERT INTO sessions
           (created_at, grammar, vocabulary, answer_relevance, sentence_quality,
            clarity, keyword_usage, overall, sentiment, grammar_mistakes, strengths, improvements, user_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(),
            r["grammar"], r["vocabulary"], r["answer_relevance"], r["sentence_quality"],
            r["clarity"], r["keyword_usage"], r["overall"], r["sentiment"],
            json.dumps(r["grammar_mistakes"]), json.dumps(r["strengths"]), json.dumps(r["improvements"]),
            user_id,
        ),
    )
    conn.commit()
    conn.close()

    return {"raw": raw}


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


app.mount("/", StaticFiles(directory="static", html=True), name="static")