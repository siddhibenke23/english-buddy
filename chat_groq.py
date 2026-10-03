import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

SYSTEM =  """You are a friendly English speaking partner for a learner.

Every time you reply, follow this exact structure:
1. Respond naturally to what the learner said, 1-2 sentences, staying in the conversation.
2. Ask one follow-up question.
3. Only if the learner's LAST message had a grammar mistake, add a new line starting with "Correction:" that shows ONLY the learner's sentence fixed. Never correct your own sentences. If there was no mistake, skip this line entirely.
"""

messages = [{"role": "system", "content": SYSTEM}]

def ask():
    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        reasoning_effort="low",
        stream=True,
    )
    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""
    return reply

def say(text):
    messages.append({"role": "user", "content": text})
    start = time.time()
    try:
        reply = ask()
    except Exception:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            reasoning_effort="medium",
            stream=False,
        )
        reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"AI: {reply}")
    print(f"[{time.time() - start:.1f}s]\n")

print("Type 'quit' to stop.\n")
say("Start the conversation with a friendly greeting.")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    say(user)