import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.1-flash-lite"  # change to a flash-lite name from step 1

SYSTEM = """You are a friendly English speaking partner for a learner.
Speak in short, natural sentences (2-3 sentences max).
Ask one question at a time and keep the conversation going.
If the learner makes a grammar mistake, first reply naturally to what
they said, then briefly show the corrected sentence."""

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    ),
)

def say(text):
    start = time.time()
    first = None
    print("AI: ", end="", flush=True)
    for chunk in chat.send_message_stream(text):
        if first is None:
            first = time.time() - start
        print(chunk.text or "", end="", flush=True)
    print(f"\n[first word after {(first or 0):.1f}s, total {time.time() - start:.1f}s]\n")

print("Type 'quit' to stop.\n")
say("Start the conversation with a friendly greeting.")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    say(user)