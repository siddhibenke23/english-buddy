from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

for m in client.models.list().data:
    print(m.id)
    