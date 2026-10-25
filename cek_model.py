import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Daftar model Groq yang aktif saat ini:")
for model in client.models.list().data:
    print(model.id)