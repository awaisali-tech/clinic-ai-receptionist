import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

DATA_PATH = "data/clinic_data.json"

VECTORSTORE_PATH = "vectorstore/faiss"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"