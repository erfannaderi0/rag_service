import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("GROQ_API_KEY")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-m3"
