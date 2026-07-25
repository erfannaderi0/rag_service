import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
API_KEY = os.getenv("GROQ_API_KEY")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-m3"
