import psycopg2
from pgvector.psycopg2 import register_vector
from app import config

DB_CONFIG = dict(
    dbname=config.POSTGRES_DB,
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT,
)

def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn
