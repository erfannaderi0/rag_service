import psycopg2
from pgvector.psycopg2 import register_vector

from app import config


def get_connection():
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn
