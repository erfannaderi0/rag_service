import psycopg2
from pgvector.psycopg2 import register_vector

from app import config


def get_connection():
    if config.NEON_DATABASE_URL:
        conn = psycopg2.connect(config.NEON_DATABASE_URL)
    else:
        conn = psycopg2.connect(
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
        )

    register_vector(conn)
    return conn
