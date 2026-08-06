import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from app import config


load_dotenv()


def main():
    # Local PostgreSQL exposed by Docker on localhost:5433.
    local_conn = psycopg2.connect(
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host="localhost",
        port=5433,
    )

    # Neon PostgreSQL.
    neon_conn = psycopg2.connect(config.NEON_DATABASE_URL)

    register_vector(local_conn)
    register_vector(neon_conn)

    local_cur = local_conn.cursor()
    neon_cur = neon_conn.cursor()

    # Read all existing chunks and embeddings.
    local_cur.execute(
        """
        SELECT content, metadata, embedding
        FROM documents
        ORDER BY id
        """
    )

    rows = local_cur.fetchall()

    print(f"Found {len(rows)} rows in local database.")

    rows = [
        (content, Json(metadata), embedding)
        for content, metadata, embedding in rows
    ]

    # Insert existing chunks + metadata + embeddings into Neon.
    neon_cur.executemany(
        """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s)
        """,
        rows,
    )

    neon_conn.commit()

    # Verify migration.
    neon_cur.execute("SELECT COUNT(*) FROM documents")
    count = neon_cur.fetchone()[0]

    print(f"Neon now contains {count} rows.")

    local_cur.close()
    neon_cur.close()
    local_conn.close()
    neon_conn.close()


if __name__ == "__main__":
    main()
