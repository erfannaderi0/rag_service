# ---- Base image: a small Linux + Python 3.11 install ----
FROM python:3.11-slim

# Don't write .pyc files, and print logs immediately instead of buffering them
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages needed to build/run some Python libs (psycopg2, fitz, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Everything from here on happens inside /app in the container
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Now install everything else
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY app ./app
COPY db ./db

# The container listens on 8000; this is documentation, it doesn't
# actually publish the port (that happens in docker-compose.yml)
EXPOSE 8000

# Default command: start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
