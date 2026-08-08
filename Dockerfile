# ---- Base image: a small Linux + Python 3.11 install ----
FROM python:3.11-slim

# Don't write .pyc files, and print logs immediately instead of buffering them
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages: just curl at this point.
# build-essential and libpq-dev were removed — every Python dependency in
# requirements.txt (psycopg2-binary, pymupdf, tiktoken, transformers) ships
# prebuilt wheels, and torch is installed from PyTorch's prebuilt CPU wheel
# index below, so nothing here actually needs to compile from source.
# This also sidesteps the gcc-14 download failures entirely, since gcc is
# never installed.
#
# Notes on the flags below (kept as a safety net for flaky/VPN connections,
# even though curl alone is a small, fast download):
# - Debian trixie (the base of python:3.11-slim) uses the new deb822 sources
#   format in /etc/apt/sources.list.d/debian.sources, not sources.list.
#   We point it at a German mirror (ftp.de.debian.org) instead of the
#   anycast deb.debian.org, which routes to whichever CDN edge is "closest"
#   and can be inconsistent over a VPN.
# - Acquire::ForceIPv4=true avoids IPv6-over-VPN connections dying mid-download.
# - Acquire::Retries and http::Timeout give apt more patience before giving up.
RUN sed -i 's/deb.debian.org/ftp.de.debian.org/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=5 \
    -o Acquire::http::Timeout=60 \
    -o Acquire::ForceIPv4=true \
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
