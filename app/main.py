from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config
from app.generation.generate import generate
import logging


app = FastAPI(title="RAG Service")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = config.TOP_K


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    return generate(request.query, top_k=request.top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
