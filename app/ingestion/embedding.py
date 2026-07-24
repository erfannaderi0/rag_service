from FlagEmbedding import BGEM3FlagModel
from app.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, use_fp16: bool = True, batch_size: int = 16):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # dense_vecs is (n_texts, 1024) — matches your VECTOR(1024) column
        return output["dense_vecs"].tolist()
