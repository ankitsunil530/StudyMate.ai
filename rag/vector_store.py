import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build_index(self, embeddings, texts):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts = texts

    def search(self, query_embedding, k=5):
        D, I = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        return [self.texts[i] for i in I[0]]