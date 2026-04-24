import numpy as np

class VectorStore:
    def build_index(self, embeddings, texts):
        self.embeddings = np.array(embeddings)
        self.texts = texts

    def search(self, query_embedding, k=5):
        query = np.array(query_embedding)

        # cosine similarity
        similarities = np.dot(self.embeddings, query) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query) + 1e-10
        )

        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [self.texts[i] for i in top_k_idx]