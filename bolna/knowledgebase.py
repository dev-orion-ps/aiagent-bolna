from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Knowledgebase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of the chosen model
        self.documents = []

    def add_document(self, document):
        sentences = document.split('.')
        embeddings = self.model.encode(sentences)
        self.index.add(np.array(embeddings))
        self.documents.extend(sentences)

    def query(self, question, k=5):
        question_embedding = self.model.encode([question])
        _, indices = self.index.search(np.array(question_embedding), k)
        return [self.documents[i] for i in indices[0]]

# Usage example
# kb = Knowledgebase()
# kb.add_document("Your document content here.")
# relevant_info = kb.query("User's question here")
