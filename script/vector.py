import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class VectorStore:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # connect to Pinecone
        self.pc = Pinecone(api_key=api_key)
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',  
                    region='us-east-1'
                )
            )
            print(f" Created new Pinecone index: {index_name}")
        else:
            print(f"Reusing existing Pinecone index: {index_name}")


        self.index = self.pc.Index(index_name)

    def add(self, embeddings, chunks):
        vectors = []
        for idx, emb in enumerate(embeddings):
            vectors.append((
                f"chunk-{idx}",
                emb,
                {"text": chunks[idx]["content"], "source": chunks[idx]["source"], "position": chunks[idx]["chunk_index"]}
            ))
        self.index.upsert(vectors)

    def search(self, query_embedding, top_k=5):
        query_embedding = query_embedding
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [
            {
                "text": item["metadata"]["text"],
                "source": item["metadata"]["source"],
                "position": item["metadata"]["position"],
                "score": item["score"]
            }
            for item in results["matches"]
        ]
