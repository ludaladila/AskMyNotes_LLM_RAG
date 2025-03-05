from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large') 

def get_embedding(text):
    return embedding_model.encode(text).tolist() 
