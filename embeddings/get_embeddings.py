# get_embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # for backup try model 'all-MiniLM-L6-v2'
    # Use Q&A specific embeddings model!
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    return embeddings
