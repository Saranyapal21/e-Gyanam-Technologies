from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from pinecone import Pinecone



#   Pinecone VectorDB connection initialization
pc = Pinecone(api_key="abcdefghijklmnopqrstuvwxyz-12345678910")    #  A dummy api_key
index_name = "ramayana-embeddings"
index = pc.Index(index_name)



#   Model Initializations
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
reranking_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
llm = ChatOllama(model='llama3', temperature=0.3)
