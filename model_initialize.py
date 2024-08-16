from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from pinecone import Pinecone



#   Pinecone VectorDB connection initialization
pc = Pinecone(api_key="fff4e04f-76f9-4e1e-bb15-db36d7379fdb")
index_name = "ramayana-embeddings"
index = pc.Index(index_name)



#   Model Initializations
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
reranking_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
llm = ChatOllama(model='llama3', temperature=0.3)