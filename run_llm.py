from model_initialize import embedding_model, reranking_model, llm
from model_initialize import index
# from new import query           #   Query
from langchain.prompts.prompt import PromptTemplate


def run_llm(query):
    
    #   Semantic Search
    xq = embedding_model.encode(query).tolist()
    top_results = index.query(vector=xq, top_k=10, include_metadata=True)
    top_scores = top_results.matches  
    documents = [match['metadata']['text'] for match in top_scores]


    #   Re-ranking
    final_results = reranking_model.rank(query, documents, return_documents=True, top_k=5)
    information = final_results[0]['text']



    #   Prompt
    summary_template = """
        Given the information {information} about a prompt, I want to answer the given query from the Indian epic Ramayana: {query}.
        You can use your knowledge along with the provide information to answer the question. However, it it given that the information passed here
        is exactly correct, although it may have some spelling mistakes. Despite all these, if you don't know the answer, simply say No. Don't try to generate
        wrong answers to the given query.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )



    #   Answer Generation using LLM
    chain = summary_prompt_template | llm
    res = chain.invoke(input= {"information": information,
                            "query": query})


    new_result = {
        "query": query,
        "result": res.content,
        "source_documents": final_results[0]['text']
    }


    return new_result