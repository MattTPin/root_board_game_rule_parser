# query_db.py

import argparse

from langchain_chroma import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from embeddings.get_embeddings import get_embedding_function
from llm.llm_client import LLMClient



CHROMA_PATH = "chroma"


SYSTEM_PROMPT = (
    "You are a rules expert for the board game ROOT. "
    "Answer the provided question concisely using the excerpts from the rules provided. "
    "Simply answer the question, don't allude to having received any rules directly."
)

PROMPT_TEMPLATE = """
QUESTION: {question}
---
OFFICIAL RULES:
{context}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    _embed_and_query_llm(query_text)


def _embed_and_query_llm(query_text: str):
    # Initalize the LLM client (with env settings) and test it
    llm_client = LLMClient()
    llm_client.test_connection()
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)
    
    # Filter out low quality results
    results = [r for r in results if r[1] >= 0.75]  # example similarity threshold

    # Get actual PDF context text
    pdf_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Load the system prompt
    system_prompt = SYSTEM_PROMPT
    
    # Load and setup the "user prompt"
    user_prompt = PROMPT_TEMPLATE.format(
        context=pdf_context_text, 
        question=query_text
    )
    
    print(
        "system prompt is:\n",
        system_prompt,
        "-------------------- user_prompt is:\n\n",
        user_prompt,
        "\n--------------------\n\n"
    )
    
    result, token_count = llm_client.query(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
        expect_json=False,
    )
    
    print("tokens used: ", token_count)
    print("FINAL RESPONSE is:\n", result)
    
    return result


if __name__ == "__main__":
    main()
