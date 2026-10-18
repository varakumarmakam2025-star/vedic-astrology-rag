from google import genai
from src.config import settings
from src.chunking import chunk_text_by_sentence
from src.vector_db import setup_vector_database, get_gemini_embedding

client_llm = genai.Client(api_key=settings.GEMINI_API_KEY)

def initialize_rag():
    """Loads text, chunks it, and sets up the vector database."""
    chunks = chunk_text_by_sentence(settings.DATA_PATH)
    collection = setup_vector_database(chunks)
    return collection, len(chunks)

def retrieve_context(query, collection, top_k=3):
    """Searches the vector database using the user's question embedding."""
    query_embedding = get_gemini_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(query, collection):
    """Retrieves relevant context and uses Gemini to generate an answer."""
    retrieved_chunks = retrieve_context(query, collection)
    context_text = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
    Answer the question based ONLY on the provided context below.
    If the context does not contain the answer, say so clearly.

    Context:
    {context_text}

    Question: {query}
    Answer:
    """
    
    response = client_llm.models.generate_content(
        model=settings.PRIMARY_MODEL,
        contents=prompt
    )
    return response.text, retrieved_chunks