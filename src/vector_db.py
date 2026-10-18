import os
import chromadb
from google import genai
from src.config import settings

# Initialize the Gemini client using the new google-genai SDK
client_gemini = genai.Client(api_key=settings.GEMINI_API_KEY)

def get_gemini_embedding(text: str):
    """
    Generates an embedding vector for a given text chunk using Gemini.
    """
    response = client_gemini.models.embed_content(
        model="gemini-embedding-001",  # Updated to the current standard embedding model
        contents=text
    )
    # Extract the embedding values
    return response.embeddings[0].values

def setup_vector_database(chunks):
    """
    Initializes a persistent ChromaDB client, creates a collection,
    and upserts text chunks along with their Gemini embeddings.
    """
    # Use persistent storage in a local directory
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create the collection
    collection = chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME
    )
    
    # If collection is already populated, return it directly
    if collection.count() > 0:
        print(f"[INFO] Loaded existing ChromaDB collection with {collection.count()} items.")
        return collection

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks using Gemini...")
    
    embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
    
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    
    print(f"[INFO] Successfully stored {len(chunks)} chunks in ChromaDB!")
    return collection