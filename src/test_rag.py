from src.pipeline import initialize_rag, generate_answer

if __name__ == "__main__":
    print("Initializing RAG system...")
    collection, num_chunks = initialize_rag()
    print(f"Successfully loaded {num_chunks} chunks into ChromaDB!\n")
    
    # Test query
    test_query = "What does the Payment Intent API do?"
    print(f"Testing Question: {test_query}")
    
    answer, chunks = generate_answer(test_query, collection)
    
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
