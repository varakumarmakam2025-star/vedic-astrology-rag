import os

def chunk_text_by_sentence(file_path="enterprise_docs/stripe_api_docs.txt"):
    """
    Reads the Stripe API documentation and splits it into clean chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find document at: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Your document at {file_path} is empty! Please add some text to it.")

    # Split text line by line or paragraph by paragraph cleanly
    lines = text.split("\n")
    chunks = []
    
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            chunks.append(f"Stripe API Guide: {cleaned_line}")

    # Fallback safety net
    if not chunks:
        chunks.append("Stripe API Guide: Default system overview.")

    print(f"[INFO] Successfully prepared {len(chunks)} text chunks for embedding.")
    return chunks