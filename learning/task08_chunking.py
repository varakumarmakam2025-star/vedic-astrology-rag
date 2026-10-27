def chunk_text(text, chunk_size=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


text = "Leo is a fire sign ruled by the Sun and known for confidence"
result = chunk_text(text)
print(result)
for chunk in result:
    print(chunk)
result = chunk_text(text, chunk_size=3)
print(result)