import re
import chromadb
from task12_rag_pipeline import (load_data, csv_rows_to_texts,
                                 chunk_text, chunk_by_paragraph)

# (question, text that must appear in a retrieved chunk)
TESTS = [
    ("What is Leo like?", "confidence"),
    ("What is Aries like?", "courage"),
    ("What is Sagittarius like?", "optimism"),
    ("What do fire signs have in common?", "impulsive"),
    ("What does the Sun represent in a birth chart?", "vitality"),
    ("What does Mars represent?", "willpower"),
    ("What does Jupiter represent?", "growth"),
    ("Which sign is ruled by Venus?", "ruled by Venus"),
    ("Which signs belong to the water element?", "Water sign"),
    ("What are the weaknesses of Aries?", "impatience"),
    ("Which planet governs Sagittarius?", "ruled by Jupiter"),
    ("Which sign is ruled by the Moon?", "ruled by Moon"),
    ("Tell me about the Earth signs", "Earth sign"),
    ("Which signs are air signs?", "Air sign"),
    ("Which sign does Saturn rule?", "ruled by Saturn"),
    ("Which sign likes taking the first step?", "first step"),
    ("Which fire sign values freedom and learning?", "freedom"),
    ("Which sign is generous but can be stubborn?", "stubbornness"),
    ("Which sign has a quick temper?", "quick temper"),
    ("What does the ruler of Leo stand for?", "vitality"),
]


def chunk_by_sentence_with_label(text):
    """One chunk per sentence, prefixed with its paragraph's subject."""
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        label = paragraph.split(" (")[0] if " (" in paragraph else paragraph.split(" ")[0]
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if sentence.strip():
                chunks.append(f"{label}: {sentence.strip()}")
    return chunks


def build_collection(name, note_chunks, csv_texts):
    client = chromadb.Client()
    collection = client.create_collection(name=name)
    docs = note_chunks + csv_texts
    ids = [f"{name}_{i}" for i in range(len(docs))]
    collection.add(documents=docs, ids=ids)
    return collection


def evaluate(name, note_chunks, csv_texts, k=3):
    collection = build_collection(name, note_chunks, csv_texts)
    hits = 0
    total_words = 0
    print(f"--- {name} ({len(note_chunks)} note chunks) ---")
    for question, expected in TESTS:
        results = collection.query(query_texts=[question], n_results=k)
        retrieved_docs = results["documents"][0]
        joined = " ".join(retrieved_docs)
        total_words += len(joined.split())
        ok = expected.lower() in joined.lower()
        hits += int(ok)
        print("PASS" if ok else "FAIL", "|", question)
        if not ok:
            for d in retrieved_docs:
                print("      got:", d[:90])
    score = hits / len(TESTS) * 100
    avg_words = total_words / len(TESTS)
    print(f"RESULT {name}: {hits}/{len(TESTS)} = {score:.0f}% recall@{k} | avg {avg_words:.0f} words retrieved\n")
    return score


if __name__ == "__main__":
    raw_text, csv_data = load_data()
    csv_texts = csv_rows_to_texts(csv_data)
    evaluate("eval_fixed", chunk_text(raw_text), csv_texts)
    evaluate("eval_paragraph", chunk_by_paragraph(raw_text), csv_texts)
    evaluate("eval_sentence", chunk_by_sentence_with_label(raw_text), csv_texts)