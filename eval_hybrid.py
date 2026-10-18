import chromadb
from rank_bm25 import BM25Okapi
from eval_retrieval import TESTS, chunk_by_sentence_with_label
from task12_rag_pipeline import load_data, csv_rows_to_texts


def tokenize(text):
    words = [w.strip(".,?!()':;").lower() for w in text.split()]
    return [w for w in words if w]


def hybrid_search(question, docs, collection, bm25, k=3, pool=10, rrf_k=60):
    # ranking 1: by meaning (vectors)
    vec = collection.query(query_texts=[question],
                           n_results=min(pool, len(docs)))["documents"][0]
    # ranking 2: by exact words (keywords)
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:pool]
    kw = [docs[i] for i in order]
    # combine the two rankings (reciprocal rank fusion)
    fused = {}
    for ranking in (vec, kw):
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0) + 1 / (rrf_k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)[:k]


def evaluate_hybrid(k=3):
    raw_text, csv_data = load_data()
    docs = chunk_by_sentence_with_label(raw_text) + csv_rows_to_texts(csv_data)
    client = chromadb.Client()
    collection = client.create_collection(name="eval_hybrid")
    collection.add(documents=docs, ids=[f"h_{i}" for i in range(len(docs))])
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    hits, total_words = 0, 0
    print("--- eval_hybrid (sentence chunks, keyword + vector) ---")
    for question, expected in TESTS:
        top = hybrid_search(question, docs, collection, bm25, k=k)
        joined = " ".join(top)
        total_words += len(joined.split())
        ok = expected.lower() in joined.lower()
        hits += int(ok)
        print("PASS" if ok else "FAIL", "|", question)
        if not ok:
            for d in top:
                print("      got:", d[:90])
    score = hits / len(TESTS) * 100
    print(f"RESULT eval_hybrid: {hits}/{len(TESTS)} = {score:.0f}% recall@{k} "
          f"| avg {total_words / len(TESTS):.0f} words retrieved")


if __name__ == "__main__":
    evaluate_hybrid()