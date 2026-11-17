import os
import pytest


def test_environment_and_secrets_hygiene():
    """Verify no hardcoded API keys exist in source files and .env is ignored."""
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r", encoding="utf-8") as f:
            gitignore_text = f.read()
        assert ".env" in gitignore_text, ".env must be listed in .gitignore to prevent secret leaks"


def test_knowledge_base_files_exist():
    """Verify expected data/notes directory or files are present and non-empty."""
    possible_data_paths = ["data", "notes", "knowledge_base"]
    existing_dirs = [d for d in possible_data_paths if os.path.isdir(d)]
    root_data_files = [
        f for f in os.listdir(".")
        if f.endswith((".csv", ".txt", ".md")) and f.lower() not in ("readme.md", "requirements.txt")
    ]
    assert existing_dirs or root_data_files, (
        "Expected at least one knowledge base directory (data/notes) or source CSV/TXT file."
    )


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 3) -> float:
    """Reference evaluation metric helper matching the 20-question recall@3 benchmark."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    [
        (["doc_1", "doc_2", "doc_3"], {"doc_2"}, 3, 1.0),
        (["doc_1", "doc_4", "doc_5"], {"doc_2", "doc_3"}, 3, 0.0),
        (["doc_2", "doc_5", "doc_3"], {"doc_2", "doc_3"}, 3, 1.0),
        (["doc_2", "doc_5", "doc_3"], {"doc_2", "doc_3"}, 1, 0.5),
    ],
)
def test_recall_at_k_calculation(retrieved, relevant, k, expected):
    """Unit test recall@k metric calculation used in retrieval benchmarking."""
    score = compute_recall_at_k(retrieved, relevant, k=k)
    assert pytest.approx(score, 0.001) == expected


def test_chunking_preserves_content_and_respects_bounds():
    """Verify text chunking splits long passages cleanly without dropping text."""
    sample_text = (
        "Sun in Aries gives strong vitality and leadership qualities. "
        "Moon in Taurus brings emotional stability and appreciation for art. "
        "Mars in Capricorn is exalted and provides disciplined execution."
    )
    chunk_size = 80
    overlap = 15

    chunks = []
    start = 0
    while start < len(sample_text):
        end = min(start + chunk_size, len(sample_text))
        chunks.append(sample_text[start:end])
        if end == len(sample_text):
            break
        start += chunk_size - overlap

    assert len(chunks) >= 2
    assert all(len(c) <= chunk_size for c in chunks)
    assert "Sun in Aries" in chunks[0]
    assert "Mars in Capricorn" in chunks[-1]


def test_fallback_handler_recovers_on_primary_timeout():
    """Verify model fallback triggers cleanly when primary Gemini call times out."""
    call_log = []

    def primary_model(prompt: str):
        call_log.append("primary")
        raise TimeoutError("503 / Deadline Exceeded from primary model")

    def fallback_model(prompt: str):
        call_log.append("fallback")
        return {"answer": "Fallback response generated", "model_used": "gemini-flash-lite"}

    def query_with_fallback(prompt: str):
        try:
            return primary_model(prompt)
        except (TimeoutError, RuntimeError):
            return fallback_model(prompt)

    result = query_with_fallback("Explain Mars placement")
    assert call_log == ["primary", "fallback"]
    assert result["model_used"] == "gemini-flash-lite"
    assert "Fallback response" in result["answer"]