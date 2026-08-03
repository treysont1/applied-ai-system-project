import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import build_relevant_ids, compute_metrics


def test_build_relevant_ids_for_structured_query():
    songs = [
        {"id": "s1", "genre": "pop", "energy": 0.82, "tempo_bpm": 120, "mood": "happy"},
        {"id": "s2", "genre": "pop", "energy": 0.62, "tempo_bpm": 100, "mood": "happy"},
        {"id": "s3", "genre": "rock", "energy": 0.85, "tempo_bpm": 140, "mood": "happy"},
        {"id": "s4", "genre": "pop", "energy": 0.81, "tempo_bpm": 160, "mood": "sad"},
    ]

    query = {
        "id": "Q01",
        "query": "energetic pop songs",
        "type": "structured",
    }

    relevant = build_relevant_ids(query, songs)

    assert relevant == {"s1", "s4"}


def test_compute_metrics_returns_expected_values():
    relevant = {"s1", "s3"}
    retrieved = ["s1", "s2", "s3"]

    metrics = compute_metrics(retrieved, relevant, k=5)

    assert metrics["precision@5"] == 2 / 3
    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg@5"] > 0.8
