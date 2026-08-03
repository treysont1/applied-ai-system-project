import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import build_relevant_ids, compute_metrics, evaluate_path_metrics, evaluate_query_appropriateness, summarize_benchmark_results


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


def test_evaluate_query_appropriateness_marks_semantic_queries_for_review():
    songs = [
        {"id": "s1", "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "danceability": 0.7},
        {"id": "s2", "genre": "rock", "mood": "sad", "energy": 0.5, "acousticness": 0.8, "danceability": 0.4},
    ]
    query = {
        "id": "Q21",
        "query": "songs for a rainy day",
        "type": "semantic",
    }

    result = evaluate_query_appropriateness(query, songs)

    assert result["status"] == "semantic-review"
    assert result["appropriate"] is True
    assert result["rule_count"] == 0


def test_evaluate_path_metrics_returns_metrics_for_all_paths():
    songs = [
        {"id": "s1", "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "danceability": 0.7},
        {"id": "s2", "genre": "rock", "mood": "sad", "energy": 0.5, "acousticness": 0.8, "danceability": 0.4},
        {"id": "s3", "genre": "pop", "mood": "happy", "energy": 0.85, "acousticness": 0.1, "danceability": 0.8},
    ]
    query = {
        "id": "Q01",
        "query": "pop songs",
        "type": "structured",
    }

    metrics = evaluate_path_metrics(
        query,
        songs,
        retrieval_results=[{"id": "s1"}, {"id": "s2"}],
        rule_results=[{"id": "s2"}, {"id": "s1"}],
        hybrid_results=[{"id": "s1"}, {"id": "s3"}],
        k=3,
    )

    assert set(metrics) == {"retrieval", "rule-based", "hybrid"}
    assert metrics["retrieval"]["precision@3"] == 1 / 2
    assert metrics["hybrid"]["precision@3"] == 1.0


def test_summarize_benchmark_results_averages_metrics_across_queries():
    per_query_results = [
        {
            "retrieval": {"precision@5": 0.2, "mrr": 0.5},
            "rule-based": {"precision@5": 0.4, "mrr": 0.6},
            "hybrid": {"precision@5": 0.6, "mrr": 0.7},
        },
        {
            "retrieval": {"precision@5": 0.4, "mrr": 0.8},
            "rule-based": {"precision@5": 0.6, "mrr": 0.9},
            "hybrid": {"precision@5": 0.8, "mrr": 1.0},
        },
    ]

    summary = summarize_benchmark_results(per_query_results)

    assert summary["retrieval"]["precision@5"] == 0.3
    assert summary["hybrid"]["mrr"] == 0.85
