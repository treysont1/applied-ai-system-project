import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation import load_queries, load_tracks, evaluate_path_metrics, summarize_benchmark_results
from pipeline import get_rag_results, get_hybrid_results, _retrieve_rules
from ranking import extract_preferences


if __name__ == "__main__":
    tracks = load_tracks()
    queries = load_queries()
    per_query_results = []

    for query in queries:
        rag_songs, _ = get_rag_results(query["query"])
        prefs = extract_preferences(query["query"])
        rule_songs = _retrieve_rules(prefs, k=50)
        hybrid_songs = get_hybrid_results(query["query"])

        metrics = evaluate_path_metrics(
            query,
            tracks,
            retrieval_results=rag_songs[:10],
            rule_results=rule_songs[:10],
            hybrid_results=hybrid_songs[:10],
            k=5,
        )
        per_query_results.append(metrics)

        print(query["id"], query["query"])
        for name, path_metrics in metrics.items():
            print(f"  {name} precision@5:", path_metrics["precision@5"])

    summary = summarize_benchmark_results(per_query_results)
    print("\nAggregate summary")
    print("=" * 60)
    for path in ["retrieval", "rule-based", "hybrid"]:
        print(path)
        for metric, value in summary[path].items():
            print(f"  {metric}: {value}")
