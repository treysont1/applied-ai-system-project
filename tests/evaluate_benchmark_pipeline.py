import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation import load_queries, load_tracks, build_relevant_ids, compute_metrics
from pipeline import get_rag_results, get_hybrid_results


if __name__ == "__main__":
    tracks = load_tracks()
    queries = load_queries()

    for query in queries[:4]:
        rag_songs, _ = get_rag_results(query["query"])
        hybrid_songs = get_hybrid_results(query["query"])
        relevant_ids = build_relevant_ids(query, tracks)

        rag_ids = [f"{song.get('title', '')}|{song.get('artist', '')}" for song in rag_songs[:10]]
        hybrid_ids = [f"{song.get('title', '')}|{song.get('artist', '')}" for song in hybrid_songs[:10]]

        print(query["id"], query["query"])
        print("  rag precision@5:", compute_metrics(rag_ids, relevant_ids, k=5)["precision@5"])
        print("  hybrid precision@5:", compute_metrics(hybrid_ids, relevant_ids, k=5)["precision@5"])
