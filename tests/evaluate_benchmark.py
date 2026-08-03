import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation import load_queries, load_tracks, build_relevant_ids, compute_metrics


if __name__ == "__main__":
    tracks = load_tracks()
    queries = load_queries()

    print("Benchmark preview")
    print("=" * 60)
    for query in queries[:6]:
        relevant = build_relevant_ids(query, tracks)
        print(query["id"], query["query"], "->", len(relevant), "relevant songs")

    print("\nUse this script as a scaffold for comparing baseline vs hybrid outputs.")
