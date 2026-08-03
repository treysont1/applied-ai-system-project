import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
TRACKS_CSV = ROOT / "data" / "spotify_tracks.csv"
QUERIES_JSON = ROOT / "benchmark" / "queries.json"


def load_tracks(csv_path: str | Path = TRACKS_CSV) -> list[dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["id"] = _song_id(row)

    return rows


def load_queries(json_path: str | Path = QUERIES_JSON) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _as_float(value: Any) -> float:
    return float(value)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _threshold_for(songs: list[dict[str, Any]], field: str, direction: str, fallback: float) -> float:
    values = []
    for song in songs:
        raw = song.get(field)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return fallback
    if len(values) < 5:
        return fallback
    if direction == "high":
        return round(_percentile(values, 0.75), 3)
    if direction == "low":
        return round(_percentile(values, 0.25), 3)
    return round(_percentile(values, 0.5), 3)


def _build_rule(query: dict[str, Any], songs: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], str]:
    q = query["query"].lower()
    rules: dict[str, Any] = {}

    if songs is None:
        songs = []

    energy_threshold = _threshold_for(songs, "energy", "high", 0.75)
    acoustic_threshold_high = _threshold_for(songs, "acousticness", "high", 0.7)
    acoustic_threshold_low = _threshold_for(songs, "acousticness", "low", 0.2)
    danceability_threshold = _threshold_for(songs, "danceability", "high", 0.7)
    valence_threshold_low = _threshold_for(songs, "valence", "low", 0.3)
    tempo_threshold = _threshold_for(songs, "tempo_bpm", "high", 150)
    popularity_threshold = _threshold_for(songs, "track_popularity", "high", 90)
    instrumental_threshold = _threshold_for(songs, "instrumentalness", "high", 0.5)

    genre_terms = ("pop", "rock", "country", "jazz", "electronic", "folk", "indie")
    for genre in genre_terms:
        if genre in q:
            rules["genre"] = genre
            break

    if "happy" in q:
        rules["mood"] = "happy"
    if "sad" in q:
        rules["mood"] = "sad"
    if "mellow" in q or "relaxing" in q or "chill" in q:
        rules["mood"] = "chill"

    if any(term in q for term in ["energetic", "high-energy", "high energy", "fast", "upbeat", "workout"]):
        rules["energy"] = {"op": ">=", "value": energy_threshold}
    if "instrumental" in q:
        rules["instrumentalness"] = {"op": ">=", "value": instrumental_threshold}
    if "low acousticness" in q:
        rules["acousticness"] = {"op": "<=", "value": acoustic_threshold_low}
    elif "highly acoustic" in q or "acoustic" in q:
        rules["acousticness"] = {"op": ">=", "value": acoustic_threshold_high}
    if "danceable" in q or "dance" in q:
        rules["danceability"] = {"op": ">=", "value": danceability_threshold}
    if "low-valence" in q or "low valence" in q:
        rules["valence"] = {"op": "<=", "value": valence_threshold_low}
    if "150 bpm" in q or "high-tempo" in q or "high tempo" in q or "tempo" in q:
        rules["tempo_bpm"] = {"op": ">=", "value": tempo_threshold}
    if "popular" in q:
        rules["track_popularity"] = {"op": ">=", "value": popularity_threshold}

    return rules, q


def _song_id(song: dict[str, Any]) -> str:
    title = song.get("title", "")
    artist = song.get("artist", "")
    if title or artist:
        return f"{title}|{artist}"
    if song.get("id"):
        return str(song["id"])
    return ""


def _satisfies_rule(song: dict[str, Any], field: str, expected: Any) -> bool:
    if isinstance(expected, dict):
        op = expected.get("op", ">=")
        value = expected.get("value", 0)
    else:
        op = ">="
        value = expected

    if field == "genre":
        return str(song.get("genre", "")).lower() == str(value).lower()
    if field == "mood":
        return str(song.get("mood", "")).lower() == str(value).lower()

    current = song.get(field)
    if current is None:
        return False
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return False

    if op == ">=":
        return current_value >= float(value)
    if op == "<=":
        return current_value <= float(value)
    return False


def build_relevant_ids(query: dict[str, Any], songs: list[dict[str, Any]]) -> set[str]:
    rules, _ = _build_rule(query, songs)
    relevant: set[str] = set()

    for song in songs:
        matches = 0
        for field, expected in rules.items():
            if _satisfies_rule(song, field, expected):
                matches += 1

        if matches == len(rules):
            relevant.add(_song_id(song))

    return relevant


def evaluate_query_appropriateness(query: dict[str, Any], songs: list[dict[str, Any]]) -> dict[str, Any]:
    rules, _ = _build_rule(query, songs)
    relevant = build_relevant_ids(query, songs)
    query_type = query.get("type", "unknown")

    if not rules:
        if query_type == "semantic":
            return {
                "appropriate": True,
                "status": "semantic-review",
                "relevant_count": len(relevant),
                "rule_count": 0,
                "reason": "No metadata rules could be inferred from the query, so it should be reviewed manually.",
            }
        return {
            "appropriate": False,
            "status": "needs-manual-review",
            "relevant_count": len(relevant),
            "rule_count": 0,
            "reason": "No metadata rules could be inferred from the query.",
        }

    if query_type == "semantic":
        return {
            "appropriate": len(relevant) >= 3,
            "status": "semantic-review",
            "relevant_count": len(relevant),
            "rule_count": len(rules),
            "reason": "Semantic queries need manual review because the metadata rules are only approximate.",
        }

    if len(relevant) >= 10:
        status = "strong-fit"
        appropriate = True
    elif len(relevant) >= 3:
        status = "moderate-fit"
        appropriate = True
    else:
        status = "weak-fit"
        appropriate = False

    return {
        "appropriate": appropriate,
        "status": status,
        "relevant_count": len(relevant),
        "rule_count": len(rules),
        "reason": "The query maps to metadata rules and the catalog has enough matching songs.",
    }


def compute_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    retrieved_k = retrieved_ids[:k]
    hits = [rid for rid in retrieved_k if rid in relevant_ids]
    denominator = min(k, len(retrieved_k)) if retrieved_k else 0

    precision = len(hits) / denominator if denominator else 0.0
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
    mrr = 0.0
    for rank, rid in enumerate(retrieved_k, start=1):
        if rid in relevant_ids:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, rid in enumerate(retrieved_k, start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant = len(relevant_ids)
    ideal_dcg = 0.0
    for rank in range(1, min(ideal_relevant, len(retrieved_k)) + 1):
        ideal_dcg += 1.0 / math.log2(rank + 1)

    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "precision@5": precision,
        "precision@3": len(hits[:3]) / min(3, len(retrieved_k)) if retrieved_k else 0.0,
        "recall@5": recall,
        "mrr": mrr,
        "ndcg@5": ndcg,
    }


def evaluate_path_metrics(
    query: dict[str, Any],
    songs: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    rule_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, dict[str, float]]:
    relevant_ids = build_relevant_ids(query, songs)

    def _to_ids(results: list[dict[str, Any]]) -> list[str]:
        ids = []
        for song in results:
            title = song.get("title", "")
            artist = song.get("artist", "")
            if title or artist:
                ids.append(f"{title}|{artist}")
            elif song.get("id"):
                ids.append(str(song["id"]))
            else:
                ids.append("")
        return ids

    return {
        "retrieval": compute_metrics(_to_ids(retrieval_results), relevant_ids, k),
        "rule-based": compute_metrics(_to_ids(rule_results), relevant_ids, k),
        "hybrid": compute_metrics(_to_ids(hybrid_results), relevant_ids, k),
    }


def summarize_benchmark_results(per_query_results: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not per_query_results:
        return {}

    paths = ["retrieval", "rule-based", "hybrid"]
    metrics = ["precision@5", "precision@3", "recall@5", "mrr", "ndcg@5"]
    summary: dict[str, dict[str, float]] = {}

    for path in paths:
        summary[path] = {}
        for metric in metrics:
            values = [result[path][metric] for result in per_query_results if metric in result[path]]
            if not values:
                summary[path][metric] = 0.0
            else:
                summary[path][metric] = round(sum(values) / len(values), 4)

    return summary
