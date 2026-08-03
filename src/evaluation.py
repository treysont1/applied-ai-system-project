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


def _build_rule(query: dict[str, Any]) -> tuple[dict[str, Any], str]:
    q = query["query"].lower()
    rules: dict[str, Any] = {}

    if "pop" in q:
        rules["genre"] = "pop"
    if "rock" in q:
        rules["genre"] = "rock"
    if "country" in q:
        rules["genre"] = "country"
    if "jazz" in q:
        rules["genre"] = "jazz"
    if "electronic" in q:
        rules["genre"] = "electronic"
    if "folk" in q:
        rules["genre"] = "folk"
    if "indie" in q:
        rules["genre"] = "indie"
    if "happy" in q:
        rules["mood"] = "happy"
    if "sad" in q:
        rules["mood"] = "sad"
    if "upbeat" in q:
        rules["mood"] = "happy"
    if "mellow" in q:
        rules["mood"] = "chill"
    if "relaxing" in q:
        rules["mood"] = "chill"
    if "energetic" in q or "high-energy" in q or "high energy" in q or "fast" in q or "upbeat" in q:
        rules["energy"] = 0.75
    if "low acousticness" in q or "highly acoustic" in q or "acoustic" in q:
        rules["acousticness"] = 0.7 if "highly acoustic" in q or "acoustic" in q else 0.2
    if "danceable" in q or "dance" in q:
        rules["danceability"] = 0.7
    if "valence" in q or "low-valence" in q:
        rules["valence"] = 0.3
    if "150 bpm" in q or "high-tempo" in q or "tempo" in q:
        rules["tempo_bpm"] = 150
    if "popular" in q:
        rules["track_popularity"] = 90

    return rules, q


def _song_id(song: dict[str, Any]) -> str:
    title = song.get("title", "")
    artist = song.get("artist", "")
    if title or artist:
        return f"{title}|{artist}"
    if song.get("id"):
        return str(song["id"])
    return ""


def build_relevant_ids(query: dict[str, Any], songs: list[dict[str, Any]]) -> set[str]:
    rules, _ = _build_rule(query)
    relevant: set[str] = set()

    for song in songs:
        matches = 0
        for field, expected in rules.items():
            if field == "genre":
                if song.get("genre", "").lower() == expected:
                    matches += 1
            elif field == "mood":
                if song.get("mood", "").lower() == expected:
                    matches += 1
            elif field == "energy":
                if _as_float(song.get("energy", 0.0)) >= expected:
                    matches += 1
            elif field == "acousticness":
                if expected == 0.7:
                    if _as_float(song.get("acousticness", 0.0)) >= 0.7:
                        matches += 1
                else:
                    if _as_float(song.get("acousticness", 0.0)) <= expected:
                        matches += 1
            elif field == "danceability":
                if _as_float(song.get("danceability", 0.0)) >= expected:
                    matches += 1
            elif field == "valence":
                if _as_float(song.get("valence", 0.0)) <= expected:
                    matches += 1
            elif field == "tempo_bpm":
                if _as_float(song.get("tempo_bpm", 0.0)) >= expected:
                    matches += 1
            elif field == "track_popularity":
                if _as_float(song.get("track_popularity", 0.0)) >= expected:
                    matches += 1

        if matches == len(rules):
            relevant.add(_song_id(song))

    return relevant


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
