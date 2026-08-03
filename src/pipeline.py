"""
Recommendation pipeline orchestrator.

Coordinates:
  validate → extract_prefs → parallel retrieve → fuse → generate
"""

import re
from pathlib import Path

from ranking import extract_preferences, fuse, KEYWORD_BANK
from recommender import load_songs, recommend_songs

try:
    from rag import retrieve, generate, K_RETRIEVE
except Exception:
    retrieve = None
    generate = None
    K_RETRIEVE = 50

# Constants
K_FINAL = 10
LOW_CONFIDENCE_THRESHOLD = 0.3
MIN_QUERY_LENGTH = 3
EXTRA_MUSIC_WORDS = {
    "song", "music", "listen", "playlist", "vibe", "beat", "track",
    "artist", "genre", "mood", "tempo", "energy", "relax", "morning",
    "night", "jazz", "indie",
}
MUSIC_KEYWORDS = set(KEYWORD_BANK.keys()) | EXTRA_MUSIC_WORDS

ROOT = Path(__file__).parent.parent
TRACKS_CSV = ROOT / "data" / "spotify_tracks.csv"
_all_songs = load_songs(str(TRACKS_CSV))


def _validate_query(query: str) -> str | None:
    """
    Returns an error message if the query should be rejected, otherwise None.
    Checks for minimum length and whether the query seems music-related.
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return "Query is too short. Please describe what you want to listen to."

    words = set(query.lower().split())
    if not words & MUSIC_KEYWORDS:
        return (
            "Your query doesn't seem music-related. "
            "Try describing a mood, activity, genre, or vibe (e.g. 'chill music for studying')."
        )

    return None


def _retrieve_rules(prefs: dict, k: int) -> list[dict]:
    if not prefs:
        return []
    scored = recommend_songs(prefs, _all_songs, k=k)
    return [song for song, _score, _explanation in scored]


def _fallback_retrieve(query: str, k: int) -> tuple[list[dict], list[float]]:
    terms = set(re.findall(r"\b\w+\b", query.lower()))
    if not terms:
        return [], []

    scored: list[tuple[dict, float]] = []
    for song in _all_songs:
        song_text = " ".join([
            song.get("title", ""),
            song.get("artist", ""),
            song.get("genre", ""),
            song.get("mood", ""),
            str(song.get("energy", "")),
            str(song.get("tempo_bpm", "")),
            str(song.get("valence", "")),
            str(song.get("danceability", "")),
            str(song.get("acousticness", "")),
        ]).lower()
        song_terms = set(re.findall(r"\b\w+\b", song_text))
        overlap = len(terms & song_terms)
        if overlap > 0:
            scored.append((song, overlap))

    scored.sort(key=lambda item: item[1], reverse=True)
    ranked_songs = [song for song, _ in scored[:k]]
    similarities = [round(min(1.0, score / max(1, len(terms))), 3) for _, score in scored[:k]]
    return ranked_songs, similarities


def get_rag_results(query: str) -> tuple[list[dict], list[float]]:
    if retrieve is None:
        return _fallback_retrieve(query, k=K_RETRIEVE)
    return retrieve(query, k=K_RETRIEVE)


def get_hybrid_results(query: str, respect_confidence: bool = False) -> list[dict]:
    error = _validate_query(query)
    if error:
        return []

    prefs = extract_preferences(query)
    rag_songs, similarities = get_rag_results(query)
    rule_songs = _retrieve_rules(prefs, k=K_RETRIEVE)

    avg_similarity = 0.0
    if similarities:
        avg_similarity = round(sum(similarities) / len(similarities), 3)
    if respect_confidence and avg_similarity < LOW_CONFIDENCE_THRESHOLD:
        return []

    return fuse(rag_songs, rule_songs, k_final=K_FINAL)


def recommend(query: str) -> str:
    """
    Takes a natural language query and returns a Gemini-generated recommendation.
    Requires build_index() to have been run first.
    """
    error = _validate_query(query)
    if error:
        return f"[Input validation] {error}"

    fused_songs = get_hybrid_results(query, respect_confidence=True)
    if not fused_songs:
        return "[Low confidence] The catalog doesn't have strong matches for that query."

    if generate is None:
        return "[Benchmark mode] Hybrid ranking completed, but Gemini generation is unavailable."

    return generate(query, fused_songs)
