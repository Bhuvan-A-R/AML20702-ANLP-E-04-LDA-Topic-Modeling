"""Heuristic, keyword-based topic descriptions. These are suggestions, NOT validated labels."""
from __future__ import annotations

THEME_LEXICON: dict[str, set[str]] = {
    "machine learning methods, model training and predictive analytics": {
        "learning", "model", "data", "training", "algorithm", "prediction", "neural", "network", "deep",
        "language", "text", "loss", "supervised", "reward", "feature", "classification", "learn"},
    "cybersecurity, attacks, encryption and network defence": {
        "attack", "security", "malware", "phishing", "ransomware", "encryption", "password", "breach",
        "vulnerability", "firewall", "network", "authentication", "access", "cyber", "malicious"},
    "healthcare, clinical care and patient treatment": {
        "patient", "doctor", "disease", "hospital", "treatment", "medical", "clinical", "health",
        "medication", "vaccine", "care", "nurse", "therapy", "drug", "diagnose"},
    "education, teaching and student learning": {
        "student", "teacher", "school", "education", "university", "course", "classroom", "exam",
        "curriculum", "academic", "lesson", "training", "skill", "degree"},
    "environment, climate and sustainability": {
        "climate", "carbon", "energy", "emission", "pollution", "species", "forest", "renewable",
        "water", "waste", "wildlife", "ocean", "plastic", "temperature", "conservation"},
    "finance, banking and financial technology": {
        "bank", "financial", "investment", "stock", "market", "loan", "credit", "payment", "money",
        "risk", "insurance", "interest", "blockchain", "cryptocurrency", "transaction", "investor"},
}


def interpret_topic(words: list[str], top_k: int = 7) -> str:
    """Return a hedged sentence describing a topic from its top words."""
    top = words[:top_k]
    if not top:
        return "No words passed the probability threshold, so no interpretation is available."
    scores = {theme: sum(1 for w in words[:10] if w in lex) for theme, lex in THEME_LEXICON.items()}
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    if hits >= 2:
        return (f"This topic appears to concern {best} (matched {hits} of its top 10 words). "
                f"Key terms: {', '.join(top)}. This is a heuristic suggestion that needs human validation.")
    return (f"No predefined theme matched strongly. The topic is characterised by the terms "
            f"{', '.join(top)}; a human reader should decide what, if anything, it represents.")


def interpret_all(topic_words: dict[int, list[str]]) -> dict[int, str]:
    """Interpretations for every topic id."""
    return {k: interpret_topic(v) for k, v in topic_words.items()}
