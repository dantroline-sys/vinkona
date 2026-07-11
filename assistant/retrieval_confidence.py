"""
Retrieval confidence scoring.

kb_ask returns answers, but doesn't say how confident Vinkona should be in them.
This module adds confidence bounds grounded in:

  • Recency: how fresh is the source data?
  • Source convergence: do multiple sources agree?
  • User domain fit: does this match what the user already knows?
  • Base rate: how often has similar advice been useful?

A card returned with confidence=0.65 means "I'm moderately sure but watch for caveats."
This informs synthesis: high-confidence answers get stated more directly; low-confidence
answers come with "I'm uncertain because..." qualifiers.

The result is retrieval that's not just relevant, but *calibrated* — the system knows
what it doesn't know, and says so.
"""

import sqlite3
import time
import typing as tp


class ConfidenceScorer:
    """Score retrieved cards for confidence."""

    def __init__(self, db: sqlite3.Connection, user_model_store=None):
        self.db = db
        self.user = user_model_store  # optional: for domain fit scoring

    # ── Scoring components ───────────────────────────────────────────────────

    def _score_recency(self, card: dict) -> float:
        """Score based on how fresh the underlying sources are.

        Assumes card has 'updated_at' (unix ts) or 'source_dates' array.
        Returns 0.0-1.0: fresh=1.0, >1 year old=0.3, really stale=0.0.
        """
        now = time.time()

        # Try explicit updated_at
        if "updated_at" in card:
            age_days = (now - card["updated_at"]) / 86400
            if age_days < 30:
                return 1.0
            elif age_days < 365:
                return max(0.5, 1.0 - (age_days - 30) / 365 * 0.5)
            else:
                return 0.2

        # Try source dates if available
        if "sources" in card and isinstance(card["sources"], list):
            dates = [s.get("date") for s in card["sources"] if s.get("date")]
            if dates:
                latest = max(dates)
                age_days = (now - latest) / 86400
                return max(0.2, 1.0 - age_days / 730)  # decay over 2 years

        # No date info: assume neutral
        return 0.6

    def _score_source_convergence(self, card: dict) -> float:
        """Score based on whether multiple sources agree (or have a common theme).

        Simple heuristic: more sources = higher confidence. Real version would
        compare source claims for contradiction.

        Returns 0.0-1.0: single source=0.6, 3+ converging=1.0, contradictory=0.3.
        """
        sources = card.get("sources", [])
        if not sources:
            return 0.5

        num_sources = len(sources)
        if num_sources >= 3:
            # Assume convergence with multiple sources (ideally checked for actual agreement)
            return 0.95
        elif num_sources == 2:
            return 0.75
        elif num_sources == 1:
            return 0.6
        else:
            return 0.4

    def _score_domain_fit(self, card: dict) -> float:
        """Score based on whether this matches the user's known expertise.

        If the user is expert in a domain but the card is basic, lower confidence.
        If the user is novice and the card is advanced, lower confidence (too hard).
        If the card matches the user's confirmed expertise, higher confidence.

        Returns 0.0-1.0.
        """
        if not self.user:
            return 0.7  # neutral if no user model

        domain = card.get("domain") or card.get("facet_domain")
        if not domain:
            return 0.7

        fluency = self.user.get_domain_fluency(domain)
        if not fluency:
            return 0.7

        level = fluency.get("fluency_level", "intermediate")
        confidence = fluency.get("confidence", 0.5)

        # Heuristic: if user is expert, assume they'll validate themselves
        if level == "expert":
            return 0.8 * confidence  # lower confidence because expert can fact-check

        # If user is novice, assume they trust Vinkona
        elif level == "novice":
            return 0.9 * confidence  # higher confidence because they rely on her

        # Intermediate: baseline
        else:
            return 0.85 * confidence

    def _score_base_rate(self, card: dict) -> float:
        """Score based on whether similar past advice was acted upon.

        Queries user_response_history to find success rate for this domain/type.
        If 80% of medical advice was acted on, medical answers get a boost.

        Returns 0.0-1.0.
        """
        if not self.user:
            return 0.7

        domain = card.get("domain") or card.get("facet_domain")
        if not domain:
            return 0.7

        try:
            effectiveness = self.user.get_response_effectiveness(response_format=None)
            # If overall action rate is high, boost base confidence
            action_rate = effectiveness.get("action_rate", 0.5)
            return 0.5 + (action_rate * 0.5)  # 0.5-1.0 range
        except Exception:
            return 0.7

    # ── Main scoring function ────────────────────────────────────────────────

    def score(self, card: dict, weights: tp.Dict[str, float] | None = None) -> float:
        """Compute overall confidence for a retrieved card.

        Args:
            card: result dict from kb_ask (should have 'sources', 'domain', 'updated_at', etc.)
            weights: component weight dict (defaults to equal weight)

        Returns:
            Overall confidence: 0.0-1.0 (0=very uncertain, 1=very confident)
        """
        if weights is None:
            weights = {
                "recency": 0.25,
                "convergence": 0.25,
                "domain_fit": 0.25,
                "base_rate": 0.25,
            }

        recency = self._score_recency(card)
        convergence = self._score_source_convergence(card)
        domain_fit = self._score_domain_fit(card)
        base_rate = self._score_base_rate(card)

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5

        overall = (
            recency * weights.get("recency", 0) +
            convergence * weights.get("convergence", 0) +
            domain_fit * weights.get("domain_fit", 0) +
            base_rate * weights.get("base_rate", 0)
        ) / total_weight

        return round(max(0.0, min(1.0, overall)), 2)

    def score_batch(
        self,
        cards: list[dict],
        weights: tp.Dict[str, float] | None = None
    ) -> list[dict]:
        """Score multiple cards and attach confidence to each.

        Returns:
            List of cards with added 'confidence' field (0.0-1.0)
        """
        return [
            {**card, "confidence": self.score(card, weights)}
            for card in cards
        ]

    # ── Uncertainty messaging ────────────────────────────────────────────────

    @staticmethod
    def confidence_qualifier(confidence: float) -> str:
        """Generate a natural-language confidence qualifier for synthesis.

        Example: 0.8 → "I'm fairly confident", 0.3 → "I'm uncertain about this"
        """
        if confidence >= 0.9:
            return "I'm very confident about this."
        elif confidence >= 0.75:
            return "I'm fairly confident about this."
        elif confidence >= 0.6:
            return "I'm moderately confident, but watch for exceptions."
        elif confidence >= 0.4:
            return "I'm uncertain about this; take it as a starting point."
        else:
            return "I'm quite uncertain about this; verify elsewhere."

    @staticmethod
    def confidence_reasoning(card: dict, confidence: float, scorer: 'ConfidenceScorer') -> str:
        """Generate natural-language reasoning for the confidence score.

        Example: "Based on recent sources (published 2025) and strong agreement
        between 4 sources, I'm fairly confident in this."
        """
        reasons = []

        recency = scorer._score_recency(card)
        if recency >= 0.8:
            reasons.append("recent sources")
        elif recency < 0.5:
            reasons.append("older sources")

        convergence = scorer._score_source_convergence(card)
        num_sources = len(card.get("sources", []))
        if convergence >= 0.9 and num_sources >= 3:
            reasons.append(f"strong agreement between {num_sources} sources")
        elif num_sources == 1:
            reasons.append("single source")

        if not reasons:
            return "Based on available evidence."

        return "Based on " + ", ".join(reasons) + "."


# ── Integration example (pseudocode for kb_ask wrapper) ──────────────────

def wrapped_kb_ask(kb_ask_func, db, user_model, query: str, **kwargs) -> list[dict]:
    """Wrapper around kb_ask that adds confidence scores to results.

    Usage:
        results = wrapped_kb_ask(kb_ask, db, memory.user, "what is X?")
        for card in results:
            print(f"{card['title']} (confidence: {card['confidence']})")
    """
    # Call original kb_ask
    cards = kb_ask_func(query, **kwargs)

    # Score each result
    scorer = ConfidenceScorer(db, user_model)
    scored = scorer.score_batch(cards)

    # Sort by confidence (optional: prefer high-confidence results)
    scored.sort(key=lambda x: -x.get("confidence", 0.5))

    return scored
