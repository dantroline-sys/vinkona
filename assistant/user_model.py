"""
User model — Vinkona's learned understanding of THIS user.

Instead of generic retrieval + generation, the system builds an explicit profile:
  • domain fluency (where is the user expert, novice, intermediate?)
  • communication style (narrative vs bulleted, academic vs practical, speed tradeoff?)
  • correction history (what topics has the user corrected, how do they think?)
  • interaction outcomes (did they act on research? ignore certain domains?)

This lives in memory.db (alongside memories, documents, etc.) and is passively updated
whenever the user clarifies intent or challenges a finding. It then informs every
retrieval and response generation.

Two update paths:
  1. PASSIVE: system records corrections, clarifications, interactions automatically
  2. EXPLICIT: user or Vinkona can manually set preferences via UI

Retrieval and the LM bridge query this to personalize response ranking and synthesis.
"""

import sqlite3
import time
import typing as tp
import json


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_preferences (
  key TEXT PRIMARY KEY,
  value TEXT,
  last_updated REAL
);

CREATE TABLE IF NOT EXISTS user_domain_fluency (
  domain TEXT PRIMARY KEY,
  fluency_level TEXT CHECK (fluency_level IN ('novice', 'intermediate', 'expert')),
  confidence REAL,
  last_updated REAL,
  interaction_count INTEGER DEFAULT 0,
  correction_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_corrections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp REAL,
  query TEXT,
  vinkona_response TEXT,
  user_correction TEXT,
  domain TEXT,
  correction_type TEXT CHECK (correction_type IN ('clarification', 'wrong_domain', 'factual_error', 'misunderstood_intent', 'domain_expert')),
  source_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_corrections_domain ON user_corrections(domain, timestamp);

CREATE TABLE IF NOT EXISTS user_interactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp REAL,
  query TEXT,
  domain TEXT,
  response_source_count INTEGER,
  user_followed_up BOOLEAN,
  user_acted_on_response BOOLEAN,
  explicit_feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_domain ON user_interactions(domain, timestamp);

CREATE TABLE IF NOT EXISTS user_communication_pattern (
  pattern_name TEXT PRIMARY KEY,
  pattern_value TEXT,
  confidence REAL,
  last_updated REAL,
  evidence_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_response_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp REAL,
  query TEXT,
  domain TEXT,
  response_format TEXT,
  response_length_chars INTEGER,
  user_rated_helpful BOOLEAN,
  user_rating INTEGER
);
CREATE INDEX IF NOT EXISTS idx_response_domain ON user_response_history(domain, timestamp);
"""


class UserModelStore:
    """Query and update the user model in memory.db."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # ── Schema Setup ──────────────────────────────────────────────────────────

    def ensure_schema(self):
        """Idempotent schema creation (already called in __init__)."""
        self._init_schema()

    # ── Preference Management (explicit) ─────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        """Set an explicit user preference (e.g., output_format='narrative')."""
        now = time.time()
        self.db.execute(
            "INSERT OR REPLACE INTO user_preferences(key, value, last_updated) VALUES(?, ?, ?)",
            (key, str(value), now)
        )
        self.db.commit()

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a user preference."""
        row = self.db.execute(
            "SELECT value FROM user_preferences WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def get_all_preferences(self) -> dict[str, str]:
        """Get all user preferences as a dict."""
        rows = self.db.execute("SELECT key, value FROM user_preferences").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ── Domain Fluency (passive + explicit) ────────────────────────────────────

    def record_domain_interaction(
        self,
        domain: str,
        correct: bool = True,
        inferred_level: str | None = None
    ) -> None:
        """Update domain fluency based on an interaction. If correct=False and
        inferred_level is provided, it hints that the user is more expert than we thought."""
        now = time.time()
        row = self.db.execute(
            "SELECT * FROM user_domain_fluency WHERE domain=?", (domain,)
        ).fetchone()

        if row:
            # Update existing: bump interaction count, maybe adjust fluency
            count = row["interaction_count"] + 1
            conf = min(row["confidence"] + 0.05, 1.0) if correct else row["confidence"]
            level = row["fluency_level"]
            if inferred_level and inferred_level != level:
                # User corrected us — hints at expertise
                if inferred_level == "expert":
                    level = "expert"
                    conf = 0.9
                elif inferred_level == "intermediate" and level == "novice":
                    level = "intermediate"
                    conf = 0.7
            self.db.execute(
                "UPDATE user_domain_fluency SET interaction_count=?, "
                "confidence=?, fluency_level=?, last_updated=? WHERE domain=?",
                (count, conf, level, now, domain)
            )
        else:
            # First interaction in this domain
            level = inferred_level or "intermediate"
            conf = 0.5
            self.db.execute(
                "INSERT INTO user_domain_fluency(domain, fluency_level, confidence, "
                "last_updated, interaction_count) VALUES(?, ?, ?, ?, ?)",
                (domain, level, conf, now, 1)
            )
        self.db.commit()

    def set_domain_fluency(self, domain: str, level: str) -> None:
        """Explicitly set domain fluency (novice/intermediate/expert)."""
        if level not in ("novice", "intermediate", "expert"):
            raise ValueError("level must be novice/intermediate/expert")
        now = time.time()
        self.db.execute(
            "INSERT OR REPLACE INTO user_domain_fluency(domain, fluency_level, "
            "confidence, last_updated, interaction_count) VALUES(?, ?, ?, ?, "
            "(SELECT interaction_count FROM user_domain_fluency WHERE domain=?) OR 0)",
            (domain, level, 1.0, now, domain)
        )
        self.db.commit()

    def get_domain_fluency(self, domain: str) -> dict | None:
        """Get the user's fluency in a domain."""
        row = self.db.execute(
            "SELECT * FROM user_domain_fluency WHERE domain=?", (domain,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_domain_fluency(self) -> dict[str, dict]:
        """Get all domain fluency as domain->info dict."""
        rows = self.db.execute("SELECT * FROM user_domain_fluency").fetchall()
        return {row["domain"]: dict(row) for row in rows}

    # ── Corrections (passive) ─────────────────────────────────────────────────

    def record_correction(
        self,
        query: str,
        vinkona_response: str,
        user_correction: str,
        domain: str | None = None,
        correction_type: str = "clarification",
        source_ref: str | None = None
    ) -> None:
        """Record when the user corrects or clarifies Vinkona's response.

        correction_type: 'clarification' (I meant X), 'wrong_domain' (this isn't about X),
                        'factual_error' (that's wrong), 'misunderstood_intent' (I wanted Y),
                        'domain_expert' (actually, I know this better)
        """
        if correction_type not in (
            "clarification",
            "wrong_domain",
            "factual_error",
            "misunderstood_intent",
            "domain_expert",
        ):
            raise ValueError(f"unknown correction_type: {correction_type}")

        now = time.time()
        self.db.execute(
            "INSERT INTO user_corrections(timestamp, query, vinkona_response, "
            "user_correction, domain, correction_type, source_ref) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (now, query[:500], vinkona_response[:500], user_correction[:500], domain, correction_type, source_ref),
        )
        self.db.commit()

        # Update domain fluency if type hints at expertise
        if domain and correction_type == "domain_expert":
            self.record_domain_interaction(domain, correct=True, inferred_level="expert")
        elif domain and correction_type == "factual_error":
            self.record_domain_interaction(domain, correct=False)

    def get_corrections_for_domain(self, domain: str, limit: int = 10) -> list[dict]:
        """Get recent corrections in a domain (for analysis/display)."""
        rows = self.db.execute(
            "SELECT * FROM user_corrections WHERE domain=? ORDER BY timestamp DESC LIMIT ?",
            (domain, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_correction_summary(self) -> dict:
        """Summarize all corrections: count by type and by domain."""
        by_type = self.db.execute(
            "SELECT correction_type, COUNT(*) as count FROM user_corrections "
            "GROUP BY correction_type"
        ).fetchall()
        by_domain = self.db.execute(
            "SELECT domain, COUNT(*) as count FROM user_corrections "
            "WHERE domain IS NOT NULL GROUP BY domain ORDER BY count DESC"
        ).fetchall()
        return {
            "by_type": {row["correction_type"]: row["count"] for row in by_type},
            "by_domain": {row["domain"]: row["count"] for row in by_domain},
        }

    # ── Interactions (passive) ─────────────────────────────────────────────────

    def record_interaction(
        self,
        query: str,
        domain: str | None = None,
        response_source_count: int = 0,
        user_followed_up: bool = False,
        user_acted_on_response: bool = False,
        explicit_feedback: str | None = None,
    ) -> None:
        """Record an interaction outcome: did the user follow up, act on advice, etc.?"""
        now = time.time()
        self.db.execute(
            "INSERT INTO user_interactions(timestamp, query, domain, "
            "response_source_count, user_followed_up, user_acted_on_response, "
            "explicit_feedback) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                query[:500],
                domain,
                response_source_count,
                user_followed_up,
                user_acted_on_response,
                explicit_feedback,
            ),
        )
        self.db.commit()

        # Update domain fluency: note that user acted on advice
        if domain and user_acted_on_response:
            self.record_domain_interaction(domain, correct=True)

    def get_interaction_summary(self, domain: str | None = None) -> dict:
        """Summarize interactions: follow-up rate, action rate, etc."""
        if domain:
            rows = self.db.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN user_followed_up THEN 1 ELSE 0 END) as followups, "
                "SUM(CASE WHEN user_acted_on_response THEN 1 ELSE 0 END) as actions "
                "FROM user_interactions WHERE domain=?",
                (domain,),
            ).fetchone()
        else:
            rows = self.db.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN user_followed_up THEN 1 ELSE 0 END) as followups, "
                "SUM(CASE WHEN user_acted_on_response THEN 1 ELSE 0 END) as actions "
                "FROM user_interactions"
            ).fetchone()

        total = rows["total"] or 0
        followups = rows["followups"] or 0
        actions = rows["actions"] or 0
        return {
            "total_interactions": total,
            "followup_rate": followups / total if total > 0 else 0,
            "action_rate": actions / total if total > 0 else 0,
        }

    # ── Communication Patterns (inferred) ──────────────────────────────────────

    def record_communication_pattern(
        self, pattern_name: str, pattern_value: str, confidence: float = 0.7
    ) -> None:
        """Record an inferred communication pattern (e.g., prefers_narrative=true)."""
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be 0-1")
        now = time.time()
        row = self.db.execute(
            "SELECT * FROM user_communication_pattern WHERE pattern_name=?", (pattern_name,)
        ).fetchone()

        if row:
            # Update: average confidence, bump evidence
            evidence = row["evidence_count"] + 1
            avg_conf = (row["confidence"] * row["evidence_count"] + confidence) / evidence
            self.db.execute(
                "UPDATE user_communication_pattern SET confidence=?, last_updated=?, "
                "evidence_count=? WHERE pattern_name=?",
                (avg_conf, now, evidence, pattern_name),
            )
        else:
            self.db.execute(
                "INSERT INTO user_communication_pattern(pattern_name, pattern_value, "
                "confidence, last_updated, evidence_count) VALUES(?, ?, ?, ?, ?)",
                (pattern_name, pattern_value, confidence, now, 1),
            )
        self.db.commit()

    def get_communication_patterns(self, min_confidence: float = 0.65) -> dict[str, str]:
        """Get confirmed communication patterns (above confidence threshold)."""
        rows = self.db.execute(
            "SELECT pattern_name, pattern_value FROM user_communication_pattern "
            "WHERE confidence >= ? ORDER BY confidence DESC",
            (min_confidence,),
        ).fetchall()
        return {row["pattern_name"]: row["pattern_value"] for row in rows}

    # ── Response History (for analytics) ───────────────────────────────────────

    def record_response(
        self,
        query: str,
        domain: str | None,
        response_format: str = "default",
        response_length_chars: int = 0,
        user_rated_helpful: bool | None = None,
        user_rating: int | None = None,
    ) -> None:
        """Record a response that Vinkona gave (for tracking format effectiveness)."""
        now = time.time()
        self.db.execute(
            "INSERT INTO user_response_history(timestamp, query, domain, "
            "response_format, response_length_chars, user_rated_helpful, user_rating) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (now, query[:500], domain, response_format, response_length_chars, user_rated_helpful, user_rating),
        )
        self.db.commit()

    def get_response_effectiveness(self, response_format: str | None = None) -> dict:
        """Summarize how helpful responses have been (by format if specified)."""
        if response_format:
            rows = self.db.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN user_rated_helpful=1 THEN 1 ELSE 0 END) as helpful, "
                "AVG(user_rating) as avg_rating "
                "FROM user_response_history WHERE response_format=?",
                (response_format,),
            ).fetchone()
        else:
            rows = self.db.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN user_rated_helpful=1 THEN 1 ELSE 0 END) as helpful, "
                "AVG(user_rating) as avg_rating "
                "FROM user_response_history"
            ).fetchone()

        total = rows["total"] or 0
        helpful = rows["helpful"] or 0
        avg_rating = rows["avg_rating"] or 0
        return {
            "total_responses": total,
            "helpful_count": helpful,
            "helpfulness_rate": helpful / total if total > 0 else 0,
            "avg_rating": avg_rating,
        }

    # ── Analysis / Summary ─────────────────────────────────────────────────────

    def get_user_summary(self) -> dict:
        """Generate a comprehensive summary of the user profile for the LM to use."""
        prefs = self.get_all_preferences()
        domains = self.get_all_domain_fluency()
        patterns = self.get_communication_patterns()
        corrections = self.get_correction_summary()
        interactions = self.get_interaction_summary()

        return {
            "preferences": prefs,
            "domain_fluency": domains,
            "communication_patterns": patterns,
            "correction_summary": corrections,
            "interaction_summary": interactions,
        }

    def get_user_context_for_lm(self) -> str:
        """Generate a natural-language summary of the user for inclusion in LM prompts.

        This is what the LM bridge passes to the big LM for personalized synthesis.
        Keep it concise — just the high-confidence, actionable parts.
        """
        summary = self.get_user_summary()
        prefs = summary.get("preferences", {})
        domains = summary.get("domain_fluency", {})
        patterns = summary.get("communication_patterns", {})

        parts = []

        # Expertise summary
        experts = [d for d, info in domains.items() if info.get("fluency_level") == "expert"]
        if experts:
            parts.append(f"The user is expert in: {', '.join(experts)}")

        intermediates = [d for d, info in domains.items() if info.get("fluency_level") == "intermediate"]
        if intermediates:
            parts.append(f"The user is intermediate in: {', '.join(intermediates)}")

        # Communication preferences
        if patterns:
            for pname, pval in sorted(patterns.items()):
                parts.append(f"Communication: {pname} = {pval}")

        # Interaction patterns
        interactions = summary.get("interaction_summary", {})
        if interactions.get("action_rate", 0) > 0.5:
            parts.append(
                f"The user typically acts on advice (action rate: {interactions['action_rate']:.0%})"
            )

        if not parts:
            return ""

        return (
            "## User Profile\n\n"
            + "\n".join(f"- {p}" for p in parts)
            + "\n\nUse this to personalize responses: tailor depth to expertise, "
            "respect communication preferences."
        )
