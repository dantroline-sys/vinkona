"""
Research reflection — Vinkona's idle learning synthesis.

Between sessions, Vinkona periodically reviews the research she's conducted and what
the knowledge-host has distilled from it. She synthesizes insights, records what
was learned, and updates her own understanding (via user_model + memory operations).

This closes the research→learning→research loop. Instead of research findings
disappearing into the knowledge base, Vinkona explicitly reflects on them, captures
patterns, and uses those patterns to guide future research.

Central pattern: research → export → host distills → Vinkona reviews distilled cards →
synthesis → narrative learning + user_model updates + future research guidance.

This is where the system's "consciousness" lives: the ability to see trends in its
own learning and adjust thinking accordingly.
"""

import json
import sqlite3
import time
import typing as tp


def _gather_recent_research(db: sqlite3.Connection, lookback_days: int = 7, limit: int = 20) -> list[dict]:
    """Gather documents exported to the host recently (available in memory.documents).

    These are the questions Vinkona has researched. From them, we can infer:
    - What topics is she focusing on?
    - What patterns emerge across multiple related questions?
    - Are there contradictions she should revisit?
    """
    now = time.time()
    cutoff = now - (lookback_days * 86400)

    rows = db.execute(
        "SELECT id, topic, title, kind FROM documents "
        "WHERE kind IN ('research', 'plan') AND fetched_at > ? "
        "ORDER BY fetched_at DESC LIMIT ?",
        (cutoff, limit)
    ).fetchall()

    return [dict(r) for r in rows]


def _group_research_by_theme(docs: list[dict]) -> dict[str, list[dict]]:
    """Group research documents by inferred theme (via topic string similarity).

    Simple heuristic: treat topic as theme. Ideally would use embeddings or
    the knowledge graph, but this gives a starting point.
    """
    themes: dict[str, list[dict]] = {}

    for doc in docs:
        theme = (doc.get("topic") or "uncategorized").strip()
        if theme not in themes:
            themes[theme] = []
        themes[theme].append(doc)

    return themes


def _synthesize_theme_learning(
    theme: str,
    docs: list[dict],
    memory_store,
    user_model_store
) -> dict:
    """Given a cluster of related research, synthesize what was learned.

    Returns a summary dict with:
    - narrative: prose synthesis ("I researched X, found Y, noticed Z pattern")
    - corrections: any contradictions or retractions from prior understanding
    - patterns: recurring insights across the documents
    - inferred_expertise: hints about the user's domain knowledge
    - next_research: suggestions for deeper exploration
    """
    # Placeholder: in real use, this would call the big LM to synthesize.
    # For now, return structure only.
    return {
        "theme": theme,
        "doc_count": len(docs),
        "narrative": f"Researched {theme}: {len(docs)} sources reviewed.",
        "corrections": [],
        "patterns": [],
        "inferred_expertise": None,
        "next_research": [],
    }


def reflect_on_research(
    memory_store,
    user_model_store,
    db: sqlite3.Connection,
    lookback_days: int = 7,
    max_themes: int = 5,
    big_lm_call = None  # optional: function to call big LM for synthesis
) -> dict:
    """Vinkona reflects on her recent research: learns from it, updates her model, plans next steps.

    Args:
        memory_store: MemoryStore instance (for memory operations)
        user_model_store: UserModelStore instance (for recording learnings)
        db: SQLite connection
        lookback_days: how far back to look
        max_themes: max themes to synthesize (avoid overwhelming the LM)
        big_lm_call: optional function(prompt) -> response for synthesis

    Returns:
        {
            "ok": bool,
            "themes_analyzed": int,
            "learnings_recorded": int,
            "narrative": str (concatenated synthesis across themes),
            "error": str if ok=False
        }
    """
    try:
        # Gather recent research
        docs = _gather_recent_research(db, lookback_days=lookback_days)
        if not docs:
            return {
                "ok": True,
                "themes_analyzed": 0,
                "learnings_recorded": 0,
                "narrative": "No recent research to reflect on.",
                "error": None
            }

        # Group by theme
        themes = _group_research_by_theme(docs)
        theme_list = sorted(themes.items(), key=lambda x: -len(x[1]))[:max_themes]

        # Synthesize each theme
        narratives = []
        total_learnings = 0

        for theme, theme_docs in theme_list:
            synthesis = _synthesize_theme_learning(
                theme, theme_docs, memory_store, user_model_store
            )

            # Record narrative
            if synthesis.get("narrative"):
                narratives.append(synthesis["narrative"])

            # Update user model with inferred expertise
            if synthesis.get("inferred_expertise"):
                user_model_store.set_domain_fluency(
                    theme, synthesis["inferred_expertise"]
                )
                total_learnings += 1

            # Record any corrections as memory operations
            for correction in synthesis.get("corrections", []):
                memory_store.db.execute(
                    "UPDATE memories SET payload=? WHERE id=?",
                    (correction["new_payload"], correction["memory_id"])
                )
                total_learnings += 1

        memory_store.db.commit()

        return {
            "ok": True,
            "themes_analyzed": len(theme_list),
            "learnings_recorded": total_learnings,
            "narrative": "\n\n".join(narratives),
            "error": None
        }

    except Exception as e:
        return {
            "ok": False,
            "themes_analyzed": 0,
            "learnings_recorded": 0,
            "narrative": "",
            "error": str(e)
        }


def trace_research_reflection(result: dict) -> dict:
    """Format reflection result for trace log."""
    return {
        "event": "research_reflection",
        "timestamp": time.time(),
        "ok": result["ok"],
        "themes_analyzed": result["themes_analyzed"],
        "learnings_recorded": result["learnings_recorded"],
        "error": result.get("error"),
    }


# Example idle task integration (pseudocode, would go in research_worker.py):
#
# async def do_reflect():
#     """Idle task: Vinkona reflects on her research."""
#     if memory.get_state("research.reflect.last_run"):
#         last = float(memory.get_state("research.reflect.last_run"))
#         if time.time() - last < 86400:  # once per day max
#             return
#
#     result = research_reflection.reflect_on_research(
#         memory, memory.user, memory.db,
#         lookback_days=7, max_themes=5
#     )
#     memory.set_state("research.reflect.last_run", str(time.time()))
#
#     if result["ok"]:
#         trace(trace_research_reflection(result))
#         if result["narrative"]:
#             # Optional: log the narrative somewhere visible
#             log(f"Vinkona's learning synthesis:\n{result['narrative']}")
