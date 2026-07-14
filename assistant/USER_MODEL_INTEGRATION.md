# User Model Integration Guide

The user model is now built and integrated into memory.db. This guide shows how to wire it into the cascade for passive tracking.

## Quick Start

### 1. Access the user model in code

```python
from memory import MemoryStore
import json

cfg = json.load(open("config.json"))
memory = MemoryStore(cfg)

# Query the user profile
fluency = memory.user.get_domain_fluency("pain medicine")
prefs = memory.user.get_preference("output_format", default="default")
context = memory.user.get_user_context_for_lm()  # for LM prompts
```

### 2. Record a correction when user clarifies

In cascade_server.py, when a user says "actually, I meant X" or "that's wrong":

```python
from cascade_server import memory  # assume memory is a module-level variable

# In the correction handler:
memory.user.record_correction(
    query="your question here",
    vinkona_response="what Vinkona said",
    user_correction="what user actually meant / correct info",
    domain="inferred_domain",  # or None; auto-infer if possible
    correction_type="clarification",  # or "factual_error", "misunderstood_intent", "domain_expert"
    source_ref="optional source URL or reference"
)
```

### 3. Record an interaction when user acts on advice

When you detect follow-up or user acting on advice:

```python
memory.user.record_interaction(
    query="original query",
    domain="inferred_domain",
    response_source_count=number_of_sources_returned,
    user_followed_up=True,  # did they ask more about this?
    user_acted_on_response=True,  # did they use the advice?
    explicit_feedback="optional direct rating"
)
```

### 4. Use user context in LM synthesis

In llm_bridge.py (or wherever you call the big LM for synthesis):

```python
from memory import MemoryStore

# Before generating a response:
user_context = memory.user.get_user_context_for_lm()

# Include it in the system prompt:
if user_context:
    system_prompt = f"<<USER CONTEXT>>\n{user_context}\n<</USER CONTEXT>>\n\n" + base_system_prompt

# Now call the LM with personalized context
response = call_llm(user_prompt, system_prompt)
```

## How It Works

### Passive Updates

The user model updates itself automatically:

1. **Domain fluency bumps up on corrections** — If user says "I'm actually expert in this" (correction_type="domain_expert"), their fluency in that domain jumps to "expert" with high confidence.

2. **Fluency tracks action rate** — If user acts on research advice (record_interaction(..., user_acted_on_response=True)), the system notes that advice in that domain is effective.

3. **Patterns inferred from interaction style** — System can call record_communication_pattern() to note things like "prefers_narrative=true" or "wants_sources_for_medical=true".

### Retrieval Ranking

Once the user model is populated, kb_ask can use it to rank results:

```python
# Example (not yet wired):
user_fluency = memory.user.get_domain_fluency(kb_result["domain"])
if user_fluency and user_fluency["fluency_level"] == "expert":
    # Deprioritize basic results, show advanced/novel findings
    result["rank_boost"] = -1
elif user_fluency and user_fluency["fluency_level"] == "novice":
    # Prioritize foundational sources
    result["rank_boost"] = 1
```

### LM Synthesis

When the big LM generates a response, the user context tells it:

- **What the user already knows** → Don't waste tokens on basics, focus on novel angles
- **Communication style** → If user prefers bullets over narrative, tailor accordingly
- **Interaction history** → If they act on your advice, your confidence is warranted
- **Correction history** → If they often correct you on a topic, be more cautious there

## Example: Wiring a correction into cascade_server

In cascade_server.py, when handling a user message that contradicts Vinkona's last response:

```python
async def handle_user_correction():
    """User has clarified or corrected Vinkona's last response."""
    user_msg = request.json.get("text")
    vinkona_last_response = memory.entries[...]["payload"]  # retrieve context
    
    # Infer domain from the conversation context
    domain = infer_domain_from_query(conversation_history)
    
    # Record the correction
    memory.user.record_correction(
        query=last_user_query,
        vinkona_response=vinkona_last_response,
        user_correction=user_msg,
        domain=domain,
        correction_type="clarification",
    )
    
    # Future responses in this domain will benefit from this learning
    return {"status": "correction recorded"}
```

## Infer Domain (stub)

You'll need a domain inference function. Simple version:

```python
def infer_domain_from_query(query: str) -> str | None:
    """Guess the domain from the query text or kb_ask facets."""
    query_lower = query.lower()
    
    # Simple keyword matching (replace with kb_ask facets in production)
    keywords = {
        "medicine": ["disease", "treatment", "drug", "symptom", "diagnosis"],
        "nutrition": ["food", "diet", "eat", "calorie", "vitamin"],
        "psychology": ["anxiety", "depression", "trauma", "therapy", "mental"],
    }
    
    for domain, words in keywords.items():
        if any(w in query_lower for w in words):
            return domain
    
    return None
```

Better: use kb_ask results to extract the facet:

```python
async def infer_domain_from_kb_results(kb_results: list[dict]) -> str | None:
    """Extract domain from kb_ask facet if available."""
    if kb_results and kb_results[0].get("facet_domain"):
        return kb_results[0]["facet_domain"]
    return None
```

## Next Steps

1. ~~**Wire record_correction()**~~ — DONE: idle reflection (memory.idle_reflect) now spots
   corrections in each review window and banks them (source_ref="idle_reflect"); the
   corrections idle task (memory.review_corrections) then turns fresh ones into
   generalized research questions → case/procedure cue cards via the research pipeline.
2. **Wire record_interaction() into the follow-up/action handler** — detect when user acts on advice
3. ~~**Pass get_user_context_for_lm() to big LM prompts**~~ — DONE: `user_profile_hook` on the
   LM bridge injects it into the big LM's briefing and deliberation (cached per session);
   the fast voice prompt deliberately stays lean.
4. **Add UI view in config_ui.html** — show the user their inferred profile (Settings tab)
5. **Extend infer_domain()** — use kb_ask facets instead of keywords
6. **Retrieval ranking** — kb_ask uses get_all_domain_fluency() to adjust result order
