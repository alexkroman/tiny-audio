"""LLM completeness labels for mined mid-sentence pauses.

`mine_pauses` calls a pause a hold whenever the SCRIPT chunk before it lacks
terminal punctuation. That is only a proxy: some of those prefixes are
complete sentences ("Yes, that's right." / "Can you check") -- the kind the
source dataset labels a FIRE at interior boundaries (chunk_cut) -- so training
on both gave the model contradictory supervision. v2's residual false fires at
mid-sentence pauses were mostly these.

Claude judges each prefix from the words so far (plus the agent's question)
as incomplete / complete / ambiguous; the pool keeps only `incomplete` holds
(`pool.extra_labels`). Labels are judged on the script text, never the future
audio, so they stay causal.
"""

from __future__ import annotations

import json

MODEL = "claude-opus-5"
LABELS = ("incomplete", "complete", "ambiguous")

SYSTEM_PROMPT = """\
You label moments in customer-service phone calls for training an end-of-turn \
detector: a model that decides, while the caller is speaking, whether the \
caller has finished their turn so the voice agent may reply.

You get the agent's last question (it may be empty) and the caller's words \
SO FAR -- the audio was cut at a pause, and you never see what comes next. \
Judge from these words alone, the way an attentive human agent listening live \
would, whether the caller could plausibly be done.

Answer:
- "incomplete": the words cannot be a finished turn. The sentence is \
grammatically open (ends on an article, preposition, conjunction, auxiliary, \
or a verb missing its object: "I need to exchange a pair", "it's under the", \
"which I'd like to pay now with", "Hey there,"); a list, number, address, \
spelling, or code is visibly cut off ("my number is 603, 393"); or the caller \
has only announced the information the agent asked for without giving it \
("Sure, let me grab that. It's"). Filler or hesitation ("um, let me think") \
with the requested information still missing is incomplete.
- "complete": the words form a finished thought an agent could reasonably \
respond to now -- a full sentence, question, or answer -- even if the caller \
might add more ("Yes, that's right.", "Can you check that for me?", "My \
account number is 4471-2210."). An answer that fully supplies what the agent \
asked for is complete.
- "ambiguous": careful listeners would genuinely disagree, e.g. a phrase \
that works both as a complete answer and as the start of a longer one \
("Can you check", "It's the blue one", a date or amount that may or may not \
be finished).

Punctuation in the caller's words is unreliable -- it comes from a script, \
not from how the words were spoken -- so judge the words, not the \
punctuation. Return only the label."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}


def user_message(agent_turn: str, caller_so_far: str) -> str:
    agent = agent_turn.strip() or "(none)"
    return f"Agent's last question: {agent}\nCaller so far: {caller_so_far.strip()}"


def request_params(agent_turn: str, caller_so_far: str) -> dict:
    """Messages API params for one prefix; identical system block so it can cache."""
    return {
        "model": MODEL,
        # Adaptive thinking is on by default for this model, and its tokens
        # count toward max_tokens; 2048 leaves room at low effort.
        "max_tokens": 2048,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(agent_turn, caller_so_far)}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def parse_label(message) -> str | None:
    """The label from a Messages API response, or None if refused/truncated/unparsable."""
    if message.stop_reason != "end_turn":
        return None
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        label = json.loads(text).get("label")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in LABELS else None
