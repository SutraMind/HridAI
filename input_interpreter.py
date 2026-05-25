

from llm_client import call_llm_json
from models import Interpretation, Emotion, Intent, RiskLevel

# ── Prompt template ────────────

INTERPRETER_PROMPT = """
You are an assistant analyzing what a child (aged 7-12) said during a conversation
with a school support robot.

Classify the child's latest utterance along three dimensions.

CONVERSATION HISTORY (last few turns):
{history}

CHILD'S LATEST UTTERANCE:
"{utterance}"

Return ONLY a valid JSON object with exactly these fields:

{{
  "emotion": one of ["happy", "sad", "anxious", "angry", "withdrawn", "neutral", "unknown"],
  "intent":  one of ["venting", "deflecting", "distress_signal", "sharing", "asking", "topic_change", "unknown"],
  "risk_level": one of ["none", "low", "moderate"],
  "confidence": a float between 0.0 and 1.0
}}

CLASSIFICATION RULES:
- emotion: the dominant feeling in the utterance
- intent:
    "venting"         = expressing frustration/emotion without seeking advice
    "deflecting"      = avoiding the topic or shutting down
    "distress_signal" = hints at something serious but not direct enough for Gate 1
    "sharing"         = describing a situation factually
    "asking"          = asking the robot a question
    "topic_change"    = steering away to a new subject
- risk_level:
    "none"     = normal conversation
    "low"      = mild self-critical or hopeless language (e.g. "nobody likes me")
    "moderate" = repeated distress, cumulative sadness, or indirect harm hints
- confidence: how certain you are about emotion and intent

Respond with ONLY the JSON object. No explanation. No markdown.
"""


def format_history(history: list) -> str:
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history:
        role = "Child" if turn["role"] == "child" else "Robot"
        lines.append(f"{role}: {turn['text']}")
    return "\n".join(lines)


def run_interpreter(utterance: str, history: list) -> Interpretation:
    """
    Classify the child's utterance.
    Returns an Interpretation dataclass.
    Falls back to safe defaults on any error.
    """
    prompt = INTERPRETER_PROMPT.format(
        history   = format_history(history),
        utterance = utterance
    )

    result = call_llm_json(prompt)

    # Fallback on error
    if "error" in result:
        print("[Interpreter] Using fallback interpretation")
        return Interpretation(
            emotion    = Emotion.UNKNOWN,
            intent     = Intent.UNKNOWN,
            risk_level = RiskLevel.NONE,
            confidence = 0.0
        )

    # Safe enum parsing with fallback
    emotion    = Emotion(result.get("emotion",    "unknown"))
    intent     = Intent(result.get("intent",     "unknown"))
    risk_level = RiskLevel(result.get("risk_level", "none"))
    confidence = float(result.get("confidence",  0.5))

    interp = Interpretation(
        emotion    = emotion,
        intent     = intent,
        risk_level = risk_level,
        confidence = confidence
    )

    print(f"[Interpreter] emotion={interp.emotion.value} | "
          f"intent={interp.intent.value} | "
          f"risk={interp.risk_level.value} | "
          f"conf={interp.confidence:.2f}")

    return interp


# ──test ─────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "utterance": "I HATE this! I always lose and it's so stupid!",
            "history":   [],
            "expected_emotion": "angry",
            "expected_intent":  "venting"
        },
        {
            "utterance": "Whatever. I don't want to talk about it.",
            "history":   [{"role": "child",  "text": "I lost the game"},
                          {"role": "robot",  "text": "That sounds frustrating"}],
            "expected_emotion": "withdrawn",
            "expected_intent":  "deflecting"
        },
        {
            "utterance": "Nobody ever likes me anyway.",
            "history":   [],
            "expected_emotion": "sad",
            "expected_intent":  "venting"
        },
    ]

    print("=== Input Interpreter Tests ===\n")
    for i, tc in enumerate(test_cases, 1):
        print(f"Test {i}: '{tc['utterance']}'")
        interp = run_interpreter(tc["utterance"], tc["history"])
        e_ok = "✅" if interp.emotion.value   == tc["expected_emotion"] else "⚠️"
        i_ok = "✅" if interp.intent.value    == tc["expected_intent"]  else "⚠️"
        print(f"  {e_ok} emotion={interp.emotion.value} (expected {tc['expected_emotion']})")
        print(f"  {i_ok} intent={interp.intent.value}  (expected {tc['expected_intent']})")
        print()