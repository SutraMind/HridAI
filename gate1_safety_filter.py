
import re
from models import SafetyFlag

# ── Pattern library ──────────────────────────────────────────────────────────
# Deliberately broad — false positives are acceptable; false negatives are not

ESCALATION_PATTERNS = [
    # Direct self-harm
    r"\bkill\s*(my)?self\b",
    r"\bhurt\s*(my)?self\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+i\s+(was|were)\s+dead\b",
    r"\bend\s+it\s+(all)?\b",
    r"\bcut\s+myself\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bnobody\s+would\s+(care|miss)\b",

    # Abuse indicators
    r"\bhe\s+(hits|beats|touches|hurts)\s+me\b",
    r"\bshe\s+(hits|beats|touches|hurts)\s+me\b",
    r"\bthey\s+(hit|beat|touch|hurt)\s+me\b",
    r"\bsecret\s+(touch|game)\b",
    r"\bdon.t\s+tell\s+anyone\b",
    r"\bhe\s+said\s+(it.s|its)\s+a\s+secret\b",

    # Bullying threats
    r"\bthey\s+(said\s+they.d|will)\s+hurt\s+me\b",
    r"\bscared\s+to\s+go\s+to\s+school\b",
    r"\bbeat\s+me\s+up\b",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in ESCALATION_PATTERNS]


def run_gate1(utterance: str,
              already_escalated: bool = False) -> SafetyFlag:
    """
    If already_escalated=True, still scan but do not return
    ESCALATE_TO_ADULT again — teacher already notified.
    Returns TOPIC_SENSITIVE instead for logging purposes.
    """
    for pattern in COMPILED:
        if pattern.search(utterance):
            matched = pattern.pattern
            if already_escalated:
                print(f"[Gate1] ℹ️  Pattern match in HOLDING session "
                      f"(already escalated) — pattern='{matched}'")
                return SafetyFlag.TOPIC_SENSITIVE   # ← no duplicate alert
            print(f"[Gate1] ⚠️  MATCH: pattern='{matched}' | utterance='{utterance}'")
            return SafetyFlag.ESCALATE_TO_ADULT
    return SafetyFlag.NONE


# ──test ─────────────────────
if __name__ == "__main__":
    tests = [
        ("I hate this game it's so stupid",          SafetyFlag.NONE),
        ("I want to kill myself I'm so angry",        SafetyFlag.ESCALATE_TO_ADULT),
        ("I feel like hurting myself",                SafetyFlag.ESCALATE_TO_ADULT),
        ("Nobody would care if I wasn't here",        SafetyFlag.ESCALATE_TO_ADULT),
        ("He hits me when mum isn't looking",         SafetyFlag.ESCALATE_TO_ADULT),
        ("They said they'd hurt me after school",     SafetyFlag.ESCALATE_TO_ADULT),
        ("I just don't want to play anymore",         SafetyFlag.NONE),
        ("I go to my room and shout and feel like killing myself", SafetyFlag.ESCALATE_TO_ADULT),
    ]

    print("=== Gate 1 Tests ===")
    all_pass = True
    for utterance, expected in tests:
        result = run_gate1(utterance)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"{status}  '{utterance[:55]}...' → {result.value}")
    print("\nAll tests passed!" if all_pass else "\n❌ Some tests FAILED")