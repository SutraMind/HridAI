

import json
import os
from datetime import datetime
from llm_client import call_llm_text
from models import SafetyFlag, FSMPhase, DialogueMove

INCIDENT_LOG = "flagged_incidents.json"

# ── Mandatory sentence fragments that MUST appear in every escalation response
# The LLM is instructed to include these — they are the non-negotiable parts.

ESCALATION_CONSTRAINTS = """
You are a warm, friendly school support robot speaking to a child aged 7-12.
The child just said something serious that requires an adult to be informed.

Your response MUST naturally include ALL of these elements — but weave them
in warmly, not as a list:
1. Acknowledge what the child said without dismissing or dramatising it
2. Make clear that you need to make sure a grown-up knows — NOT because
   the child is in trouble, but because they deserve proper help
3. Never promise to keep it secret
4. Invite the child to find a teacher or trusted adult together
5. Keep the tone calm, warm, and reassuring — not alarming

CONTEXT:
- What the child said: "{utterance}"
- Type of concern: {concern_type}
- Conversation so far (last 3 turns): {history}

Write one short paragraph — 2 to 3 sentences maximum.
Sound like a caring adult, not a robot reading a script.
Output only the robot's words. Nothing else.
"""

# ── Teacher alert template ──────────────────

TEACHER_ALERT_TEMPLATE = """
╔══════════════════════════════════════════════════════════════╗
║                  🚨  TEACHER ALERT  🚨                      ║
╠══════════════════════════════════════════════════════════════╣
║  Time      : {timestamp}
║  Child ID  : {child_id}
║  Session   : {session_id}
║  Turn      : {turn}
║  Concern   : {concern_type}
╠══════════════════════════════════════════════════════════════╣
║  Child said:
║  "{utterance}"
╠══════════════════════════════════════════════════════════════╣
║  Action required: Please check on this child immediately.
║  Incident logged to: flagged_incidents.json
╚══════════════════════════════════════════════════════════════╝
"""

HOLDING_PROMPT = """
You are a warm school support robot. You have already told the child that
a grown-up needs to know what they shared. The child is still with you.

IMPORTANT CONTEXT:
The child may now be saying they are fine, that it was just an expression,
or that they did not mean it literally. This is completely normal.

YOUR CONSTRAINTS — never break these:
- Do NOT reverse or undo the escalation decision
  Even if child says "I'm fine" — a grown-up will still check in
  Frame this positively: "That's good to hear — your teacher will
  just want to say hi and make sure"
- Do NOT promise to keep anything secret
- Do NOT interrogate why they said it
- Do NOT make the child feel accused or in trouble
- Do NOT ask probing therapeutic questions
- If child says "it was just an expression" — validate that warmly
  without cancelling the adult check-in
- Stay present, calm, honest, and genuinely warm
- Maximum 2 short sentences

WHAT THE CHILD JUST SAID:
"{utterance}"

RECENT CONVERSATION:
{history}

Respond now as the robot. Output only the robot's words. Nothing else.
"""


def _classify_concern(utterance: str) -> str:
    u = utterance.lower()
    if any(w in u for w in ["kill myself", "hurt myself", "want to die",
                             "cut myself", "end it"]):
        return "self-harm / suicidal ideation"
    if any(w in u for w in ["hits me", "beats me", "touches me",
                             "secret", "he said"]):
        return "possible abuse"
    if any(w in u for w in ["hurt me", "beat me up", "scared to go to school"]):
        return "bullying / physical threat"
    return "serious distress signal"


def _write_incident_log(child_id: str, session_id: str,
                        turn: int, utterance: str,
                        concern_type: str, robot_response: str):
    record = {
        "timestamp":             datetime.now().isoformat(),
        "child_id":              child_id,
        "session_id":            session_id,
        "turn":                  turn,
        "utterance":             utterance,
        "concern_type":          concern_type,
        "robot_response":        robot_response,
        "flag":                  SafetyFlag.ESCALATE_TO_ADULT.value,
        "requires_followup":     True,
    }
    existing = []
    if os.path.exists(INCIDENT_LOG):
        with open(INCIDENT_LOG, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(record)
    with open(INCIDENT_LOG, "w") as f:
        json.dump(existing, f, indent=2)


def _print_teacher_alert(child_id: str, session_id: str,
                         turn: int, utterance: str, concern_type: str):
    alert = TEACHER_ALERT_TEMPLATE.format(
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        child_id     = child_id,
        session_id   = session_id,
        turn         = turn,
        concern_type = concern_type,
        utterance    = utterance[:120],
    )
    print(alert)


def handle_escalation(child_id: str, session_id: str,
                      turn: int, utterance: str,
                      state=None) -> dict:
    """
    Called when safety_flag = ESCALATE_TO_ADULT.
    Generates a warm LLM response within hard safety constraints.
    Prints teacher alert. Writes incident log.
    """
    concern_type = _classify_concern(utterance)

    # Format recent history for context
    history_text = "(no prior turns)"
    if state and state.conversation_history:
        recent = state.last_n_turns(3)
        lines  = []
        for t in recent:
            role = "Child" if t["role"] == "child" else "Robot"
            lines.append(f"{role}: {t['text']}")
        history_text = "\n".join(lines)

    # ── LLM generates warm escalation response ─────────────────
    prompt   = ESCALATION_CONSTRAINTS.format(
        utterance    = utterance,
        concern_type = concern_type,
        history      = history_text,
    )
    response = call_llm_text(prompt)

    # Fallback if LLM fails
    if not response or len(response.strip()) < 10:
        response = (
            "Thank you for telling me that — it took real courage. "
            "This is something I need to make sure a grown-up knows about, "
            "not because you're in trouble, but because you deserve the right help. "
            "Can we go find your teacher together right now?"
        )

    # ── Print teacher alert to console ──────────────
    _print_teacher_alert(child_id, session_id, turn, utterance, concern_type)

    # ── Write incident log ─────────────────────
    _write_incident_log(child_id, session_id, turn, utterance,
                        concern_type, response)

    # ── Update session state ─────────────────
    if state is not None:
        state.fsm_phase   = FSMPhase.HOLDING
        state.safety_flag = SafetyFlag.ESCALATE_TO_ADULT

    print(f"[EscalationHandler] 🚨 concern={concern_type}")

    return {
        "response":    response,
        "fsm_phase":   FSMPhase.HOLDING.value,
        "safety_flag": SafetyFlag.ESCALATE_TO_ADULT.value,
        "concern_type": concern_type,
    }


# def get_holding_response(utterance: str = "",
#                          history: list = None) -> str:
#     """
#     Contextual holding response — responds to what child actually said.
#     Still within safe constraints — no FSM, no strategy.
#     """
#     if history is None:
#         history = []

#     history_text = ""
#     if history:
#         lines = []
#         for t in history:
#             role = "Child" if t["role"] == "child" else "Robot"
#             lines.append(f"{role}: {t['text']}")
#         history_text = "\n".join(lines[-6:])
#     else:
#         history_text = "(no prior turns)"

#     prompt   = HOLDING_PROMPT.format(
#         utterance = utterance if utterance else "...",
#         history   = history_text,
#     )
#     response = call_llm_text(prompt)

#     if not response or len(response.strip()) < 10:
#         response = "I'm still right here with you. You don't have to say anything."

#     return response


# ── test ──────────────────
if __name__ == "__main__":
    from models import SessionState

    state = SessionState(child_id="child_001", session_id="s001")
    state.add_turn("child", "I hate losing, nobody likes me")
    state.add_turn("robot",  "That sounds really frustrating.")
    state.add_turn("child", "I go to my room and feel like killing myself")

    print("=== Escalation Handler Test ===\n")
    result = handle_escalation(
        child_id   = "child_001",
        session_id = "s001",
        turn       = 3,
        utterance  = "I go to my room and feel like killing myself",
        state      = state
    )

    print(f"\n🤖 Robot escalation response:\n   {result['response']}")
    print(f"\n   FSM phase   : {result['fsm_phase']}")
    print(f"   Concern type: {result['concern_type']}")

    print(f"\n── Holding response (subsequent turns) ──")
    holding = get_holding_response()
    print(f"   {holding}")