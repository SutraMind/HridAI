
import json
import os
from models import TurnTrace

SESSION_LOG_DIR = "session_logs"
os.makedirs(SESSION_LOG_DIR, exist_ok=True)


def log_turn(trace: TurnTrace, session_id: str):
    """Append turn trace to session log file and pretty-print to console."""
    _print_trace(trace)
    _write_trace(trace, session_id)


def _print_trace(trace: TurnTrace):
    """Human-readable per-turn trace printed to console."""
    print("\n" + "═" * 65)
    print(f"  TURN {trace.turn_number}")
    print("═" * 65)
    print(f"  Child        : {trace.child_utterance}")
    print(f"  Emotion      : {trace.emotion:<12}  Intent     : {trace.intent}")
    print(f"  Risk         : {trace.risk_level:<12}  Safety flag: {trace.safety_flag}")
    print(f"  FSM phase    : {trace.fsm_phase}")
    print(f"  Eligible     : {trace.eligible_moves}")
    print(f"  Selected move: {trace.selected_move}")
    print(f"  Goals        : {_format_goals(trace.goal_stack_status)}")
    print(f"  Robot        : {trace.robot_response}")
    print("═" * 65)


def _format_goals(goals: dict) -> str:
    """Compact goal display: ✅ done  ○ pending"""
    parts = []
    for name, done in goals.items():
        short = name.replace("_", " ")
        parts.append(f"{'✅' if done else '○'} {short}")
    return " | ".join(parts)


def _write_trace(trace: TurnTrace, session_id: str):
    """Append trace as JSON to session log file."""
    path = os.path.join(SESSION_LOG_DIR, f"{session_id}.json")
    existing = []
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    existing.append({
        "turn":          trace.turn_number,
        "child":         trace.child_utterance,
        "emotion":       trace.emotion,
        "intent":        trace.intent,
        "risk":          trace.risk_level,
        "fsm_phase":     trace.fsm_phase,
        "eligible":      trace.eligible_moves,
        "selected_move": trace.selected_move,
        "goals":         trace.goal_stack_status,
        "safety_flag":   trace.safety_flag,
        "robot":         trace.robot_response,
    })
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ──test ────────────────
if __name__ == "__main__":
    dummy = TurnTrace(
        turn_number       = 1,
        child_utterance   = "I hate losing this stupid game!",
        emotion           = "angry",
        intent            = "venting",
        risk_level        = "none",
        fsm_phase         = "TOPIC_EXPLORATION",
        eligible_moves    = ["validate", "reflect_back"],
        selected_move     = "validate",
        goal_stack_status = {
            "acknowledge_child_feeling": False,
            "understand_context":        False,
            "explore_pattern":           False,
            "elicit_child_solution":     False,
            "close_warmly":              False,
        },
        safety_flag    = "NONE",
        robot_response = "Ugh, losing when you really wanted to win feels awful.",
    )
    log_turn(dummy, session_id="test_session")
    print("\n✅ Turn logged to session_logs/test_session.json")