

from models import (
    SessionState, Interpretation, FSMPhase,
    EngagementLevel, ResponseQuality, DialogueMove, Goal
)

# ── Default goal stack for every session ──────────

DEFAULT_GOALS = [
    "acknowledge_child_feeling",
    "understand_context",
    "explore_pattern",
    "elicit_child_solution",
    "close_warmly",
]

# ── Risk decay constants ─────────────────
_RISK_SCORE_MAP = {"none": 0, "low": 1, "moderate": 3, "high": 5}

# deflecting and distress_signal do NOT advance the streak
_SAFE_INTENTS = {"sharing", "asking", "topic_change", "venting"}

_SAFE_STREAK_REQUIRED = 3    # consecutive safe turns needed to earn 1 decay
_DECAY_AMOUNT         = 1    # points removed per completed streak
_MAX_DECAY_FRACTION   = 0.5  # score cannot drop below 50% of session peak

def _update_risk_score(state: SessionState,
                       interpretation: Interpretation) -> None:
    """
    Conservative asymmetric risk scoring:
      - Accumulation : immediate, full weight, single turn
      - Decay        : only after _SAFE_STREAK_REQUIRED consecutive safe turns
      - Per streak   : only _DECAY_AMOUNT points removed (never a big drop)
      - Session floor: score never decays below 50% of the session peak
      - Floor        : 0, never negative
      - Streak reset : any risk > 0 OR intent is distress_signal / deflecting
    """
    from models import Intent

    risk_val   = interpretation.risk_level.value
    intent_val = interpretation.intent.value
    points     = _RISK_SCORE_MAP.get(risk_val, 0)

    # ── ACCUMULATION PATH ─────────────────────
    if points > 0:
        state.cumulative_risk_score += points
        state.safe_turn_streak       = 0          # reset streak immediately
        state.peak_risk_score        = max(
            state.peak_risk_score,
            state.cumulative_risk_score
        )
        return

    # ── SAFE TURN CHECK ────────────────────
    # A turn must have risk=none AND a non-suspicious intent to advance streak.
    # Neutral/unknown intents neither advance nor reset — they are ignored.
    is_safe = (risk_val == "none" and intent_val in _SAFE_INTENTS)

    if not is_safe:
        # Suspicious intent (deflecting, distress_signal) — do not advance streak
        # but also do not reset it (too harsh for a single deflection)
        # Only a risk > 0 resets the streak (handled above)
        return

    # ── STREAK ACCUMULATION ────────────
    state.safe_turn_streak += 1

    # ── DECAY: only fires when streak is complete ────────────
    if state.safe_turn_streak >= _SAFE_STREAK_REQUIRED:
        session_floor = int(state.peak_risk_score * _MAX_DECAY_FRACTION)
        new_score     = state.cumulative_risk_score - _DECAY_AMOUNT
        state.cumulative_risk_score = max(new_score, session_floor, 0)
        state.safe_turn_streak      = 0   # must earn next streak from scratch
        print(
            f"[RiskTracker] ✅ Safe streak completed → "
            f"score={state.cumulative_risk_score} "
            f"(floor={session_floor})"
        )


def build_initial_state(child_id: str, session_id: str) -> SessionState:
    """Create a fresh SessionState at session start."""
    state = SessionState(child_id=child_id, session_id=session_id)
    state.goal_stack = [Goal(name=g) for g in DEFAULT_GOALS]
    return state


def _compute_engagement(state: SessionState, utterance: str) -> EngagementLevel:
    """
    Estimate engagement from utterance length + deflection count.
    Simple heuristic — no LLM needed.
    """
    if state.deflection_count >= 3:
        return EngagementLevel.LOW
    word_count = len(utterance.split())
    if word_count >= 15:
        return EngagementLevel.HIGH
    if word_count >= 5:
        return EngagementLevel.MODERATE
    return EngagementLevel.LOW


def _compute_response_quality(utterance: str, intent) -> ResponseQuality:
    """
    Estimate whether the child gave a substantive answer.
    """
    from models import Intent
    vague_phrases = [
        "i don't know", "idk", "dunno", "maybe", "i guess",
        "not sure", "whatever", "nothing", "don't care", "no idea"
    ]
    u_lower = utterance.lower().strip()

    if len(u_lower) <= 3:
        return ResponseQuality.NONE
    if any(phrase in u_lower for phrase in vague_phrases):
        return ResponseQuality.VAGUE
    if intent == Intent.DEFLECTING:
        return ResponseQuality.NONE
    return ResponseQuality.CLEAR


def _update_goals(state: SessionState):
    """
    Mark goals as done based on current state signals.
    Called after every turn with the child's response already processed.

    KEY RULE: elicit_child_solution is only marked done when the child
    has actually responded clearly to socratic_generate — not when the
    move fires. This prevents premature session closing.
    """
    moves = set(m.value for m in state.moves_used)

    for goal in state.goal_stack:
        if goal.done:
            continue

        if goal.name == "acknowledge_child_feeling":
            if (DialogueMove.VALIDATE.value in moves or
                    DialogueMove.REFLECT_BACK.value in moves):
                goal.done = True

        elif goal.name == "understand_context":
            # Probe fired AND child gave substantive content in response
            if (DialogueMove.SOCRATIC_PROBE.value in moves and
                    state.child_gave_content):
                goal.done = True

        elif goal.name == "explore_pattern":
            # Explore fired AND child gave content
            if (DialogueMove.SOCRATIC_EXPLORE.value in moves and
                    state.child_gave_content):
                goal.done = True
                state.pattern_named = True

        elif goal.name == "elicit_child_solution":
            # Path A: teach_skill was used — robot provided the skill
            if DialogueMove.TEACH_SKILL.value in moves:
                goal.done = True
            # Path B: socratic_generate fired AND child gave a CLEAR answer
            # Both conditions must be true — move fired is not enough
            elif (DialogueMove.SOCRATIC_GENERATE.value in moves and
                  state.child_response_quality == ResponseQuality.CLEAR and
                  state.child_gave_content):
                goal.done = True
            # If child said "I don't know" → goal stays open → teach_skill unlocks

        elif goal.name == "close_warmly":
            if state.fsm_phase == FSMPhase.CLOSING:
                goal.done = True


def goal_stack_status(state: SessionState) -> dict:
    """Return goal stack as a plain dict for the turn trace."""
    return {g.name: g.done for g in state.goal_stack}


def update_state(state: SessionState,
                 utterance: str,
                 interpretation: Interpretation,
                 selected_move: DialogueMove) -> SessionState:
    """
    Update SessionState after one complete turn.
    Called by the Turn Orchestrator after Layer 3 selects a move
    and after the child's utterance has been recorded.
    """
    from models import Intent

    # ── Basic counters ──────────────
    state.turn_count += 1

    # ── Emotion trajectory ────────────────────
    state.emotion_trajectory.append(interpretation.emotion)

    # ── Deflection tracking ───────────────────
    if interpretation.intent == Intent.DEFLECTING:
        state.deflection_count += 1
    else:
        state.deflection_count = 0          # reset on re-engagement

        # ── Track last intent explicitly ───────────────────
    state.last_intent = interpretation.intent.value
    # ── Cumulative risk scoring — conservative asymmetric decay ──────────
    _update_risk_score(state, interpretation)

    # ── Move history (keep duplicates for loop detection) ─────────────
    state.moves_used.append(selected_move)

    # ── Loop counter: consecutive repeats of same move ────────────────
    move_val = selected_move.value
    # Increment current move, reset all others
    for k in list(state.loop_counter.keys()):
        state.loop_counter[k] = 0
    state.loop_counter[move_val] = state.loop_counter.get(move_val, 0) + 1

    # ── Engagement and response quality ─────────
    state.engagement_level       = _compute_engagement(state, utterance)
    state.child_response_quality = _compute_response_quality(
                                       utterance, interpretation.intent)
    state.child_gave_content     = (
        state.child_response_quality == ResponseQuality.CLEAR
    )

    # ── Update goal stack ────────────────
    _update_goals(state)

    # ── Phase auto-advance ────────────────────────────────────────────
    # Only advance to CLOSING when all non-close goals are genuinely done.
    # Never before turn 6 — give the conversation room to breathe.
    goals_by_name = {g.name: g for g in state.goal_stack}
    non_close     = [g for g in state.goal_stack if g.name != "close_warmly"]

    if (state.turn_count >= 6 and
            all(g.done for g in non_close) and
            state.fsm_phase not in (FSMPhase.CLOSING, FSMPhase.HOLDING)):
        state.fsm_phase = FSMPhase.CLOSING

    return state


# ── test ───────────────────────
if __name__ == "__main__":
    from models import Emotion, Intent, RiskLevel, DialogueMove

    print("=== Session State Tracker Test ===\n")
    state  = build_initial_state("child_001", "s001")
    interp = Interpretation(
        emotion    = Emotion.ANGRY,
        intent     = Intent.VENTING,
        risk_level = RiskLevel.NONE,
        confidence = 0.92
    )

    print("Before update:")
    print(f"  turn_count       = {state.turn_count}")
    print(f"  emotion_traj     = {state.emotion_trajectory}")
    print(f"  engagement       = {state.engagement_level}")
    print(f"  goals            = {goal_stack_status(state)}")

    # Turn 1 — validate
    state.add_turn("child", "I HATE this! I always lose!")
    state = update_state(state, "I HATE this! I always lose!",
                         interp, DialogueMove.VALIDATE)
    state.add_turn("robot", "Ugh, losing when you really wanted to win feels awful.")
    print(f"\nAfter turn 1 (validate):")
    print(f"  turn_count       = {state.turn_count}")
    print(f"  emotion_traj     = {[e.value for e in state.emotion_trajectory]}")
    print(f"  engagement       = {state.engagement_level.value}")
    print(f"  response_quality = {state.child_response_quality.value}")
    print(f"  goals            = {goal_stack_status(state)}")

    # Turn 2 — deflection
    interp2 = Interpretation(
        emotion=Emotion.WITHDRAWN, intent=Intent.DEFLECTING,
        risk_level=RiskLevel.NONE, confidence=0.85
    )
    state.add_turn("child", "Whatever.")
    state = update_state(state, "Whatever.", interp2,
                         DialogueMove.NORMALIZE_AND_HOLD_SPACE)
    print(f"\nAfter turn 2 (deflection):")
    print(f"  deflection_count = {state.deflection_count}")
    print(f"  engagement       = {state.engagement_level.value}")
    print(f"  response_quality = {state.child_response_quality.value}")

    # Simulate socratic_generate fires but child gives vague answer
    for move, utterance, emotion, intent in [
        (DialogueMove.REFLECT_BACK,       "Jake cheated",          Emotion.ANGRY,   Intent.SHARING),
        (DialogueMove.SOCRATIC_PROBE,     "Yeah it felt horrible", Emotion.ANGRY,   Intent.SHARING),
        (DialogueMove.SOCRATIC_EXPLORE,   "With my brother too",   Emotion.ANGRY,   Intent.SHARING),
        (DialogueMove.SOCRATIC_GENERATE,  "I don't know",          Emotion.NEUTRAL, Intent.SHARING),
    ]:
        interp_n = Interpretation(emotion=emotion, intent=intent,
                                  risk_level=RiskLevel.NONE, confidence=0.9)
        state.add_turn("child", utterance)
        state = update_state(state, utterance, interp_n, move)

    print(f"\nAfter socratic_generate with vague response:")
    print(f"  elicit_child_solution done = "
          f"{goal_stack_status(state)['elicit_child_solution']}")
    print(f"  FSM phase = {state.fsm_phase.value}  (should NOT be CLOSING yet)")
    print(f"  Expected: teach_skill should now be eligible")

    # Now child gives clear answer
    interp_clear = Interpretation(emotion=Emotion.NEUTRAL, intent=Intent.SHARING,
                                  risk_level=RiskLevel.NONE, confidence=0.9)
    state.add_turn("child", "I could go build something with Lego to calm down")
    state = update_state(state, "I could go build something with Lego to calm down",
                         interp_clear, DialogueMove.TEACH_SKILL)
    print(f"\nAfter teach_skill + clear child response:")
    print(f"  elicit_child_solution done = "
          f"{goal_stack_status(state)['elicit_child_solution']}")
    print(f"  FSM phase = {state.fsm_phase.value}")
    print(f"  All goals = {goal_stack_status(state)}")