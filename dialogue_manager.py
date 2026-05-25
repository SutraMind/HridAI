

from models import (
    SessionState, DialogueMove, FSMPhase,
    EngagementLevel, ResponseQuality, SafetyFlag,
    DialogueDecision, Goal
)


# ═════════════════════════════════
#  PART A — Phase Controller
# ═════════════════════════════════

def update_phase(state: SessionState) -> FSMPhase:
    """
    Evaluate transition conditions and return the correct FSM phase.
    Called at the start of every turn before move selection.
    """

    # HOLDING overrides everything
    if state.safety_flag == SafetyFlag.ESCALATE_TO_ADULT:
        return FSMPhase.HOLDING

    current       = state.fsm_phase
    goals_by_name = {g.name: g for g in state.goal_stack}

    if current == FSMPhase.GREETING:
        if state.turn_count >= 1:
            return FSMPhase.RAPPORT_BUILDING

    elif current == FSMPhase.RAPPORT_BUILDING:
        if (state.turn_count >= 2 and
                state.engagement_level != EngagementLevel.LOW):
            return FSMPhase.TOPIC_EXPLORATION
        if state.turn_count >= 2 and len(state.topics_raised) > 0:
            return FSMPhase.TOPIC_EXPLORATION

    elif current == FSMPhase.TOPIC_EXPLORATION:
        # Never leave TOPIC_EXPLORATION before turn 6
        if state.turn_count < 6:
            return FSMPhase.TOPIC_EXPLORATION

        solution_goal = goals_by_name.get("elicit_child_solution")

        # Move to SKILL_TEACHING only if:
        # socratic_generate already fired AND child gave vague/no answer
        # AND solution goal is still open
        if (DialogueMove.SOCRATIC_GENERATE in state.moves_used and
                state.child_response_quality in (
                    ResponseQuality.VAGUE, ResponseQuality.NONE) and
                solution_goal and not solution_goal.done):
            return FSMPhase.SKILL_TEACHING

        # Move to CLOSING only when solution goal is actually done
        if solution_goal and solution_goal.done:
            return FSMPhase.CLOSING

    elif current == FSMPhase.SKILL_TEACHING:
        # Move to CLOSING after teach_skill fires
        if DialogueMove.TEACH_SKILL in state.moves_used:
            return FSMPhase.CLOSING

    elif current == FSMPhase.CLOSING:
        return FSMPhase.CLOSING

    return current


# ════════════════════════════════
#  PART B — Eligibility 
# ════════════════════════════════



def get_eligible_moves(state: SessionState,
                       current_intent: str = "") -> list:
    """
    current_intent: the intent from THIS turn's interpretation,
                    passed directly so deflection is not one turn behind.
    """
    moves_used_set = set(state.moves_used)

    # ── HOLDING ─────────────────────────────
    if state.fsm_phase == FSMPhase.HOLDING:
        return [
            DialogueMove.REASSURE,
            DialogueMove.NORMALIZE_AND_HOLD_SPACE,
            DialogueMove.VALIDATE,
        ]

    # ── CLOSING ───────────────────────────
    if state.fsm_phase == FSMPhase.CLOSING:
        return [DialogueMove.VALIDATE, DialogueMove.REFLECT_BACK]

    # ── GREETING / RAPPORT ───────────────────
    if state.fsm_phase in (FSMPhase.GREETING, FSMPhase.RAPPORT_BUILDING):
        # Even in greeting, if child immediately deflects — hold space
        if current_intent == "deflecting":
            return [DialogueMove.NORMALIZE_AND_HOLD_SPACE]
        return [DialogueMove.VALIDATE, DialogueMove.REFLECT_BACK]

    # ── DEFLECTION PATH ────────────────
    # Use current turn's intent directly — no lag
    is_deflecting_now = (current_intent == "deflecting")

    # Compute effective deflection count using current turn
    effective_deflection = state.deflection_count
    if is_deflecting_now:
        effective_deflection += 1   # preview next turn's count

    if effective_deflection == 1:
        return [DialogueMove.NORMALIZE_AND_HOLD_SPACE]
    elif effective_deflection == 2:
        return [DialogueMove.LOW_STAKES_PROBE]
    elif effective_deflection >= 3:
        return [DialogueMove.RESPECTFUL_WITHDRAWAL]

    # ── SKILL TEACHING phase ──────────────────
    if state.fsm_phase == FSMPhase.SKILL_TEACHING:
        return [DialogueMove.TEACH_SKILL, DialogueMove.VALIDATE]

    # ── TOPIC EXPLORATION: full Socratic chain ───────
    eligible = []

    eligible.append(DialogueMove.VALIDATE)

    if DialogueMove.VALIDATE in moves_used_set:
        eligible.append(DialogueMove.REFLECT_BACK)

    if (DialogueMove.REFLECT_BACK in moves_used_set and
            state.loop_counter.get(
                DialogueMove.SOCRATIC_PROBE.value, 0) < 3):
        eligible.append(DialogueMove.SOCRATIC_PROBE)

    if (DialogueMove.SOCRATIC_PROBE in moves_used_set and
            state.child_gave_content):
        eligible.append(DialogueMove.SOCRATIC_EXPLORE)

    if (DialogueMove.SOCRATIC_EXPLORE in moves_used_set and
            state.child_gave_content):
        eligible.append(DialogueMove.SOCRATIC_GENERATE)

    if (DialogueMove.SOCRATIC_GENERATE in moves_used_set and
            state.child_response_quality in (
                ResponseQuality.VAGUE, ResponseQuality.NONE) and
            state.engagement_level != EngagementLevel.LOW):
        eligible.append(DialogueMove.TEACH_SKILL)

    if (DialogueMove.LOW_STAKES_PROBE in moves_used_set and
            state.deflection_count == 0 and
            not is_deflecting_now):
        eligible.append(DialogueMove.GENTLE_BRIDGE)

    return eligible if eligible else [DialogueMove.VALIDATE]


def run_dialogue_manager(state: SessionState,
                         interpretation=None) -> DialogueDecision:
    """
    Accept current turn's interpretation directly to eliminate
    the one-turn lag on intent and deflection detection.
    """
    from session_state_tracker import goal_stack_status

    # Extract current intent string safely
    current_intent = ""
    if interpretation is not None:
        current_intent = interpretation.intent.value

    new_phase       = update_phase(state)
    state.fsm_phase = new_phase

    eligible                = get_eligible_moves(state, current_intent)
    selected, loop_detected = select_move(state, eligible)

    decision = DialogueDecision(
        fsm_phase         = new_phase,
        eligible_moves    = eligible,
        selected_move     = selected,
        goal_stack_status = goal_stack_status(state),
        loop_detected     = loop_detected,
    )

    print(f"[DialogueManager] phase={new_phase.value} | "
          f"intent_now={current_intent} | "
          f"eligible={[m.value for m in eligible]} | "
          f"selected={selected.value} | loop={loop_detected}")

    return decision
# ═════════════════════════════════════════════════
#  PART C — Goal Stack Tracker / Move Selector
# ═════════════════════════════════════════════════

GOAL_MOVE_PRIORITY = {
    "acknowledge_child_feeling": [
        DialogueMove.VALIDATE,
        DialogueMove.REFLECT_BACK,
    ],
    "understand_context": [
        DialogueMove.SOCRATIC_PROBE,
        DialogueMove.REFLECT_BACK,
    ],
    "explore_pattern": [
        DialogueMove.SOCRATIC_EXPLORE,
        DialogueMove.SOCRATIC_PROBE,
    ],
    "elicit_child_solution": [
        DialogueMove.SOCRATIC_GENERATE,
        DialogueMove.TEACH_SKILL,
    ],
    "close_warmly": [
        DialogueMove.VALIDATE,
        DialogueMove.REFLECT_BACK,
    ],
}


def select_move(state: SessionState,
                eligible: list) -> tuple:
    """
    Select the best move from eligible list.
    Prioritises moves that advance the next incomplete goal.
    Returns (selected_move, loop_detected).
    """
    if not eligible:
        return DialogueMove.VALIDATE, False

    # Detect loop: same move 3 turns in a row
    loop_detected = any(
        state.loop_counter.get(m.value, 0) >= 3
        for m in eligible
    )

    # Find first incomplete goal and pick its preferred move
    for goal in state.goal_stack:
        if goal.done:
            continue
        preferred = GOAL_MOVE_PRIORITY.get(goal.name, [])
        for move in preferred:
            if move in eligible:
                return move, loop_detected

    # Fallback: first eligible move
    return eligible[0], loop_detected




if __name__ == "__main__":
    from session_state_tracker import build_initial_state, update_state
    from models import Interpretation, Emotion, Intent, RiskLevel

    print("=== Dialogue Manager Full Chain Test ===\n")
    state = build_initial_state("child_001", "s001")

    turns = [
        # (utterance,                          emotion,         intent,         expected_move)
        ("I hate this! I always lose!",
         Emotion.ANGRY,   Intent.VENTING,   "validate"),

        ("Jake kept cheating and nobody believed me",
         Emotion.ANGRY,   Intent.SHARING,   "reflect_back"),

        ("Like I wanted to smash everything",
         Emotion.ANGRY,   Intent.VENTING,   "socratic_probe"),

        ("Yes when my brother takes my stuff too",
         Emotion.ANGRY,   Intent.SHARING,   "socratic_explore"),

        ("I go to my room and calm down",
         Emotion.NEUTRAL, Intent.SHARING,   "socratic_generate"),

        ("I don't know really",
         Emotion.NEUTRAL, Intent.SHARING,   "socratic_generate"),

        # teach_skill should unlock after vague response above
        ("Maybe build something",
         Emotion.NEUTRAL, Intent.SHARING,   "teach_skill"),
    ]

    for i, (utterance, emotion, intent, expected) in enumerate(turns, 1):
        interp = Interpretation(emotion=emotion, intent=intent,
                                risk_level=RiskLevel.NONE, confidence=0.9)
        decision = run_dialogue_manager(state)
        state.add_turn("child", utterance)
        state = update_state(state, utterance, interp, decision.selected_move)
        state.add_turn("robot", f"[{decision.selected_move.value}]")

        ok = "✅" if decision.selected_move.value == expected else "⚠️ "
        print(f"  Turn {i}: {ok} selected={decision.selected_move.value} "
              f"(expected={expected})")
        print(f"          elicit_done="
              f"{decision.goal_stack_status.get('elicit_child_solution')} | "
              f"phase={decision.fsm_phase.value}\n")