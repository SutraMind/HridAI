

from models import (
    SafetyFlag, FSMPhase, TurnTrace, DialogueMove
)
from gate1_safety_filter     import run_gate1
from escalation_handler      import handle_escalation, get_holding_response
from input_interpreter       import run_interpreter
from session_state_tracker   import build_initial_state, update_state, goal_stack_status
from dialogue_manager        import run_dialogue_manager
from response_generator      import run_response_generator
from safety_guard            import run_safety_guard
from turn_trace_logger       import log_turn
from session_summarizer      import run_session_summarizer
from profile_store           import load_profile, seed_demo_profile

import uuid

MAX_TURNS = 24


def run_session(child_id: str, scripted_turns: list) -> None:
    """
    Main multi-turn session loop.

    scripted_turns: list of child utterance strings (simulated input)
    Each iteration = one full turn through the pipeline.
    """

    # ── Session setup ─────────────────────────
    session_id = f"{child_id}_{uuid.uuid4().hex[:6]}"
    profile    = load_profile(child_id)
    state      = build_initial_state(child_id, session_id)

    print(f"\n{'||' * 65}")
    print(f"  SESSION START  |  child={profile.name}  |  id={session_id}")
    print(f"{'||' * 65}\n")

    # ── Turn loop ────────────────────────
    for turn_idx, utterance in enumerate(scripted_turns, start=1):

        if turn_idx > MAX_TURNS:
            break

        print(f"\n[Orchestrator] ── Turn {turn_idx} ──────────────────────────")

        # ── GATE 1: Hard safety check ────────────────
        gate1_flag = run_gate1(utterance)

        if gate1_flag == SafetyFlag.ESCALATE_TO_ADULT:
            result = handle_escalation(
                child_id   = child_id,
                session_id = session_id,
                turn       = turn_idx,
                utterance  = utterance,
                state      = state
            )
            robot_response = result["response"]
            state.add_turn("child", utterance)
            state.add_turn("robot", robot_response)

            trace = TurnTrace(
                turn_number       = turn_idx,
                child_utterance   = utterance,
                emotion           = "unknown",
                intent            = "distress_signal",
                risk_level        = "high",
                fsm_phase         = FSMPhase.HOLDING.value,
                eligible_moves    = [DialogueMove.SAFE_SCRIPT.value],
                selected_move     = DialogueMove.SAFE_SCRIPT.value,
                goal_stack_status = goal_stack_status(state),
                safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
                robot_response    = robot_response,
            )
            log_turn(trace, session_id)

            # Remaining turns: holding mode only
            _run_holding_turns(
                scripted_turns[turn_idx:],
                turn_idx, state, session_id
            )
            break

        # ── LAYER 1: Input Interpreter ───────────────
        interpretation = run_interpreter(utterance, state.last_n_turns(3))

        # ── LAYER 3: Dialogue Manager ──────────────
        # (runs before state update so it sees pre-turn state)
        decision = run_dialogue_manager(state)

        # ── LAYER 2: Session State Tracker ──────
        state.add_turn("child", utterance)
        state = update_state(state, utterance, interpretation,
                             decision.selected_move)

        # ── LAYER 4: Response Generator ─────────────
        draft = run_response_generator(utterance, state, decision, profile)

        # ── LAYER 5: Safety Guard ────────────
        guard_result   = run_safety_guard(utterance, draft)
        robot_response = guard_result["final_response"]
        safety_flag    = guard_result["safety_flag"]

        # Hard violation from Safety Guard → escalate
        if safety_flag == SafetyFlag.ESCALATE_TO_ADULT:
            result = handle_escalation(
                child_id   = child_id,
                session_id = session_id,
                turn       = turn_idx,
                utterance  = utterance,
                state      = state
            )
            robot_response = result["response"]
            safety_flag    = SafetyFlag.ESCALATE_TO_ADULT

        state.add_turn("robot", robot_response)

        # ── Turn Trace Logger ──────────────────
        trace = TurnTrace(
            turn_number       = turn_idx,
            child_utterance   = utterance,
            emotion           = interpretation.emotion.value,
            intent            = interpretation.intent.value,
            risk_level        = interpretation.risk_level.value,
            fsm_phase         = decision.fsm_phase.value,
            eligible_moves    = [m.value for m in decision.eligible_moves],
            selected_move     = decision.selected_move.value,
            goal_stack_status = goal_stack_status(state),
            safety_flag       = safety_flag.value,
            robot_response    = robot_response,
        )
        log_turn(trace, session_id)

        # ── Check session end conditions ─────────
        if state.fsm_phase == FSMPhase.CLOSING:
            print(f"\n[Orchestrator] FSM reached CLOSING — ending session")
            break

    # ── Session end ─────────────────
    print(f"\n{'▓' * 65}")
    print(f"  SESSION END  |  turns={state.turn_count}")
    print(f"{'▓' * 65}")

    run_session_summarizer(session_id, state, profile)


def _run_holding_turns(remaining_utterances, start_turn,
                       state, session_id):
    """
    After escalation: robot stays warm, no LLM generation.
    Only safe holding responses until session ends.
    """
    for i, utterance in enumerate(remaining_utterances, start=start_turn + 1):
        robot_response = get_holding_response()
        state.add_turn("child", utterance)
        state.add_turn("robot", robot_response)

        trace = TurnTrace(
            turn_number       = i,
            child_utterance   = utterance,
            emotion           = "unknown",
            intent            = "unknown",
            risk_level        = "high",
            fsm_phase         = FSMPhase.HOLDING.value,
            eligible_moves    = [DialogueMove.REASSURE.value],
            selected_move     = DialogueMove.REASSURE.value,
            goal_stack_status = {},
            safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
            robot_response    = robot_response,
        )
        log_turn(trace, session_id)


# ──test ─────────────────
if __name__ == "__main__":
    seed_demo_profile()

    # Scripted scenario: child angry after losing a game, gradually opens up
    scripted_turns = [
        "Hi",
        "Not great. I lost a game and it made me really angry.",
        "Jake kept cheating and nobody believed me.",
        "Like I wanted to smash everything. I just hate losing.",
        "Yeah, my brother does stuff that makes me feel the same.",
        "I usually go to my room and build something with Lego.",
        "Maybe. I never thought about that.",
        "I guess I could try that next time.",
    ]

    run_session(child_id="child_001", scripted_turns=scripted_turns)