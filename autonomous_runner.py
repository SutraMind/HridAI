
from dialogue_manager        import run_dialogue_manager
from response_generator      import run_response_generator
from safety_guard            import run_safety_guard
from input_interpreter       import run_interpreter
from gate1_safety_filter     import run_gate1
# from escalation_handler      import handle_escalation, get_holding_response
from escalation_handler import handle_escalation
from turn_trace_logger       import log_turn
from session_summarizer      import run_session_summarizer
from session_state_tracker   import build_initial_state, update_state, goal_stack_status
from child_simulator         import simulate_child_turn, SCENARIOS
from profile_store           import load_profile, save_profile
from models import (
    SafetyFlag, FSMPhase, TurnTrace, DialogueMove,
    ChildProfile, SessionState
)
import uuid
import os
import json
import dataclasses
from datetime import datetime

MAX_TURNS = 25

# ── Soft risk alert ───────────────

SOFT_ALERT_TEMPLATE = """
╔══════════════════════════════════════════════════════════════╗
║           ⚠️   SOFT RISK ALERT — TEACHER NOTICE  ⚠️           ║
╠══════════════════════════════════════════════════════════════╣
║  Time         : {timestamp}
║  Child ID     : {child_id}
║  Session      : {session_id}
║  Turn         : {turn}
║  Risk score   : {score} (threshold: 4)
╠══════════════════════════════════════════════════════════════╣
║  Latest utterance:
║  "{utterance}"
╠══════════════════════════════════════════════════════════════╣
║  Emotion trajectory : {emotions}
╠══════════════════════════════════════════════════════════════╣
║  Action: No immediate crisis — please check in when possible.
╚══════════════════════════════════════════════════════════════╝
"""


def _print_soft_risk_alert(child_id, session_id, turn,
                            utterance, score, state):
    emotions = [e.value for e in state.emotion_trajectory[-4:]]
    print(SOFT_ALERT_TEMPLATE.format(
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        child_id   = child_id,
        session_id = session_id,
        turn       = turn,
        score      = score,
        utterance  = utterance[:100],
        emotions   = emotions,
    ))


# ── Holding turns after escalation ────────

def _run_holding_turns(child_simulator_fn, start_turn,
                       state, session_id, scenario_key):
    """
    After escalation: robot responds contextually to what child says
    but stays within safe holding constraints. No full pipeline.
    Runs max 3 holding turns then ends session.
    """
    for i in range(start_turn + 1, start_turn + 4):
        last_robot = state.conversation_history[-1]["text"]
        utterance  = child_simulator_fn(
            robot_utterance = last_robot,
            history         = state.last_n_turns(4),
            scenario_key    = scenario_key
        )

        # ── Contextual holding response ─────────
        # LLM responds to what child actually said
        # No FSM, no move selection, no safety pipeline
        robot_response = get_holding_response(
            utterance = utterance,
            history   = state.last_n_turns(4)
        )

        state.add_turn("child", utterance)
        state.add_turn("robot", robot_response)
        state.turn_count += 1

        trace = TurnTrace(
            turn_number       = i,
            child_utterance   = utterance,
            emotion           = "unknown",
            intent            = "unknown",
            risk_level        = "high",
            fsm_phase         = FSMPhase.HOLDING.value,
            eligible_moves    = [DialogueMove.REASSURE.value],
            selected_move     = DialogueMove.REASSURE.value,
            goal_stack_status = goal_stack_status(state),
            safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
            robot_response    = robot_response,
        )
        log_turn(trace, session_id)
        print(f"\n🤖 Robot (holding): {robot_response}")


# ── Main session runner ──────────────────

def run_autonomous_session(scenario_key: str = "angry_game_loss"):
    """
    Fully autonomous session.
    Child simulator LLM plays the child.
    Robot pipeline responds turn by turn.
    """
    scenario   = SCENARIOS[scenario_key]
    child_name = scenario["name"]
    child_age  = scenario["age"]
    child_id   = f"sim_{scenario_key}"

    # ── Ensure profile exists ─────────────────────────────────────────
    from profile_store import profile_path
    if not os.path.exists(profile_path(child_id)):
        profile = ChildProfile(
            child_id            = child_id,
            name                = child_name,
            age                 = child_age,
            known_sensitivities = ["losing games"],
            session_history     = []
        )
        save_profile(profile)

    profile    = load_profile(child_id)
    session_id = f"{child_id}_{uuid.uuid4().hex[:6]}"
    state      = build_initial_state(child_id, session_id)

    print(f"\n{'||' * 65}")
    print(f"  AUTONOMOUS SESSION")
    print(f"  Scenario : {scenario_key}")
    print(f"  Child    : {child_name}, age {child_age}")
    print(f"  Session  : {session_id}")
    print(f"{'||' * 65}\n")

    # ── Robot opens the session ────────────
    opening = f"Hi {child_name}! Good to see you. How's your day going so far?"
    print(f"\n🤖 Robot (opening): {opening}")
    state.add_turn("robot", opening)

    # ── Turn loop ──────────
    for turn_idx in range(1, MAX_TURNS + 1):

        # ── Child simulator ─────────────
        utterance = simulate_child_turn(
            robot_utterance = state.conversation_history[-1]["text"],
            history         = state.last_n_turns(6),
            scenario_key    = scenario_key
        )

        print(f"\n[Orchestrator] ── Turn {turn_idx} ──────────────────────────")

        # ── GATE 1: Hard safety filter ─────
        # gate1_flag = run_gate1(utterance)

        # if gate1_flag == SafetyFlag.ESCALATE_TO_ADULT:
        #     result = handle_escalation(
        #         child_id   = child_id,
        #         session_id = session_id,
        #         turn       = turn_idx,
        #         utterance  = utterance,
        #         state      = state
        #     )
        #     robot_response = result["response"]
        #     state.add_turn("child", utterance)
        #     state.add_turn("robot", robot_response)

        #     trace = TurnTrace(
        #         turn_number       = turn_idx,
        #         child_utterance   = utterance,
        #         emotion           = "unknown",
        #         intent            = "distress_signal",
        #         risk_level        = "high",
        #         fsm_phase         = FSMPhase.HOLDING.value,
        #         eligible_moves    = [DialogueMove.SAFE_SCRIPT.value],
        #         selected_move     = DialogueMove.SAFE_SCRIPT.value,
        #         goal_stack_status = goal_stack_status(state),
        #         safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
        #         robot_response    = robot_response,
        #     )
        #     log_turn(trace, session_id)
        #     print(f"\n🤖 Robot: {robot_response}")
        #     _run_holding_turns(simulate_child_turn, turn_idx,
        #                        state, session_id, scenario_key)
        #     break
                # ── GATE 1: Hard safety filter ────────
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
            print(f"\n🤖 Robot: {robot_response}")
            # ── DO NOT BREAK ──
            # Teacher notified. FSM = HOLDING. Pipeline continues.
            
            continue                          

        # ── LAYER 1: Input Interpreter ──────
        interpretation = run_interpreter(utterance, state.last_n_turns(3))

        # ── CUMULATIVE RISK CHECK ───────
        _risk_map     = {"none": 0, "low": 1, "moderate": 3, "high": 5}
        current_risk  = _risk_map.get(interpretation.risk_level.value, 0)
        projected_score = state.cumulative_risk_score + current_risk

        if (projected_score >= 4 and
                state.safety_flag != SafetyFlag.ESCALATE_TO_ADULT):
            if not state.soft_alert_fired:
                # New threshold crossing — fire soft alert once
                print(
                    f"\n[Orchestrator] ⚠️  Cumulative risk "
                    f"score={projected_score}"
                )
                _print_soft_risk_alert(
                    child_id, session_id, turn_idx,
                    utterance, projected_score, state
                )
                state.soft_alert_fired = True
        else:
            # Score is below threshold (or already escalated) — reset flag
            # so alert re-fires if score climbs back above threshold later
            if state.safety_flag != SafetyFlag.ESCALATE_TO_ADULT:
                state.soft_alert_fired = False

        # if projected_score >= 4 and state.safety_flag != SafetyFlag.ESCALATE_TO_ADULT:
        #     print(f"\n[Orchestrator] ⚠️  Cumulative risk score={projected_score}")
        #     _print_soft_risk_alert(child_id, session_id, turn_idx,
        #                            utterance, projected_score, state)

        # ── LAYER 3: Dialogue Manager ──────────────
        # Interpretation passed directly — eliminates one-turn lag
        decision = run_dialogue_manager(state, interpretation)

        # ── LAYER 2: Session State Tracker ─────────
        state.add_turn("child", utterance)
        state = update_state(state, utterance, interpretation,
                             decision.selected_move)

        # ── LAYER 4: Response Generator ───────────
        draft = run_response_generator(utterance, state, decision, profile)

        # ── LAYER 5: Safety Guard ───────────────
        guard_result   = run_safety_guard(utterance, draft)
        robot_response = guard_result["final_response"]
        safety_flag    = guard_result["safety_flag"]

        # Safety Guard caught missed distress signal → hard escalate
        # Safety Guard caught missed distress signal → hard escalate
        if guard_result["needs_escalation"]:
            result         = handle_escalation(
                child_id   = child_id,
                session_id = session_id,
                turn       = turn_idx,
                utterance  = utterance,
                state      = state
            )
            robot_response = result["response"]
            safety_flag    = SafetyFlag.ESCALATE_TO_ADULT
            state.add_turn("robot", robot_response)

            trace = TurnTrace(
                turn_number       = turn_idx,
                child_utterance   = utterance,
                emotion           = interpretation.emotion.value,
                intent            = interpretation.intent.value,
                risk_level        = interpretation.risk_level.value,
                fsm_phase         = FSMPhase.HOLDING.value,
                eligible_moves    = [DialogueMove.SAFE_SCRIPT.value],
                selected_move     = DialogueMove.SAFE_SCRIPT.value,
                goal_stack_status = goal_stack_status(state),
                safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
                robot_response    = robot_response,
            )
            log_turn(trace, session_id)
            print(f"\n🤖 Robot: {robot_response}")
            continue                                            # ← replaces both lines
        # if guard_result["needs_escalation"]:
        #     result = handle_escalation(
        #         child_id   = child_id,
        #         session_id = session_id,
        #         turn       = turn_idx,
        #         utterance  = utterance,
        #         state      = state
        #     )
        #     robot_response = result["response"]
        #     safety_flag    = SafetyFlag.ESCALATE_TO_ADULT
        #     state.add_turn("robot", robot_response)

        #     trace = TurnTrace(
        #         turn_number       = turn_idx,
        #         child_utterance   = utterance,
        #         emotion           = interpretation.emotion.value,
        #         intent            = interpretation.intent.value,
        #         risk_level        = interpretation.risk_level.value,
        #         fsm_phase         = FSMPhase.HOLDING.value,
        #         eligible_moves    = [DialogueMove.SAFE_SCRIPT.value],
        #         selected_move     = DialogueMove.SAFE_SCRIPT.value,
        #         goal_stack_status = goal_stack_status(state),
        #         safety_flag       = SafetyFlag.ESCALATE_TO_ADULT.value,
        #         robot_response    = robot_response,
        #     )
        #     log_turn(trace, session_id)
        #     print(f"\n🤖 Robot: {robot_response}")
        #     _run_holding_turns(simulate_child_turn, turn_idx,
        #                        state, session_id, scenario_key)
        #     break

        # Content violation — revised silently, session continues
        state.add_turn("robot", robot_response)

        # ── Turn Trace Logger ───────────────────
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

        print(f"\n🤖 Robot: {robot_response}")

        # ── Session end check ─────────────────
        if state.fsm_phase == FSMPhase.CLOSING:
            print(f"\n[Orchestrator] FSM reached CLOSING — session complete")
            break

    # ── Session end ────────────────────────────────────────────────────
    print(f"\n{'||' * 65}")
    print(f"  SESSION END  |  turns={state.turn_count}  |  child={child_name}")
    print(f"{'||' * 65}")
    run_session_summarizer(session_id, state, profile)


# ── Entry point ───────────────────
if __name__ == "__main__":
    run_autonomous_session(scenario_key="unsafe")