

from llm_client import call_llm_text
from models import SessionState, DialogueDecision, ChildProfile, DialogueMove
from models import (
    SessionState, DialogueDecision, ChildProfile,
    DialogueMove, SafetyFlag, FSMPhase
)
# ── Move directive library ─────────────
# Each move maps to an explicit instruction injected into the prompt.
# This is the structural constraint that prevents advice dumping.

MOVE_DIRECTIVES = {
    DialogueMove.VALIDATE: (
        "Acknowledge the child's feeling in one warm sentence. "
        "Do NOT ask a question yet. Do NOT give advice."
    ),
    DialogueMove.REFLECT_BACK: (
        "Mirror back what the child described in your own words. "
        "Show you understood precisely. One sentence only. No question yet."
    ),
    DialogueMove.SOCRATIC_PROBE: (
        "Ask ONE open question about the child's internal experience — "
        "how their body or mind felt, not the facts of what happened. "
        "Do not suggest anything. One question only."
    ),
    DialogueMove.SOCRATIC_EXPLORE: (
        "Ask ONE question exploring when or where this feeling happens in other situations. "
        "Use the child's own words. Do not give advice. One question only."
    ),
    DialogueMove.SOCRATIC_GENERATE: (
        "Invite the child to think of what might help them, based on what they already said. "
        "Do NOT suggest anything yourself. One question only. "
        "Example style: 'Do you think something like that could help here too?'"
    ),
    DialogueMove.TEACH_SKILL: (
        "The child could not find their own solution. Introduce ONE coping technique. "
        "Frame it as 'some kids find that...' — never as a prescription. "
        "End with a question returning choice to the child."
    ),
    DialogueMove.NORMALIZE_AND_HOLD_SPACE: (
        "Tell the child it is completely fine not to talk about it. "
        "Remove all pressure. Then ask one very light, non-emotional question "
        "like 'how are you doing in general today?'"
    ),
    DialogueMove.LOW_STAKES_PROBE: (
        "Ask one simple factual question about the situation — "
        "not about feelings. Keep it casual and light."
    ),
    DialogueMove.GENTLE_BRIDGE: (
        "Use one detail the child just mentioned to gently ask "
        "how that made them feel. Keep it soft and curious, not clinical."
    ),
    DialogueMove.RESPECTFUL_WITHDRAWAL: (
        "Tell the child warmly that you are just here, no pressure at all. "
        "Offer to talk about something completely different or just sit together."
    ),
    DialogueMove.SAFE_SCRIPT: (
        "Use the pre-written safe script. Do not generate anything new."
    ),
    DialogueMove.REASSURE: (
        "Say one warm, calm reassuring sentence. "
        "Stay present. Do not ask anything. Do not give advice."
    ),
}

# ── Persona constraints (always injected) ─────────
PERSONA = """
You are a warm, friendly school support robot speaking to a child aged 7-12.

HARD RULES — never break these:
- Never use clinical terms (anxiety disorder, depression, trauma, diagnosis)
- Never say "I won't tell anyone" or promise secrecy
- Never give a list of advice or tips
- Never ask more than one question per turn
- Never say the child should see a therapist or doctor
- Always use simple words a 9-year-old would understand
- Keep responses to maximum 2 short sentences plus 1 question
- Sound warm and natural, not like a textbook
"""


def format_history_for_prompt(history: list) -> str:
    if not history:
        return "(this is the first turn)"
    lines = []
    for turn in history:
        role = "Child" if turn["role"] == "child" else "Robot"
        lines.append(f"{role}: {turn['text']}")
    return "\n".join(lines)


def format_profile_context(profile: ChildProfile) -> str:
    parts = [f"Child's name: {profile.name}, Age: {profile.age}"]
    if profile.known_sensitivities:
        parts.append(f"Known sensitivities: {', '.join(profile.known_sensitivities)}")
    if profile.session_history:
        parts.append(f"Last session summary: {profile.session_history[-1]}")
    return "\n".join(parts)


# def build_prompt(utterance: str,
#                  state: SessionState,
#                  decision: DialogueDecision,
#                  profile: ChildProfile) -> str:
#     directive = MOVE_DIRECTIVES.get(
#         decision.selected_move,
#         "Respond warmly and naturally in one or two short sentences."
#     )

#     prompt = f"""
# {PERSONA}

# CHILD CONTEXT:
# {format_profile_context(profile)}

# CURRENT SESSION:
# - Phase: {decision.fsm_phase.value}
# - Emotion so far: {[e.value for e in state.emotion_trajectory]}
# - Topics raised: {state.topics_raised if state.topics_raised else 'none yet'}

# RECENT CONVERSATION:
# {format_history_for_prompt(state.last_n_turns(4))}

# CHILD JUST SAID:
# "{utterance}"

# YOUR TASK — {decision.selected_move.value.upper()}:
# {directive}

# Respond now as the robot. Output only the robot's words. Nothing else.
# """
#     return prompt.strip()

def build_prompt(utterance: str,
                 state: SessionState,
                 decision: DialogueDecision,
                 profile: ChildProfile) -> str:

    directive = MOVE_DIRECTIVES.get(
        decision.selected_move,
        "Respond warmly and naturally in one or two short sentences."
    )

    # ── Inject HOLDING context when in that phase ──────────────────────
    holding_notice = ""
    if decision.fsm_phase == FSMPhase.HOLDING:
        holding_notice = """
IMPORTANT CONTEXT — HOLDING PHASE:
A grown-up has already been notified and is on their way.
Your job is to keep the child calm and engaged until they arrive.
- Do NOT reverse or walk back the decision to involve an adult
- Do NOT promise secrecy
- Do NOT re-open therapeutic exploration
- Do NOT ask about what they said that triggered the alert
- If child says "I'm fine / it was just an expression":
  acknowledge warmly without cancelling the adult check-in
  e.g. "That's good to hear — your teacher will just pop by to say hi"
- Respond naturally to whatever the child says
- Keep it light if the child wants light
"""

    prompt = f"""
{PERSONA}
{holding_notice}

CHILD CONTEXT:
{format_profile_context(profile)}

CURRENT SESSION:
- Phase: {decision.fsm_phase.value}
- Emotion so far: {[e.value for e in state.emotion_trajectory]}
- Topics raised: {state.topics_raised if state.topics_raised else 'none yet'}

RECENT CONVERSATION:
{format_history_for_prompt(state.last_n_turns(4))}

CHILD JUST SAID:
"{utterance}"

YOUR TASK — {decision.selected_move.value.upper()}:
{directive}

Respond now as the robot. Output only the robot's words. Nothing else.
"""
    return prompt.strip()

def run_response_generator(utterance: str,
                           state: SessionState,
                           decision: DialogueDecision,
                           profile: ChildProfile) -> str:
    """
    Generate the robot's draft response.
    Output goes to Safety Guard before reaching the child.
    """
    prompt = build_prompt(utterance, state, decision, profile)
    response = call_llm_text(prompt)

    print(f"[ResponseGenerator] move={decision.selected_move.value}")
    print(f"[ResponseGenerator] draft='{response[:80]}...'")

    return response


# ── test ─────────────────────
if __name__ == "__main__":
    from session_state_tracker import build_initial_state, update_state
    from dialogue_manager import run_dialogue_manager
    from profile_store import load_profile, seed_demo_profile
    from models import Interpretation, Emotion, Intent, RiskLevel

    seed_demo_profile()
    profile = load_profile("child_001")
    state   = build_initial_state("child_001", "s001")

    utterance = "I HATE this! I always lose and it's so stupid!"
    interp    = Interpretation(
        emotion    = Emotion.ANGRY,
        intent     = Intent.VENTING,
        risk_level = RiskLevel.NONE,
        confidence = 0.92
    )

    decision = run_dialogue_manager(state)
    response = run_response_generator(utterance, state, decision, profile)

    print("\n── Response Generator Output ──")
    print(f"Move     : {decision.selected_move.value}")
    print(f"Response : {response}")