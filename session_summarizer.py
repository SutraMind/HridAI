

import json
import os
from llm_client import call_llm_text
from models import ChildProfile, SessionState
from profile_store import save_profile

SUMMARIZER_PROMPT = """
You are summarizing a session between a school support robot and a child.
Write a concise 3-sentence summary suitable for a teacher or robot to read
before the next session.

Rules:
- Neutral and descriptive — not diagnostic or evaluative
- Mention: main topic discussed, dominant emotions observed,
  any unresolved threads or notable moments
- No clinical language
- Do not name the child

SESSION LOG:
{session_log}

Write exactly 3 sentences. Nothing else.
"""


def load_session_log(session_id: str) -> list:
    path = os.path.join("session_logs", f"{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def format_log_for_prompt(log: list) -> str:
    lines = []
    for turn in log:
        lines.append(f"Turn {turn['turn']}")
        lines.append(f"  Child : {turn['child']}")
        lines.append(f"  Robot : {turn['robot']}")
        lines.append(f"  Emotion: {turn['emotion']} | Move: {turn['selected_move']}")
    return "\n".join(lines)


def run_session_summarizer(session_id: str,
                           state: SessionState,
                           profile: ChildProfile) -> str:
    """
    Generate a 3-sentence session summary and append to child profile.
    Runs once at session end.
    """
    log = load_session_log(session_id)
    if not log:
        print("[Summarizer] No session log found — skipping")
        return ""

    prompt  = SUMMARIZER_PROMPT.format(
        session_log = format_log_for_prompt(log)
    )
    summary = call_llm_text(prompt)

    # Append to child profile
    profile.session_history.append(summary)

    # Update recurring emotions from this session
    session_emotions = list({t["emotion"] for t in log if t["emotion"] != "unknown"})
    for e in session_emotions:
        if e not in profile.recurring_emotions:
            profile.recurring_emotions.append(e)

    # Update previous topics
    for topic in state.topics_raised:
        if topic not in profile.previous_topics_discussed:
            profile.previous_topics_discussed.append(topic)

    save_profile(profile)

    print(f"\n[Summarizer] Session summary written to profile:")
    print(f"  {summary}")
    return summary