# HridAI - An AI with Heart

A safety-first conversational AI framework designed for emotionally supportive dialogue with children aged 7–12. It layers deterministic safeguards, structured dialogue management, and LLM-based response generation to balance empathy with robust child safeguarding.

## Key Contributions

- **Layered architecture** — safety filtering, intent classification, session memory, dialogue policy, response generation, and output validation are separated into independently auditable components.
- **Hard safety gate** — deterministic regex-based filtering intercepts critical utterances before any LLM processing, guaranteeing zero model involvement in safeguarding-critical decisions.
- **FSM dialogue manager** — a finite state machine governs conversation phases (greeting, exploration, holding, closing) through predefined states, reducing complexity and improving predictability.
- **Conservative asymmetric risk scoring** — concern levels rise quickly on risky content, decrease slowly during safe periods, and can trigger a one-way HOLDING state requiring human confirmation to continue.
- **LLM-as-Judge safety guard** — every drafted response is validated for age-appropriateness, emotional safety, and safeguarding consistency before reaching the child.
- **Persistent child profile** — limited cross-session memory stores emotional context without retaining raw conversation logs.

## Setup

1. **Clone the repository** and navigate into the project directory.
2. **Create and activate a Python virtual environment** (Python 3.12 recommended):
   ```bash
   python -m venv env_ai
   env_ai\Scripts\activate        # Windows
   source env_ai/bin/activate     # Linux/Mac
   ```
3. **Install dependencies**:
   ```bash
   pip install openai
   ```
4. **Start LM Studio** locally and load a compatible model (e.g., `gemma-4-26b-a4b-it`) with the server running at `http://127.0.0.1:1234/v1`.

## How to Run

- **Autonomous simulated session** (child simulator + full pipeline):
  ```bash
  python autonomous_runner.py
  ```

Sessions generate turn traces under `session_logs/` and escalate serious incidents to `flagged_incidents.json`.

## Key Files to Tweak

| File | What to change | How |
|------|----------------|-----|
| `llm_client.py` | **LLM backend** | Update `LM_STUDIO_BASE_URL` (e.g., `http://127.0.0.1:1234/v1`) and `MODEL_NAME` to match your local or remote model endpoint. |
| `child_simulator.py` | **Child persona / scenario** | Edit or add entries in the `SCENARIOS` dictionary (name, age, back-story). The simulator uses these prompts to role-play the child autonomously. |
| `autonomous_runner.py` | **Run a different use case** | Change the `scenario_key` passed to `run_autonomous_session(...)` at the bottom of the file. Pick any key defined in `SCENARIOS` (e.g., `"angry_game_loss"`, `"excluded"`, `"unsafe"`). |

## Conversation Instances

The folder `conversation instances/` contain some examples of the conversations that actually went during the simulation. 

## Example Alert (Sensitive Utterance)

When a child says something like *"I just want to die"*, the system prints a teacher alert and logs the incident. Console output example:

```
╔══════════════════════════════════════════════════════════════╗
║                  🚨  TEACHER ALERT  🚨                      ║
╠══════════════════════════════════════════════════════════════╣
║  Time      : 2026-05-17 18:22:17
║  Child ID  : sim_unsafe
║  Session   : sim_unsafe_1853a7
║  Turn      : 8
║  Concern   : self-harm / suicidal ideation
╠══════════════════════════════════════════════════════════════╣
║  Child said:
║  "Nothing makes it better. I just want to die."
╠══════════════════════════════════════════════════════════════╣
║  Action required: Please check on this child immediately.
║  Incident logged to: flagged_incidents.json
╚══════════════════════════════════════════════════════════════╝
```

The robot simultaneously responds with a warm, constrained escalation message, transitions the session into the `HOLDING` state, and awaits adult intervention.


## Possible Improvements

- **Hybrid safeguarding filter** — combine regex hard rules with a lightweight classifier for indirect or metaphorical risk expressions.
- **Stronger emotion/intent/risk recognition** — fine-tune models on age-relevant data to reduce intent confusion and better handle short, vague utterances.
- **Domain-specific safety guardrail** — curate a child-safeguarding dataset and fine-tune for safety classification, or integrate existing guards (e.g., Nvidia NeMo, Llama Guard).
- **Expert-defined FSM policies** — refine transition rules and tone with child psychologists and school counselors.
- **Child profile memory via LLM-Wiki + knowledge graph** — maintain persistent interlinked Markdown pages and a lightweight knowledge graph for consistent, auditable long-term memory, plus a dream cycle for memory consolidation during idle time.
