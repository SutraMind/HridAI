
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ── Enums ─────────────

class Emotion(str, Enum):
    HAPPY     = "happy"
    SAD       = "sad"
    ANXIOUS   = "anxious"
    ANGRY     = "angry"
    WITHDRAWN = "withdrawn"
    NEUTRAL   = "neutral"
    UNKNOWN   = "unknown"

class Intent(str, Enum):
    VENTING         = "venting"
    DEFLECTING      = "deflecting"
    DISTRESS_SIGNAL = "distress_signal"
    SHARING         = "sharing"
    ASKING          = "asking"
    TOPIC_CHANGE    = "topic_change"
    UNKNOWN         = "unknown"

class RiskLevel(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MODERATE = "moderate"

class SafetyFlag(str, Enum):
    NONE              = "NONE"
    TOPIC_SENSITIVE   = "TOPIC_SENSITIVE"
    ESCALATE_TO_ADULT = "ESCALATE_TO_ADULT"
    CONTENT_VIOLATION = "CONTENT_VIOLATION"

class FSMPhase(str, Enum):
    GREETING          = "GREETING"
    RAPPORT_BUILDING  = "RAPPORT_BUILDING"
    TOPIC_EXPLORATION = "TOPIC_EXPLORATION"
    SKILL_TEACHING    = "SKILL_TEACHING"
    CLOSING           = "CLOSING"
    HOLDING           = "HOLDING"

class DialogueMove(str, Enum):
    VALIDATE                  = "validate"
    REFLECT_BACK              = "reflect_back"
    SOCRATIC_PROBE            = "socratic_probe"
    SOCRATIC_EXPLORE          = "socratic_explore"
    SOCRATIC_GENERATE         = "socratic_generate"
    TEACH_SKILL               = "teach_skill"
    NORMALIZE_AND_HOLD_SPACE  = "normalize_and_hold_space"
    LOW_STAKES_PROBE          = "low_stakes_probe"
    GENTLE_BRIDGE             = "gentle_bridge"
    RESPECTFUL_WITHDRAWAL     = "respectful_withdrawal"
    SAFE_SCRIPT               = "safe_script"
    REASSURE                  = "reassure"

class EngagementLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"

class ResponseQuality(str, Enum):
    CLEAR  = "clear"
    VAGUE  = "vague"
    NONE   = "none"


# ── Child Profile ───────────────

@dataclass
class ChildProfile:
    child_id:                  str
    name:                      str
    age:                       int
    previous_topics_discussed: List[str]       = field(default_factory=list)
    recurring_emotions:        List[str]        = field(default_factory=list)
    known_sensitivities:       List[str]        = field(default_factory=list)
    session_history:           List[str]        = field(default_factory=list)


# ── Goal ────────────────

@dataclass
class Goal:
    name: str
    done: bool = False


# ── Session State ─────────────────

@dataclass
class SessionState:
    child_id:               str
    session_id:             str
    turn_count:             int                  = 0
    fsm_phase:              FSMPhase             = FSMPhase.GREETING
    emotion_trajectory:     List[Emotion]        = field(default_factory=list)
    topics_raised:          List[str]            = field(default_factory=list)
    moves_used:             List[DialogueMove]   = field(default_factory=list)
    engagement_level:       EngagementLevel      = EngagementLevel.LOW
    deflection_count:       int                  = 0
    child_gave_content:     bool                 = False
    pattern_named:          bool                 = False
    child_response_quality: ResponseQuality      = ResponseQuality.NONE
    goal_stack:             List[Goal]           = field(default_factory=list)
    safety_flag:            SafetyFlag           = SafetyFlag.NONE
    conversation_history:   List[dict]           = field(default_factory=list)
    loop_counter:           dict                 = field(default_factory=dict)
    cumulative_risk_score:  int                  = 0   
    last_intent:            str                  = "" 
    peak_risk_score: int = 0          
    safe_turn_streak: int = 0         
    soft_alert_fired: bool = False    

    def last_n_turns(self, n: int = 4) -> List[dict]:
        return self.conversation_history[-n:]

    def add_turn(self, role: str, text: str):
        self.conversation_history.append({"role": role, "text": text})


# ── Per-Turn Interpreter Output ────────────

@dataclass
class Interpretation:
    emotion:    Emotion
    intent:     Intent
    risk_level: RiskLevel
    confidence: float


# ── Dialogue Decision ─────────────────

@dataclass
class DialogueDecision:
    fsm_phase:        FSMPhase
    eligible_moves:   List[DialogueMove]
    selected_move:    DialogueMove
    goal_stack_status: dict
    loop_detected:    bool


# ── Turn Trace ─────────────────────

@dataclass
class TurnTrace:
    turn_number:       int
    child_utterance:   str
    emotion:           str
    intent:            str
    risk_level:        str
    fsm_phase:         str
    eligible_moves:    List[str]
    selected_move:     str
    goal_stack_status: dict
    safety_flag:       str
    robot_response:    str


# ── quick test ───────────────
if __name__ == "__main__":
    state = SessionState(child_id="child_001", session_id="s001")
    state.add_turn("child", "I hate losing!")
    state.add_turn("robot", "That sounds really frustrating.")
    print("SessionState OK — turns:", len(state.conversation_history))
    print("Last 2 turns:", state.last_n_turns(2))