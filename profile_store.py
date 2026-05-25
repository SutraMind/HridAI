

import json
import os
from models import ChildProfile

PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)


def profile_path(child_id: str) -> str:
    return os.path.join(PROFILES_DIR, f"{child_id}.json")


def load_profile(child_id: str) -> ChildProfile:
    path = profile_path(child_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return ChildProfile(**data)
    # First-ever session — create blank profile
    return ChildProfile(child_id=child_id, name="Unknown", age=10)


def save_profile(profile: ChildProfile):
    path = profile_path(profile.child_id)
    with open(path, "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(profile), f, indent=2)
    print(f"[ProfileStore] Saved profile for {profile.child_id}")


# ── seed a demo profile ──────────────────
def seed_demo_profile():
    profile = ChildProfile(
        child_id="child_001",
        name="Alex",
        age=9,
        previous_topics_discussed=["feeling left out at lunch"],
        recurring_emotions=["angry", "sad"],
        known_sensitivities=["losing games", "being ignored"],
        session_history=[
            "Alex talked about feeling left out during lunch. "
            "Showed frustration but engaged well. "
            "No safeguarding concerns."
        ]
    )
    save_profile(profile)
    return profile


# ──test ─────────────────────
if __name__ == "__main__":
    p = seed_demo_profile()
    loaded = load_profile("child_001")
    print("Profile loaded:", loaded.name, "| Age:", loaded.age)
    print("Previous topics:", loaded.previous_topics_discussed)
    print("Session history:", loaded.session_history)