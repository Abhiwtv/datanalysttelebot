import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

LOG_FILE = STATIC_DIR / "agent_log.jsonl"


def log_event(event_type: str, payload: Dict[str, Any]):
    """Appends a structured event log as a single line in JSONL format."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        "payload": payload,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_log_url(base_url: str = "https://army-mantis-enable.ngrok-free.dev") -> str:
    return f"{base_url}/static/agent_log.jsonl"