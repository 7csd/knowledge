import json
from datetime import datetime, timezone
from pathlib import Path


class Trace:
    def __init__(self, run_id: str, path: Path | None = None, secrets: tuple[str, ...] = ()):
        self.run_id = run_id
        self.path = path
        self.secrets = tuple(value for value in secrets if value)
        self.events: list[dict] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Each run is a separate artifact; never silently overwrite an earlier trace.
            with path.open("x", encoding="utf-8"):
                pass

    def record(self, event: str, **payload):
        item = {
            "run_id": self.run_id,
            "sequence": len(self.events),
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        serialized = json.dumps(item, ensure_ascii=False, default=str)
        for secret in self.secrets:
            serialized = serialized.replace(secret, "[REDACTED]")
        self.events.append(json.loads(serialized))
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
