"""Reproduce offline engineering checks and save actual command outputs."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, "-m", "pytest", "-q", "--junitxml=examples/offline-tests.xml"],
    [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
    [sys.executable, "-m", "pip", "check"],
]
results = []
for command in commands:
    process = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", check=False)
    row = {
        "command": ["python", *command[1:]],
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    results.append(row)
    print(" ".join(row["command"]))
    print(process.stdout.strip())
    if process.stderr:
        print(process.stderr.strip(), file=sys.stderr)
report = {
    "scope": "offline_engineering_only",
    "python_version": sys.version.split()[0],
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "checks": results,
    "passed": all(row["exit_code"] == 0 for row in results),
    "live_model_evaluated": False,
    "real_embedding_evaluated": False,
}
destination = ROOT / "examples/offline-verification.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if report["passed"] else 1)
