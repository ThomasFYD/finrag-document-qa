"""Reproduce local verification and evaluations without any paid API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = {**os.environ, "FINRAG_INTEGRATION": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_TELEMETRY": "1", "LANGSMITH_TRACING": "false"}
commands = [["-m", "pip", "check"], ["-m", "compileall", "-q", "src", "app.py", "tests"],
            ["-m", "pytest", "-q"], ["-m", "finrag", "evaluate", "--split", "dev"],
            ["-m", "finrag", "evaluate", "--split", "test"]]
if "--reindex" in sys.argv:
    commands.insert(2, ["-m", "finrag", "index"])
for command in commands:
    result = subprocess.run([sys.executable, *command], cwd=root, env=env, text=True, capture_output=True)
    print("\nRunning:", " ".join(command), flush=True)
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)
print("Local validation complete. No DeepInfra calls made.")
