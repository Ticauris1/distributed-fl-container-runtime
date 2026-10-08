"""
Container entrypoint for distributed federated learning.

SERVICE_ROLE:
- server: runs containerization.distributed_server
- client: runs containerization.distributed_client
- single: runs main.py as the single-process research pipeline
"""

import os
import subprocess
import sys

role = os.getenv("SERVICE_ROLE", "server").lower().strip()

if role == "server":
    cmd = [sys.executable, "-m", "containerization.distributed_server"]
elif role == "client":
    cmd = [sys.executable, "-m", "containerization.distributed_client"]
elif role == "single":
    cmd = [sys.executable, os.getenv("MAIN_SCRIPT", "main.py")]
else:
    raise SystemExit(f"Unknown SERVICE_ROLE={role}")

print(f"[entrypoint] SERVICE_ROLE={role}", flush=True)
print(f"[entrypoint] running: {' '.join(cmd)}", flush=True)
sys.exit(subprocess.call(cmd))
