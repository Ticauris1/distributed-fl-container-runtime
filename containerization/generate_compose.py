"""
Generate containerization/docker-compose.generated.yml from runtime/plan.json.

Usage:
    python -m containerization.prepare_distributed_run
    python -m containerization.generate_compose
    docker compose -f containerization/docker-compose.generated.yml up --build
"""

from __future__ import annotations

from pathlib import Path
import json
import os


CONTAINERIZATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONTAINERIZATION_DIR.parent


def main():
    runtime_dir = Path(os.getenv("RUNTIME_DIR", str(PROJECT_ROOT / "runtime")))
    plan_path = runtime_dir / "plan.json"
    if not plan_path.exists():
        raise RuntimeError(
            f"{plan_path} not found. Run "
            "python -m containerization.prepare_distributed_run first."
        )

    plan = json.loads(plan_path.read_text())
    n_clients = int(plan["n_clients"])

    lines = []
    lines.append("services:")
    lines.append("  fed-server:")
    lines.append("    build:")
    lines.append("      context: ..")
    lines.append("      dockerfile: containerization/Dockerfile")
    lines.append("    container_name: fed-server")
    lines.append("    environment:")
    lines.append("      SERVICE_ROLE: server")
    lines.append("      RUNTIME_DIR: /app/runtime")
    lines.append("      FED_SERVER_HOST: 0.0.0.0")
    lines.append("      FED_SERVER_PORT: 8000")
    lines.append("    ports:")
    lines.append('      - "8000:8000"')
    lines.append("    volumes:")
    lines.append("      - ../runtime:/app/runtime")
    lines.append("      - ../data:/app/data:ro")
    lines.append("      - ../models:/app/models:ro")
    lines.append("      - ../results/server:/app/results")
    lines.append("    networks:")
    lines.append("      - fl-secure-net")
    lines.append("")

    for i in range(1, n_clients + 1):
        lines.append(f"  fed-client-{i}:")
        lines.append("    build:")
        lines.append("      context: ..")
        lines.append("      dockerfile: containerization/Dockerfile")
        lines.append(f"    container_name: fed-client-{i}")
        lines.append("    depends_on:")
        lines.append("      - fed-server")
        lines.append("    environment:")
        lines.append("      SERVICE_ROLE: client")
        lines.append(f"      CLIENT_ID: {i}")
        lines.append("      RUNTIME_DIR: /app/runtime")
        lines.append("      FED_SERVER_URL: http://fed-server:8000")
        lines.append("    volumes:")
        lines.append("      - ../runtime:/app/runtime")
        lines.append("      - ../data:/app/data:ro")
        lines.append("      - ../models:/app/models:ro")
        lines.append(f"      - ../results/client_{i}:/app/results")
        lines.append("    networks:")
        lines.append("      - fl-secure-net")
        lines.append("")

    lines.append("networks:")
    lines.append("  fl-secure-net:")
    lines.append("    driver: bridge")

    out = CONTAINERIZATION_DIR / "docker-compose.generated.yml"
    out.write_text("\n".join(lines) + "\n")
    print(f"[compose] wrote {out} with {n_clients} client containers")


if __name__ == "__main__":
    main()
