"""Resume interrupted disjoint GPU workers from their verified chunk markers.

Three failures per process in one hour pause automatic retries. The supervisor
manages only the seven active disjoint workers and their merger.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from artifacts import write_json


def alive(pid, expected):
    """Accept only a non-zombie process running our expected Python script."""
    proc = Path("/proc", str(pid))
    return (proc.exists() and proc.joinpath("stat").read_text().split()[2] != "Z"
            and expected in proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode())


def run_directory(base):
    """Find a worker's newest run with its immutable manifest."""
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    return runs[-1] if runs else None


def restart(project, base, metadata, command, is_merger):
    """Launch a replacement and publish its new PID and log path atomically."""
    stamp = datetime.now(timezone.utc).strftime("restart-%Y%m%dT%H%M%S.%fZ")
    log = base / (stamp + ".log")
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=project, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    details = metadata | {"pid": process.pid, "log": str(log), "command": command,
                          "restart_time_utc": datetime.now(timezone.utc).isoformat()}
    write_json(base / (stamp + ".json"), details)
    write_json(base / ("active_merge.json" if is_merger else "active_launch.json"), details)
    return details


def main():
    """Watch source runs until the merged run has its final complete marker."""
    project = Path(__file__).resolve().parents[2]
    root = project / "runs/pretrain_disjoint"
    merged = project / "runs/pretrain_disjoint_merged"
    python = str(project.parent / ".venv/bin/python")
    retries = {}
    while True:
        state = {"time_utc": datetime.now(timezone.utc).isoformat(), "workers": {}}
        for gpu in range(7):
            key, base = f"gpu{gpu}", root / f"gpu{gpu}"
            metadata = json.loads((base / "active_launch.json").read_text())
            current = run_directory(base)
            complete = current is not None and (current / "complete.json").exists()
            running = alive(metadata["pid"], "benchmarks/pretrain/run.py")
            attempts = [tick for tick in retries.get(key, []) if time.monotonic() - tick < 3600]
            if not complete and not running and len(attempts) < 3:
                command = ([python, "-u", "benchmarks/pretrain/run.py", "--resume", str(current)]
                           if current else [python, "-u", "benchmarks/pretrain/run.py", "--config",
                                            f"benchmarks/pretrain/config_disjoint/gpu{gpu}.yaml"])
                metadata = restart(project, base, metadata, command, False)
                attempts.append(time.monotonic())
                running = True
            retries[key] = attempts
            state["workers"][key] = {"pid": metadata["pid"], "alive": running,
                                     "complete": complete, "restarts_last_hour": len(attempts),
                                     "blocked": not complete and not running and len(attempts) >= 3}
        metadata = json.loads((merged / "active_merge.json").read_text())
        complete = (merged / "complete.json").exists()
        running = alive(metadata["pid"], "benchmarks/pretrain/merge_disjoint.py")
        key = "merger"
        attempts = [tick for tick in retries.get(key, []) if time.monotonic() - tick < 3600]
        if not complete and not running and len(attempts) < 3:
            metadata = restart(project, merged, metadata,
                               [python, "-u", "benchmarks/pretrain/merge_disjoint.py", "--watch", "--interval", "30"], True)
            attempts.append(time.monotonic())
            running = True
        retries[key] = attempts
        state["merger"] = {"pid": metadata["pid"], "alive": running,
                           "complete": complete, "restarts_last_hour": len(attempts),
                           "blocked": not complete and not running and len(attempts) >= 3}
        write_json(merged / "supervisor_status.json", state)
        if complete:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
