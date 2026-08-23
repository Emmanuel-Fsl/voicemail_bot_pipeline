"""
universal_orchestrator.py

Sequentially run a set of scripts/commands with delays, robust logging,
and a final completion summary. Universal template:
- Works with Python or non-Python commands
- Logs stdout/stderr for each step
- Writes rotating log files and a JSON summary

Usage (basic):
  python universal_orchestrator.py

Optional:
  python universal_orchestrator.py --delay 5 --stop-on-failure
  python universal_orchestrator.py --log-dir logs --log-file orchestrator.log
  python universal_orchestrator.py --summary-dir summaries

Edit the STEPS list to point at your scripts.
"""

from __future__ import annotations
import sys, io, os
import argparse
import json
import logging
import os
import platform
import shlex
import smtplib
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

# Force stdout to use UTF-8 so Windows can print box characters like ─ and •
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("chcp 65001 >nul")
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE YOUR STEPS HERE
# Each step: {"name": "...", "cmd": <str or list[str]>, optional "cwd": "<path>", optional "shell": bool}
# Tip: to ensure you use the same Python interpreter, prefer [sys.executable, "your_script.py"]
# ─────────────────────────────────────────────────────────────────────────────
STEPS: List[Dict[str, Any]] = [
    {"name": "Load VA Pipeline", "cmd": [sys.executable, "va_pipeline.py"]},
    {"name": "Load WA Pipeline", "cmd": [sys.executable, "wa_pipeline.py"]},
    {"name": "Load Livekit Pipeline", "cmd": [sys.executable, "livekit_pipeline.py"]},
]
    # You can add non-Python tasks too, e.g. a shell command:    # {"name": "Touch heartbeat file", "cmd": "touch heartbeat.ok", "shell": True},    ]
# Default delay between steps (seconds)
DEFAULT_DELAY_S = 5

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path, log_file: str) -> Path:
    from logging.handlers import RotatingFileHandler

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear old handlers (if re-running inside interactive)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s • %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (5 MB × 5)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_path

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    name: str
    command: Union[str, List[str]]
    cwd: Optional[str]
    started_at_utc: str
    finished_at_utc: str
    duration_sec: float
    return_code: Optional[int]
    status: str                # "success" | "failed" | "skipped"
    attempts: int
    error: Optional[str] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def _to_cmd_and_shell(cmd: Union[str, List[str]], shell_flag: Optional[bool]) -> tuple[List[str] | str, bool]:
    """
    Normalize command + shell flag.
    - If cmd is a list -> run without shell (recommended).
    - If cmd is a string:
        * If shell_flag provided, honor it.
        * Else, split on POSIX using shlex except on Windows, where we pass the raw string to shell=True by default.
    """
    if isinstance(cmd, list):
        return cmd, False

    if shell_flag is True:
        return cmd, True

    if shell_flag is False:
        # Try to split the string into argv
        posix = os.name != "nt"
        return shlex.split(cmd, posix=posix), False

    # No shell_flag specified for string commands:
    if os.name == "nt":
        # Windows: easier to let CMD parse it
        return cmd, True
    else:
        # POSIX: split into argv
        return shlex.split(cmd), False

def run_step(
    name: str,
    cmd: Union[str, List[str]],
    cwd: Optional[Union[str, Path]] = None,
    shell: Optional[bool] = None,
    timeout: Optional[int] = None,
    retries: int = 0,
) -> StepResult:
    logger = logging.getLogger(f"step:{name}")
    normalized_cmd, use_shell = _to_cmd_and_shell(cmd, shell)

    attempts = 0
    start_ts = datetime.now(timezone.utc)
    start_str = start_ts.isoformat()

    stdout_tail = None
    stderr_tail = None
    last_error = None
    rc: Optional[int] = None
    cwd_str = str(cwd) if cwd else None

    while True:
        attempts += 1
        try:
            logger.info("Starting: %s", normalized_cmd if isinstance(normalized_cmd, list) else normalized_cmd)
            completed = subprocess.run(
                normalized_cmd,
                cwd=cwd_str,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            rc = completed.returncode

            # Log outputs
            if completed.stdout:
                for line in completed.stdout.splitlines():
                    logger.info("[stdout] %s", line)
                stdout_tail = completed.stdout

            if completed.stderr:
                for line in completed.stderr.splitlines():
                    logger.error("[stderr] %s", line)
                stderr_tail = completed.stderr

            if rc == 0:
                break  # success

            last_error = f"Non-zero exit code: {rc}"
            logger.error("Run attempt %d failed (rc=%s).", attempts, rc)

        except subprocess.TimeoutExpired as e:
            last_error = f"Timeout after {timeout}s"
            logger.error("Timeout (attempt %d): %s", attempts, last_error)
            rc = None  # timeout has no rc
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.exception("Exception during run (attempt %d).", attempts)

        if attempts > retries:
            break
        else:
            logger.info("Retrying (%d/%d)...", attempts, retries)

    end_ts = datetime.now(timezone.utc)
    end_str = end_ts.isoformat()
    duration = (end_ts - start_ts).total_seconds()

    status = "success" if (rc == 0) else "failed"
    return StepResult(
        name=name,
        command=cmd,
        cwd=cwd_str,
        started_at_utc=start_str,
        finished_at_utc=end_str,
        duration_sec=duration,
        return_code=rc,
        status=status,
        attempts=attempts,
        error=last_error,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def send_summary_email(results: List[StepResult]) -> None:
    smtp_host  = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT", "465"))
    smtp_user  = os.getenv("GMAIL_SENDER") or os.getenv("SMTP_USER", "")
    smtp_pass  = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("SMTP_PASS", "")
    email_from = os.getenv("EMAIL_FROM", smtp_user)
    email_to_raw = os.getenv("EMAIL_TO", "")

    logger = logging.getLogger("orchestrator.email")

    if not smtp_user or not smtp_pass or not email_to_raw:
        logger.warning("Email skipped — GMAIL_SENDER/GMAIL_APP_PASSWORD/EMAIL_TO not set")
        return

    recipients = [e.strip() for e in email_to_raw.split(",") if e.strip()]
    ok   = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status != "success")
    overall = "SUCCESS" if fail == 0 else "FAILED"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Build HTML body ───────────────────────────────────────────────────────
    status_colour = {"success": "#2e7d32", "failed": "#c62828", "skipped": "#f57c00"}

    rows = ""
    for r in results:
        colour = status_colour.get(r.status, "#555")
        rows += (
            f"<tr>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #eee'>{r.name}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #eee;"
            f"color:{colour};font-weight:bold'>{r.status.upper()}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #eee'>{r.duration_sec:.1f}s</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #eee'>{r.return_code}</td>"
            f"</tr>"
        )

    failure_detail = ""
    for r in results:
        if r.status != "success":
            stderr_block = (
                f"<pre style='background:#fbe9e7;padding:10px;border-radius:4px;"
                f"font-size:12px;overflow-x:auto;white-space:pre-wrap'>{r.stderr_tail or '(no stderr)'}</pre>"
            )
            failure_detail += (
                f"<h3 style='color:#c62828;margin-top:24px'>✗ {r.name}</h3>"
                f"<p><strong>Error:</strong> {r.error or '—'}</p>"
                f"<p><strong>Last stderr lines:</strong></p>{stderr_block}"
            )

    html = f"""
    <div style='font-family:Arial,sans-serif;max-width:700px;margin:0 auto;color:#333'>
      <div style='background:{"#2e7d32" if fail==0 else "#c62828"};color:#fff;
                  padding:16px 24px;border-radius:6px 6px 0 0'>
        <h2 style='margin:0'>FSS Pipeline — {overall}</h2>
        <p style='margin:4px 0 0;font-size:13px'>{ts} &nbsp;|&nbsp;
           {ok} succeeded &nbsp;/&nbsp; {fail} failed &nbsp;/&nbsp; {len(results)} total</p>
      </div>
      <div style='border:1px solid #ddd;border-top:none;padding:20px 24px;border-radius:0 0 6px 6px'>
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead>
            <tr style='background:#f5f5f5'>
              <th style='padding:8px 12px;text-align:left'>Step</th>
              <th style='padding:8px 12px;text-align:left'>Status</th>
              <th style='padding:8px 12px;text-align:left'>Duration</th>
              <th style='padding:8px 12px;text-align:left'>Exit code</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        {failure_detail}
      </div>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{overall}] FSS Pipeline — {ok}/{len(results)} steps passed ({ts})"
    msg["From"]    = f"FSS Orchestrator <{email_from}>"
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(email_from, recipients, msg.as_string())
        logger.info("Summary email sent to %s", ", ".join(recipients))
    except Exception as exc:
        logger.error("Failed to send summary email: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Universal sequential orchestrator.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Delay (seconds) between steps.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately if a step fails.")
    parser.add_argument("--timeout", type=int, default=None, help="Per-step timeout (seconds).")
    parser.add_argument("--retries", type=int, default=0, help="Retries per step on failure.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files.")
    parser.add_argument("--log-file", type=str, default="orchestrator.log", help="Log file name.")
    parser.add_argument("--summary-dir", type=str, default="summaries", help="Directory to write JSON summaries.")
    parser.add_argument("--test-email", action="store_true", help="Send a test summary email with fake results and exit.")

    args = parser.parse_args()

    log_path = setup_logging(Path(args.log_dir), args.log_file)
    logger = logging.getLogger("orchestrator")

    if args.test_email:
        fake = [
            StepResult("Load VA Pipeline", ["python", "va_pipeline.py"], None,
                       "2026-01-01T00:00:00Z", "2026-01-01T00:00:07Z", 7.1, 0, "success", 1),
            StepResult("Load WA Pipeline", ["python", "wa_pipeline.py"], None,
                       "2026-01-01T00:00:07Z", "2026-01-01T00:00:12Z", 5.2, 1, "failed", 1,
                       error="Non-zero exit code: 1",
                       stderr_tail="requests.exceptions.HTTPError: 400 Bad Request\nfor url: https://api.elevenlabs.io/..."),
        ]
        send_summary_email(fake)
        sys.exit(0)

    logger.info("Python: %s", sys.executable)
    logger.info("Platform: %s", platform.platform())
    logger.info("Log file: %s", log_path)

    results: List[StepResult] = []
    try:
        for i, step in enumerate(STEPS, start=1):
            name = step.get("name", f"step_{i}")
            cmd = step["cmd"]
            cwd = step.get("cwd")
            shell = step.get("shell")

            logger.info("(%d/%d) RUNNING: %s", i, len(STEPS), name)
            res = run_step(
                name=name,
                cmd=cmd,
                cwd=cwd,
                shell=shell,
                timeout=args.timeout,
                retries=args.retries,
            )
            results.append(res)

            # Inter-step delay
            if i < len(STEPS):
                if res.status == "failed" and args.stop_on_failure:
                    logger.error("Stopping after failure (per --stop-on-failure).")
                    break
                logger.info("Waiting %.1f seconds before next step...", args.delay)
                time.sleep(max(0.0, args.delay))

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Proceeding to summary...")

    # Summary
    ok = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status != "success")

    logger.info("" * 80)
    logger.info("COMPLETION SUMMARY")
    logger.info("Total steps run: %d | Success: %d | Failed: %d", len(results), ok, fail)
    for r in results:
        logger.info(
            "[%s] status=%s rc=%s duration=%.2fs attempts=%d",
            r.name, r.status, r.return_code, r.duration_sec, r.attempts
        )
    logger.info("" * 80)

    # Write JSON summary
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_dir / f"orchestration_summary_{ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in results], f, indent=2)
    logger.info("Summary written to: %s", summary_path)

    # Email summary (non-fatal)
    send_summary_email(results)

    # Exit code reflects overall success
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
