#!/usr/bin/env python3
"""GPU thermal guard for long training runs on a laptop (native Linux or WSL2).

Every --interval seconds it reads

    nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm,memory.used

and, when the GPU reaches --pause-at (default 83 C), freezes the whole training
process group with SIGSTOP (the GPU goes idle and cools down); at --resume-at
(default 72 C) it sends SIGCONT. Freezing is safe for single-GPU PyTorch: queued
kernels finish, nothing is lost, training simply continues later.

Extras:
  --duty-cycle 0.8      additionally pause 20 % of every --duty-period (60 s):
                        a crude power cap for laptops where `nvidia-smi -pl`
                        is not allowed (most GeForce laptop GPUs and all WSL2).
  --power-limit 80      try `nvidia-smi -pl 80` once (needs root and a GPU/driver
                        that allows it; failures are logged and ignored).
  --max-pause-min 20    if the GPU idles above --resume-at for this long (hot
                        room), resume once it is at least 3 C below --pause-at.
  --simulate 60,85,80,70  fake temperature sequence (testing without a GPU).

It exits when the guarded process exits and always sends SIGCONT on the way
out, so training is never left frozen. train.sh starts it automatically.

    python stt/thermal_guard.py --pid 12345 --log exp/thermal_guard.log
    python stt/thermal_guard.py --dry-run          # just print readings
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

FIELDS = ["temperature.gpu", "power.draw", "utilization.gpu", "clocks.sm", "memory.used"]


def find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for cand in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):  # WSL2 location first
        if os.path.exists(cand):
            return cand
    return None


def _num(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


class Reader:
    def __init__(self, gpu: int, simulate: str | None):
        self.gpu = gpu
        self.sim = [float(x) for x in simulate.split(",")] if simulate else None
        self.exe = None if self.sim else find_nvidia_smi()

    def read(self) -> dict:
        if self.sim is not None:
            t = self.sim.pop(0) if len(self.sim) > 1 else self.sim[0]
            return {"temperature.gpu": t, "power.draw": None, "utilization.gpu": None,
                    "clocks.sm": None, "memory.used": None}
        if not self.exe:
            return {}
        try:
            out = subprocess.run(
                [self.exe, f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
                 "-i", str(self.gpu)],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if not out:
            return {}
        vals = [_num(v) for v in out[0].split(",")]
        return dict(zip(FIELDS, vals))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child still "exists": check /proc
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class Target:
    def __init__(self, pid: int | None, group: bool):
        self.pid = pid
        self.pgid = os.getpgid(pid) if (pid and group) else None
        self.stopped = False

    def _send(self, sig):
        if self.pid is None:
            return
        try:
            if self.pgid is not None and self.pgid != os.getpgid(0):
                os.killpg(self.pgid, sig)
            else:
                os.kill(self.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self):
        if not self.stopped:
            self._send(signal.SIGSTOP)
            self.stopped = True

    def cont(self):
        if self.stopped:
            self._send(signal.SIGCONT)
            self.stopped = False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pid", type=int, default=None, help="training process (its process group is paused)")
    p.add_argument("--no-group", action="store_true", help="signal only --pid, not its process group")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--pause-at", type=float, default=83.0)
    p.add_argument("--resume-at", type=float, default=72.0)
    p.add_argument("--max-pause-min", type=float, default=20.0)
    p.add_argument("--duty-cycle", type=float, default=1.0, help="fraction of time allowed to run (1 = off)")
    p.add_argument("--duty-period", type=float, default=60.0)
    p.add_argument("--power-limit", type=float, default=None, help="watts; tries nvidia-smi -pl")
    p.add_argument("--log", default=None, help="append log lines here (default: stdout only)")
    p.add_argument("--dry-run", action="store_true", help="print readings, never signal")
    p.add_argument("--simulate", default=None, help="comma-separated fake temperatures (testing)")
    p.add_argument("--max-polls", type=int, default=0, help="stop after N polls (testing)")
    a = p.parse_args()
    if a.resume_at >= a.pause_at:
        p.error("--resume-at must be lower than --pause-at")
    if not 0.05 <= a.duty_cycle <= 1.0:
        p.error("--duty-cycle must be in [0.05, 1]")

    logf = open(a.log, "a", buffering=1, encoding="utf-8") if a.log else None

    def log(msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")

    reader = Reader(a.gpu, a.simulate)
    if reader.sim is None and reader.exe is None:
        log("nvidia-smi not found: nothing to guard (CPU run?) -- exiting")
        return 0
    target = Target(None if a.dry_run else a.pid, group=not a.no_group)

    def bye(signum=None, _frame=None):
        target.cont()
        log(f"guard exiting ({signal.Signals(signum).name if signum else 'done'}); training left running")
        if logf:
            logf.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGHUP, bye)

    if a.power_limit and not a.dry_run and reader.exe:
        r = subprocess.run([reader.exe, "-i", str(a.gpu), "-pl", str(a.power_limit)],
                           capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip().replace("\n", " | ")
        log(f"power limit {a.power_limit} W -> exit {r.returncode}: {msg or 'ok'}"
            + ("" if r.returncode == 0 else " (ignored; on laptops/WSL2 use the vendor tool or --duty-cycle)"))

    log(f"guarding pid={a.pid} pgid={target.pgid} pause>={a.pause_at}C resume<={a.resume_at}C "
        f"interval={a.interval}s duty={a.duty_cycle}")
    paused_since = None
    thermal_paused = False
    missing = 0
    polls = 0
    t_start = time.time()
    try:
        while True:
            if a.pid and not alive(a.pid):
                log(f"pid {a.pid} has exited")
                break
            r = reader.read()
            temp = r.get("temperature.gpu")
            now = time.time()
            if temp is None:
                missing += 1
                if missing in (1, 6) or missing % 60 == 0:
                    log(f"cannot read GPU temperature ({missing}x): {r or 'nvidia-smi failed'}")
                if thermal_paused and missing >= 6:  # never stay frozen on a broken sensor
                    thermal_paused = False
                    log("resuming: no temperature readings")
            else:
                missing = 0
                if not thermal_paused and temp >= a.pause_at:
                    thermal_paused, paused_since = True, now
                    log(f"PAUSE  temp={temp:.0f}C >= {a.pause_at:.0f}C")
                elif thermal_paused:
                    long_pause = (now - paused_since) / 60 >= a.max_pause_min
                    if temp <= a.resume_at or (long_pause and temp <= a.pause_at - 3):
                        thermal_paused = False
                        log(f"RESUME temp={temp:.0f}C after {(now - paused_since) / 60:.1f} min"
                            + (" (max pause reached)" if temp > a.resume_at else ""))
            duty_paused = False
            if a.duty_cycle < 1.0:
                phase = ((now - t_start) % a.duty_period) / a.duty_period
                duty_paused = phase >= a.duty_cycle
            want_pause = thermal_paused or duty_paused
            if want_pause:
                target.stop()
            else:
                target.cont()
            def fmt(key, unit):
                v = r.get(key)
                return "NA" if v is None else f"{v:g}{unit}"

            state = ("PAUSED(thermal)" if thermal_paused else "PAUSED(duty)") if want_pause else "running"
            log(f"temp={fmt('temperature.gpu', 'C')} power={fmt('power.draw', 'W')} "
                f"util={fmt('utilization.gpu', '%')} sm={fmt('clocks.sm', 'MHz')} "
                f"mem={fmt('memory.used', 'MiB')} state={state}")
            polls += 1
            if a.max_polls and polls >= a.max_polls:
                break
            # duty cycling needs finer steps than the thermal poll
            time.sleep(min(a.interval, a.duty_period * (1 - a.duty_cycle)) if a.duty_cycle < 1 else a.interval)
    finally:
        target.cont()
        if logf and not logf.closed:
            logf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} guard stopped\n")
            logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
