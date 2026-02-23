#!/usr/bin/env python3
"""PuffinZipAI Pre-Flight Metrics Check
========================================
Attaches to a running WebUI server (default http://127.0.0.1:5001),
optionally starts a short training run, and monitors all metric channels
for correctness.

Checks
------
1. Server reachability & version
2. System limits detection
3. Starts a short training run (configurable pop/gens)
4. Polls /api/metrics, /api/status, /api/population, /api/logs every N seconds
5. Validates:
   - Fitness values increase over time (non-degenerate)
   - Compression ratio is non-zero after a few generations
   - Population count matches requested size
   - Benchmark size grows with complexity
   - No ERROR-level log entries (warnings are OK)
   - Generation counter advances monotonically
6. Prints a colour-coded PASS/WARN/FAIL summary

Usage
-----
  # With WebUI already running on port 5001:
  python scripts/preflight_metrics_check.py

  # Custom host/port, skip auto-start, watch for 10 generations:
  python scripts/preflight_metrics_check.py --host 127.0.0.1 --port 5001 \\
      --no-start --watch-gens 10

  # Start training with custom params:
  python scripts/preflight_metrics_check.py --pop 30 --gens 15 --batch 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required.  pip install requests")
    sys.exit(1)


# ── ANSI colours (Windows 10+ and all Unix terminals) ──────────────────────
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"


def _ok(msg: str) -> str:
    return f"{_C.GREEN}[PASS]{_C.RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{_C.YELLOW}[WARN]{_C.RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{_C.RED}[FAIL]{_C.RESET} {msg}"


def _info(msg: str) -> str:
    return f"{_C.CYAN}[INFO]{_C.RESET} {msg}"


def _dim(msg: str) -> str:
    return f"{_C.DIM}{msg}{_C.RESET}"


# ── Result accumulator ─────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    detail: str = ""


@dataclass
class PreflightState:
    base_url: str = ""
    results: list[CheckResult] = field(default_factory=list)
    metrics_snapshots: list[dict[str, Any]] = field(default_factory=list)
    population_snapshots: list[Any] = field(default_factory=list)
    log_entries: list[dict[str, str]] = field(default_factory=list)
    status_snapshots: list[Any] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = ""):
        self.results.append(CheckResult(name, status, detail))
        fmt = {
            "PASS": _ok,
            "WARN": _warn,
            "FAIL": _fail,
        }.get(status, _info)
        line = fmt(name)
        if detail:
            line += f"  {_dim(detail)}"
        print(line)


# ── HTTP helpers ────────────────────────────────────────────────────────────
def _get(url: str, timeout: float = 10.0) -> Any:
    """GET JSON from *url*.  Returns parsed JSON (dict/list) or ``None``."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(url: str, payload: dict[str, Any] | None = None,
          timeout: float = 15.0) -> Any:
    """POST JSON to *url*.  Returns parsed JSON or ``None``."""
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Check steps ─────────────────────────────────────────────────────────────
def check_server_reachable(st: PreflightState):
    data = _get(f"{st.base_url}/api/status")
    if data is None:
        st.add("Server reachable", "FAIL",
               f"Cannot reach {st.base_url}/api/status")
        return False
    st.add("Server reachable", "PASS", f"{st.base_url}")
    return True


def check_system_limits(st: PreflightState):
    data = _get(f"{st.base_url}/api/status")
    if not isinstance(data, dict) or "system_limits" not in data:
        st.add("System limits", "WARN", "No system_limits in /api/status")
        return
    lim = data["system_limits"]
    st.add("System limits", "PASS",
           f"max_pop={lim.get('max_pop')}, max_gens={lim.get('max_gens')}, "
           f"default_pop={lim.get('default_pop')}, default_gens={lim.get('default_gens')}")


def start_training(st: PreflightState, pop: int, gens: int,
                   batch: int, infinite: bool) -> bool:
    payload = {
        "population_size": pop,
        "num_generations": gens,
        "batch_size": batch,
        "infinite": infinite,
    }
    resp = _post(f"{st.base_url}/api/training/start", payload)
    if isinstance(resp, dict) and resp.get("success"):
        st.add("Training started", "PASS",
               f"pop={pop}, gens={gens}, batch={batch}, "
               f"infinite={infinite}")
        return True
    reason = (resp.get("error", "unknown")
              if isinstance(resp, dict) else "no response")
    st.add("Training started", "FAIL", reason)
    return False


def poll_metrics(st: PreflightState) -> dict[str, Any] | None:
    data = _get(f"{st.base_url}/api/metrics")
    if isinstance(data, dict):
        st.metrics_snapshots.append(data)
        return data
    return None


def poll_population(st: PreflightState) -> Any:
    data = _get(f"{st.base_url}/api/population")
    if data is not None:
        st.population_snapshots.append(data)
    return data


def poll_status(st: PreflightState) -> Any:
    data = _get(f"{st.base_url}/api/status")
    if data is not None:
        st.status_snapshots.append(data)
    return data


def drain_logs(st: PreflightState) -> list[dict[str, str]]:
    data = _get(f"{st.base_url}/api/logs")
    if isinstance(data, list):
        st.log_entries.extend(data)
        return data
    return []


# ── Analysis checks (post-monitoring) ──────────────────────────────────────
def analyse_fitness_progression(st: PreflightState):
    """Check that fitness improved at least once during the run."""
    history = []
    for snap in st.metrics_snapshots:
        h = snap.get("history", [])
        for pair in h:
            if len(pair) == 2:
                history.append((pair[0], pair[1]))

    if len(history) < 2:
        st.add("Fitness progression", "WARN",
               f"Only {len(history)} data point(s) — too few to evaluate")
        return

    # Deduplicate by generation
    seen_gens: set[int] = set()
    unique: list[tuple[int, float]] = []
    for gen, fit in history:
        if gen not in seen_gens:
            seen_gens.add(gen)
            unique.append((gen, fit))
    unique.sort()

    first_fit = unique[0][1]
    last_fit = unique[-1][1]
    max_fit = max(f for _, f in unique)

    if max_fit > first_fit:
        st.add("Fitness progression", "PASS",
               f"first={first_fit:.4f} → best={max_fit:.4f} "
               f"(+{max_fit - first_fit:.4f} over {len(unique)} gens)")
    elif last_fit == first_fit:
        st.add("Fitness progression", "WARN",
               f"Fitness stayed flat at {first_fit:.4f} over {len(unique)} gens")
    else:
        st.add("Fitness progression", "WARN",
               f"Fitness did not improve: first={first_fit:.4f}, "
               f"last={last_fit:.4f}")


def analyse_compression_ratio(st: PreflightState):
    """Check the compression ratio became non-zero."""
    ratios = []
    for snap in st.metrics_snapshots:
        r = snap.get("compression_ratio")
        if r is not None:
            ratios.append(float(r))

    if not ratios:
        st.add("Compression ratio", "WARN", "No ratio data collected")
        return

    non_zero = [r for r in ratios if abs(r) > 0.001]
    last_ratio = ratios[-1]

    if non_zero:
        st.add("Compression ratio", "PASS",
               f"Non-zero in {len(non_zero)}/{len(ratios)} samples, "
               f"latest={last_ratio:.2f}%")
    else:
        st.add("Compression ratio", "FAIL",
               f"Ratio stayed 0.00% across {len(ratios)} samples — "
               f"the byte-level tracking fix may not be active")


def analyse_generation_monotonic(st: PreflightState):
    """Check that the generation counter only increases."""
    gens = []
    for snap in st.metrics_snapshots:
        g = snap.get("generation")
        if g is not None:
            gens.append(int(g))

    if len(gens) < 2:
        st.add("Generation monotonic", "WARN",
               f"Only {len(gens)} sample(s)")
        return

    violations = 0
    for i in range(1, len(gens)):
        if gens[i] < gens[i - 1]:
            violations += 1

    if violations == 0:
        st.add("Generation monotonic", "PASS",
               f"Monotonic across {len(gens)} samples "
               f"(gen {gens[0]} → {gens[-1]})")
    else:
        st.add("Generation monotonic", "FAIL",
               f"{violations} backward jump(s) in {len(gens)} samples")


def analyse_population_count(st: PreflightState, expected_pop: int):
    """Check population API returns the expected number of agents."""
    if not st.population_snapshots:
        st.add("Population count", "WARN", "No population data collected")
        return

    last = st.population_snapshots[-1]
    pop = last.get("population", []) if isinstance(last, dict) else []
    n = len(pop)

    # WebUI caps at 50, so expected is min(expected_pop, 50)
    cap = min(expected_pop, 50)
    if n == cap:
        st.add("Population count", "PASS",
               f"{n} agents (expected {cap})")
    elif n > 0:
        st.add("Population count", "WARN",
               f"{n} agents (expected {cap}) — could be still initialising")
    else:
        st.add("Population count", "FAIL", "Empty population returned")


def analyse_benchmark_size(st: PreflightState):
    """Check benchmark size is non-trivial."""
    sizes = []
    for snap in st.metrics_snapshots:
        s = snap.get("benchmark_size", "0")
        if isinstance(s, str):
            try:
                s = float(s.replace(" MB", "").strip())
            except ValueError:
                continue
        sizes.append(float(s))

    if not sizes:
        st.add("Benchmark size", "WARN", "No benchmark_size data")
        return

    last = sizes[-1]
    if last > 0.001:
        st.add("Benchmark size", "PASS", f"Latest: {last:.2f} MB")
    else:
        st.add("Benchmark size", "WARN",
               f"Benchmark size is {last:.4f} MB — very small")


def analyse_logs_for_errors(st: PreflightState):
    """Flag any ERROR-level log entries."""
    errors = [e for e in st.log_entries
              if e.get("level", "").upper() == "ERROR"]
    if errors:
        sample = errors[0].get("message", "")[:120]
        st.add("Log errors", "WARN",
               f"{len(errors)} ERROR log(s). First: {sample}")
    else:
        st.add("Log errors", "PASS",
               f"No ERROR entries in {len(st.log_entries)} log lines")


def analyse_complexity_tier(st: PreflightState):
    """Check complexity tier is reported."""
    tiers = set()
    for snap in st.metrics_snapshots:
        t = snap.get("complexity_tier", "UNKNOWN")
        if t and t != "UNKNOWN":
            tiers.add(t)

    if tiers:
        st.add("Complexity tier", "PASS",
               f"Tiers seen: {', '.join(sorted(tiers))}")
    else:
        st.add("Complexity tier", "WARN", "Only 'UNKNOWN' tier reported")


# ── Main loop ──────────────────────────────────────────────────────────────
def run_preflight(args: argparse.Namespace):
    st = PreflightState(base_url=f"http://{args.host}:{args.port}")

    print()
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}  PuffinZipAI Pre-Flight Metrics Check{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"  Target: {st.base_url}")
    print(f"  Watch:  {args.watch_gens} generations, poll every {args.poll_sec}s")
    print(f"{'=' * 60}")
    print()

    # Step 1: Server reachable
    if not check_server_reachable(st):
        print(f"\n{_fail('Aborting — server not reachable.')}")
        return st

    # Step 2: System limits
    check_system_limits(st)

    # Step 3: Check if already training
    status = _get(f"{st.base_url}/api/status")
    if not isinstance(status, dict):
        status = {}
    already_training = status.get("is_training", False)

    if already_training:
        print(_info("Training is already in progress — attaching to monitor."))
    elif not args.no_start:
        print()
        ok = start_training(st, args.pop, args.gens, args.batch,
                            args.infinite)
        if not ok:
            return st
        # Give the optimizer a moment to initialise
        print(_info("Waiting 5s for optimizer init..."))
        time.sleep(5)
    else:
        print(_info("--no-start set; will only monitor existing run."))

    # Step 4: Poll loop
    print()
    print(f"{_C.BOLD}--- Monitoring (target: {args.watch_gens} gens) ---{_C.RESET}")
    last_gen = -1
    gens_seen = 0
    start_time = time.time()
    timeout_sec = args.timeout
    stall_polls = 0
    max_stall_polls = 30  # ~30 polls with no gen change before giving up

    while True:
        elapsed = time.time() - start_time
        if timeout_sec > 0 and elapsed > timeout_sec:
            print(_warn(f"Timeout ({timeout_sec}s) reached."))
            break

        m = poll_metrics(st)
        poll_status(st)
        drain_logs(st)

        if m:
            cur_gen = m.get("generation", 0) or 0
            cur_fit = m.get("best_fitness", 0.0)
            cur_ratio = m.get("compression_ratio", 0.0)
            bench_size = m.get("benchmark_size", "?")
            tier = m.get("complexity_tier", "?")

            if cur_gen != last_gen and cur_gen > 0:
                gens_seen += 1
                stall_polls = 0
                print(
                    f"  Gen {cur_gen:>4d}  |  "
                    f"Fit: {cur_fit:>8.4f}  |  "
                    f"Ratio: {cur_ratio:>6.2f}%  |  "
                    f"Bench: {bench_size}  |  "
                    f"Tier: {tier}"
                )
                last_gen = cur_gen

                # Poll population once per gen
                poll_population(st)
            else:
                stall_polls += 1

        # Check if training ended
        s = _get(f"{st.base_url}/api/status")
        if isinstance(s, dict) and not s.get("is_training", True) and gens_seen > 0:
            print(_info("Training completed."))
            break

        if gens_seen >= args.watch_gens:
            print(_info(f"Watched {gens_seen} generations — stopping monitor."))
            break

        if stall_polls >= max_stall_polls:
            print(_warn(f"No new generation in {max_stall_polls} polls — "
                        f"assuming stalled or finished."))
            break

        time.sleep(args.poll_sec)

    # Final data drain
    drain_logs(st)
    poll_metrics(st)
    poll_population(st)

    # Step 5: Analysis
    print()
    print(f"{_C.BOLD}--- Analysis ---{_C.RESET}")
    analyse_fitness_progression(st)
    analyse_compression_ratio(st)
    analyse_generation_monotonic(st)
    analyse_population_count(st, args.pop)
    analyse_benchmark_size(st)
    analyse_complexity_tier(st)
    analyse_logs_for_errors(st)

    # Step 6: Summary
    print()
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    passes = sum(1 for r in st.results if r.status == "PASS")
    warns = sum(1 for r in st.results if r.status == "WARN")
    fails = sum(1 for r in st.results if r.status == "FAIL")
    total = len(st.results)

    summary_color = _C.GREEN if fails == 0 else _C.RED
    print(
        f"{_C.BOLD}  SUMMARY: "
        f"{_C.GREEN}{passes} PASS{_C.RESET}{_C.BOLD}, "
        f"{_C.YELLOW}{warns} WARN{_C.RESET}{_C.BOLD}, "
        f"{summary_color}{fails} FAIL{_C.RESET}{_C.BOLD}  "
        f"({total} checks){_C.RESET}"
    )

    if fails == 0 and warns == 0:
        print(f"  {_C.GREEN}{_C.BOLD}All systems nominal.{_C.RESET}")
    elif fails == 0:
        print(f"  {_C.YELLOW}Minor warnings — review above.{_C.RESET}")
    else:
        print(f"  {_C.RED}Failures detected — investigate before production.{_C.RESET}")

    print(f"{'=' * 60}\n")

    # Dump raw metrics to JSON for later inspection
    if args.dump:
        dump_path = args.dump
        dump_data = {
            "check_results": [
                {"name": r.name, "status": r.status, "detail": r.detail}
                for r in st.results
            ],
            "metrics_snapshots_count": len(st.metrics_snapshots),
            "last_metrics": st.metrics_snapshots[-1] if st.metrics_snapshots else None,
            "population_snapshots_count": len(st.population_snapshots),
            "last_population": st.population_snapshots[-1] if st.population_snapshots else None,
            "log_entries_count": len(st.log_entries),
            "error_logs": [e for e in st.log_entries
                           if e.get("level", "").upper() == "ERROR"],
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, default=str)
        print(_info(f"Raw results dumped to {dump_path}"))

    return st


# ── CLI entrypoint ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PuffinZipAI Pre-Flight Metrics Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="WebUI host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001,
                        help="WebUI port (default: 5001)")
    parser.add_argument("--pop", type=int, default=30,
                        help="Population size for test run (default: 30)")
    parser.add_argument("--gens", type=int, default=10,
                        help="Generations for test run (default: 10)")
    parser.add_argument("--batch", type=int, default=5,
                        help="Population batch size (default: 5)")
    parser.add_argument("--infinite", action="store_true",
                        help="Start in infinite mode (stop manually)")
    parser.add_argument("--no-start", action="store_true",
                        help="Don't start training; only attach and monitor")
    parser.add_argument("--watch-gens", type=int, default=8,
                        help="Number of generations to watch (default: 8)")
    parser.add_argument("--poll-sec", type=float, default=3.0,
                        help="Seconds between polls (default: 3.0)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Max seconds to monitor (default: 600)")
    parser.add_argument("--dump", type=str, default="",
                        help="Dump results to this JSON file")

    args = parser.parse_args()
    st = run_preflight(args)
    sys.exit(1 if any(r.status == "FAIL" for r in st.results) else 0)


if __name__ == "__main__":
    main()
