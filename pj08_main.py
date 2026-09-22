#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pj08_main.py Parallelized pipeline runner for pj08 (70 cores / 100 GB)

STEPS (now 11):
  [1]  [Updated 250919]  pj08_T_ci_summary.py        (merged T and summary)
  [2]  [Updated 250917]  pj08_Amount_outliers.py     (Calculate Z)
  [3]  [Updated 250917]  pj08_Sensitivity.py         (Sensitivity Tests for Z Thresholds)
  [4]  [Updated 250917]  pj08_Z_summary.py           (Z summary and t test)
  [5]  [Updated 250917]  pj08_fig_outliers.py        (9 cases fig for outliers)
  [6]  [Updated 250917]  pj08_constant_T_c_i.py      (Cases without outliers)
  [7]  [Updated 250917]  pj08_timegap.py             (timegap summary)
  [8]  [Updated 250917]  pj08_timegap_fig.py         (timegap fig V1 V2, and top 100)
  [9]  [Updated 250917]  pj08_ft.py                  (Distribution of the first time T exceeds the threshold)
  [10] [Need Update]     pj08_amount_map.py          (Regional heat map)
  [11] [Updated 250919]  pj08_regression.py          (LASSO, OLS, RF)
  [12] [Need Update]     pj08_regression_fig.py

Usage examples:
  python3 pj08_main.py
  python3 pj08_main.py --only timegap
  python3 pj08_main.py --start-from zsummary --end-at ft
  python3 pj08_main.py --dry-run

Parallel execution plan (batch-by-batch):
  Batch A:  [1]                         (must finish first)
  Batch B:  [2]  +  [amount_map]        (both depend on [1])
  Batch C:  [3][4][5][6][7][8][9]       (all depend on [2]; some also use [1] outputs)
  Batch D:  [regression]                (always runs after all previous batches)

Rationale / file-level dependencies:
  - [2] writes Z & rho used by [3][4][5][6][7][8][9] (core fan-out). [Ref: outliers]
  - [3] reads Z only (sensitivity on thresholds). [Ref: sensitivity]
  - [4] reads Z + rho (summary + t-tests). [Ref: z_summary]
  - [5] reads Z + T_c_i + rho for cases; also country/IPCR meta. [Ref: fig_outliers]
  - [6] reads T_c_i + rho, outputs WWP_rho_c_i_mid.csv. [Ref: constant_T_c_i]
  - [7] reads Z and name dictionaries, outputs multiple CSVs + histogram. [Ref: timegap]
  - [8] reads Z_all/Z_delta/Z_y + T_c_i + rho, outputs time-gap figs. [Ref: timegap_fig]
  - [9] reads Amount + Z to detect pioneers & leadtime and plot distribution. [Ref: ft]
  - [10] only needs [1]; can run alongside [2]. [Ref: amount_map]
  - [11] placed last per request; not required by earlier steps but scheduled after them by design. [Ref: regression]

Notes:
  Transition-aware logic for SU/CS/YU appears in several scripts ([4],[7],[9]).
  To mitigate CPU oversubscription from internal BLAS/NumPy/MP, we set:
      OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
    You can tune --max-parallel to balance CPU/Memory utilization.
"""

# ==== Imports ====
import argparse
import datetime as dt
import os
import sys
import subprocess
import shlex
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== Paths ====
PJ_DIR = "/data01/rong_dataset/Code/pj08"
LOG_DIR = os.path.join(PJ_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ==== Steps ====
STEPS = {
    "t_ci_summary":   ("T_c_i summary",              "pj08_T_ci_summary.py"),     # [1]
    "zscore":         ("Z-score compute",            "pj08_Amount_outliers.py"),  # [2]
    "sensitivity":    ("Sensitivity analysis",       "pj08_Sensitivity.py"),      # [3]
    "zsummary":       ("Z summary",                  "pj08_Z_summary.py"),        # [4]
    "fig_outliers":   ("Outlier figures",            "pj08_fig_outliers.py"),     # [5]
    "constant_tci":   ("Constant T_c_i filter",      "pj08_constant_T_c_i.py"),   # [6]
    "timegap":        ("Time-gap stats",             "pj08_timegap.py"),          # [7]
    "timegap_fig":    ("Time-gap figures",           "pj08_timegap_fig.py"),      # [8]
    "ft":             ("Ft distribution figs",       "pj08_ft.py"),               # [9]
    "amount_map":     ("Amount maps",                "pj08_amount_map.py"),       # [10]
    "regression":     ("Regression analysis",        "pj08_regression.py"),       # [11]
}

# Batches for parallel scheduling
BATCH_A = ["t_ci_summary"]                       # must run first
BATCH_B = ["zscore", "amount_map"]               # depend on A
BATCH_C = ["sensitivity", "zsummary", "fig_outliers",
           "constant_tci", "timegap", "timegap_fig", "ft"]  # depend on zscore
BATCH_D = ["regression"]                         # must run after all previous batches

# Default parallelism cap per batch (tunable)
DEFAULT_MAX_PARALLEL = 3

# ==== Utils ====
def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def python_bin() -> str:
    return shlex.quote(sys.executable or "python3")

def step_log_files(step_id: str) -> Tuple[str, str]:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    per_step = os.path.join(LOG_DIR, f"{ts}_{step_id}.log")
    global_log = os.path.join(LOG_DIR, "pipeline_all.log")
    return per_step, global_log

def append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def check_scripts_exist() -> None:
    missing = []
    for step_id, (_, fname) in STEPS.items():
        full = os.path.join(PJ_DIR, fname)
        if not os.path.isfile(full):
            missing.append(full)
    if missing:
        lines = ["Missing required script(s):"] + [f"  - {m}" for m in missing]
        raise FileNotFoundError("\n".join(lines))

def make_env_for_subproc(user_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    # reduce BLAS/NumPy thread oversubscription when we run multiple scripts in parallel
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if user_env:
        env.update(user_env)
    return env

def launch_step(step_id: str, dry_run: bool = False, extra_env: Optional[Dict[str, str]] = None) -> int:
    human, filename = STEPS[step_id]
    script_path = os.path.join(PJ_DIR, filename)
    per_log, global_log = step_log_files(step_id)

    header = f"[{now_str()}] >>> START STEP: {step_id} | {human} | {script_path}"
    print(header); append_line(global_log, header); append_line(per_log, header)

    cmd = f"{python_bin()} {shlex.quote(script_path)}"
    cmd_line = f"[{now_str()}] CMD: {cmd}"
    print(cmd_line); append_line(global_log, cmd_line); append_line(per_log, cmd_line)

    if dry_run:
        msg = f"[{now_str()}] DRY-RUN: Skipped execution."
        print(msg); append_line(global_log, msg); append_line(per_log, msg)
        trailer = f"[{now_str()}] <<< END STEP: {step_id} (dry-run)\n"
        print(trailer); append_line(global_log, trailer); append_line(per_log, trailer)
        return 0

    # Stream output to per-step log (avoid interleaved stdout when parallel)
    env = make_env_for_subproc(extra_env)
    with open(per_log, "a", encoding="utf-8") as fout:
        process = subprocess.Popen(
            cmd, cwd=PJ_DIR, shell=True,
            stdout=fout, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env
        )
        code = process.wait()

    if code == 0:
        ok = f"[{now_str()}] STEP OK: {step_id}"
        print(ok); append_line(global_log, ok); append_line(per_log, ok)
    else:
        err = f"[{now_str()}] STEP FAILED: {step_id} (exit code {code})"
        print(err); append_line(global_log, err); append_line(per_log, err)

    trailer = f"[{now_str()}] <<< END STEP: {step_id}\n"
    print(trailer); append_line(global_log, trailer); append_line(per_log, trailer)
    return code

def run_batch_parallel(step_ids: List[str], max_parallel: int, dry_run: bool) -> int:
    # simple parallel launcher with bounded concurrency
    if not step_ids:
        return 0
    max_parallel = max(1, int(max_parallel))
    print(f"[{now_str()}] Running batch in parallel (max={max_parallel}): {step_ids}")
    codes = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fut2id = {ex.submit(launch_step, sid, dry_run, None): sid for sid in step_ids}
        for fut in as_completed(fut2id):
            sid = fut2id[fut]
            try:
                rc = fut.result()
            except Exception as e:
                print(f"[{now_str()}] STEP CRASH: {sid}: {e}")
                return 99
            codes[sid] = rc
            if rc != 0:
                print(f"[{now_str()}] Early stop due to failure in {sid}.")
                return rc
    return 0

# ==== CLI ====
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pj08 pipeline (parallel)")
    p.add_argument("--start-from", choices=list(STEPS.keys()), default=None,
                   help="Start from this step (default: full pipeline)")
    p.add_argument("--end-at", choices=list(STEPS.keys()), default=None,
                   help="End at this step (default: last)")
    p.add_argument("--only", choices=list(STEPS.keys()), default=None,
                   help="Run only this step (overrides start/end & batches)")
    p.add_argument("--dry-run", action="store_true", help="Print plan but do not execute")
    p.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                   help=f"Max concurrent scripts per batch (default {DEFAULT_MAX_PARALLEL})")
    return p.parse_args()

def slice_plan(full_order: List[str], start_from: Optional[str], end_at: Optional[str], only: Optional[str]) -> List[str]:
    if only:
        return [only]
    if start_from:
        if start_from not in full_order: raise ValueError("Invalid --start-from")
        start_idx = full_order.index(start_from)
    else:
        start_idx = 0
    if end_at:
        if end_at not in full_order: raise ValueError("Invalid --end-at")
        end_idx = full_order.index(end_at)
    else:
        end_idx = len(full_order) - 1
    if start_idx > end_idx:
        raise ValueError("start-from appears after end-at")
    return full_order[start_idx:end_idx+1]

def plan_batches(subset: List[str]) -> List[List[str]]:
    """
    Return list of batches honoring dependencies:
      A: [1]
      B: [2], [amount_map]
      C: [3..9] (depend on [2])
      D: [regression] (always last)
    Subset slicing respected.
    """
    a = [s for s in BATCH_A if s in subset]
    b = [s for s in BATCH_B if s in subset]
    c = [s for s in BATCH_C if s in subset]
    d = [s for s in BATCH_D if s in subset]
    # Enforce dependencies: if zscore not in subset, drop C
    if "zscore" not in subset:
        c = []
    # If t_ci_summary not in subset, drop b's amount_map (it needs [1])
    if "t_ci_summary" not in subset:
        b = [s for s in b if s != "amount_map"]
    # D is deliberately placed last, independent of earlier content needs
    return [x for x in [a, b, c, d] if x]

def main() -> int:
    print(f"[{now_str()}] pj08_main.py starting...")
    print(f"[{now_str()}] Project dir: {PJ_DIR}")
    print(f"[{now_str()}] Logs dir:    {LOG_DIR}")

    try:
        check_scripts_exist()
    except Exception as e:
        print(f"[{now_str()}] Preflight error: {e}")
        return 2

    args = parse_args()

    # Full logical order (for slicing only)
    full_order = BATCH_A + BATCH_B + BATCH_C + BATCH_D
    try:
        subset = slice_plan(full_order, args.start_from, args.end_at, args.only)
    except Exception as e:
        print(f"[{now_str()}] Arg error: {e}")
        return 2

    batches = plan_batches(subset)
    print(f"[{now_str()}] Execution batches (in order): {batches}")
    if args.dry_run:
        print(f"[{now_str()}] DRY-RUN: no execution")
        return 0

    # Run each batch: A (seq), B (parallel), C (parallel), D (seq by size=1)
    for i, batch in enumerate(batches, start=1):
        rc = run_batch_parallel(batch, args.max_parallel, args.dry_run)
        if rc != 0:
            print(f"[{now_str()}] Pipeline stopped due to failure in batch {i}.")
            return rc

    print(f"[{now_str()}] Pipeline finished successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
