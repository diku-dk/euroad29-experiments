#!/usr/bin/env python3
"""Turn 'futhark bench --json' files into gnuplot data files.

One output file per benchmark program, with one row per way of computing the
Jacobian and one column per backend, giving the overhead relative to that
backend's own primal ('calculate_objective' on the same dataset).  The
baselines therefore differ between columns, which is the point: the bars show
how the cost of the Jacobian scales with the backend, not how the backends
compare to each other.

Usage: ./extract-results.py results-c.json results-multicore.json results-hip.json

The backend name is taken from the file name ('results-<backend>.json'), and
the column order follows the order of the arguments.  Output goes to 'plots'.
"""

import json
import os
import re
import statistics
import sys

OUTDIR = "plots"

# Which mode the single-mode programs use, for labelling.
MODE = {"ba": "reverse", "ht": "forward"}

# entry point -> (sort key, mode or None to look up in MODE, kind)
VARIANTS = {
    "calculate_jacobian_fwd": (0, "forward", "scalar"),
    "calculate_jacobian_fwd_vec": (1, "forward", "vector"),
    "calculate_jacobian_rev": (2, "reverse", "scalar"),
    "calculate_jacobian_rev_vec": (3, "reverse", "vector"),
    "calculate_jacobian": (0, None, "scalar"),
    "calculate_jacobian_vec": (1, None, "vector"),
}


def classify(prog, entry):
    """(sort key, label) for a Jacobian entry point, or None if not one."""
    if entry in VARIANTS:
        key, mode, kind = VARIANTS[entry]
        return key, f"{mode or MODE.get(prog, '')}\\n{kind}"
    m = re.fullmatch(r"calculate_jacobian_chunk(\d+)", entry)
    if m:
        width = int(m.group(1))
        return 4 + width, f"{MODE.get(prog, '')}\\nvector, chunk {width}"
    return None


def median_runtime(info):
    """Representative runtime in microseconds.

    The median, not the mean: the first runs of a dataset include warmup (for
    ba's Jacobian the first is 70% slower than the steady state) and the number
    of runs differs per entry point, so a mean would weight warmup differently
    in each bar.
    """
    return statistics.median(info["runtimes"])


def read(path):
    """{program: {entry: microseconds}} from one bench result file."""
    out = {}
    for key, val in json.load(open(path)).items():
        prog, entry = key.split(":", 1)
        prog = prog.removesuffix(".fut")
        times = [median_runtime(i) for i in val["datasets"].values() if "runtimes" in i]
        if times:
            # One workload per program, so there is exactly one dataset; if that
            # ever changes, the shortest is the one to compare against.
            out.setdefault(prog, {})[entry] = min(times)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    backends = [re.sub(r"^results-|\.json$", "", os.path.basename(p)) for p in sys.argv[1:]]
    data = [read(p) for p in sys.argv[1:]]

    os.makedirs(OUTDIR, exist_ok=True)
    for prog in sorted(set().union(*(d.keys() for d in data))):
        primal = [d.get(prog, {}).get("calculate_objective") for d in data]
        if None in primal:
            missing = [b for b, p in zip(backends, primal) if p is None]
            print(f"{prog}: no primal for {', '.join(missing)}, skipping", file=sys.stderr)
            continue

        rows = {}
        for i, d in enumerate(data):
            for entry, us in d.get(prog, {}).items():
                c = classify(prog, entry)
                if c:
                    rows.setdefault(c, [None] * len(backends))[i] = us

        path = os.path.join(OUTDIR, prog + ".dat")
        with open(path, "w") as f:
            print(f"# {prog}: cost of the full Jacobian, relative to the primal", file=f)
            print("#", file=f)
            for b, p in zip(backends, primal):
                print(f"# primal ({b}): {p:.0f} us", file=f)
            print("#", file=f)
            cols = " ".join(f"{b}_x" for b in backends)
            abscols = " ".join(f"{b}_us" for b in backends)
            print(f'# "method" {cols} {abscols}', file=f)
            for (_, label), times in sorted(rows.items()):
                ratios = " ".join(
                    f"{t / p:8.3f}" if t else "        -" for t, p in zip(times, primal)
                )
                absolute = " ".join(f"{t:10.0f}" if t else "         -" for t in times)
                print(f'"{label}" {ratios} {absolute}', file=f)
        print(f"{path}: {len(rows)} variants x {len(backends)} backends")


if __name__ == "__main__":
    main()
