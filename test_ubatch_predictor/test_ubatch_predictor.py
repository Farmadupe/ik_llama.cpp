"""Black-box tests for llama_ubatch_predictor.
"""

import dataclasses
import json
import math
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
TRACES = HERE / "traces"
DRIVER = HERE / "build" / "ubatch_predictor_driver"


# Note: this test should neither narrate its sampled data nor editorialize.
TOL_FLOOR = 0.05
K_KNOWN = 0.5
P_KNOWN = 2.0
K_NOVEL = 0.35
P_NOVEL = 0.5


def tolerance_for(row, prior):
    """Return (relative_tolerance, regime_label) for a row given all prior rows."""
    same = sum(1 for p in prior if p["batch_tokens"] == row["batch_tokens"])
    sizes = {p["batch_tokens"] for p in prior}
    if same >= 2:
        return TOL_FLOOR + K_KNOWN / same**P_KNOWN, "known"
    if len(sizes) >= 2 and len(prior) >= 3:
        return TOL_FLOOR + K_NOVEL / len(prior)**P_NOVEL, "novel"
    return math.inf, "unconditioned"


def iter_paths(tree, prefix=()):
    """Yield every maximal root-to-leaf row sequence from a trace tree.

    A tree is a flat array of entries in ubatch order. Each entry is an array whose
    first element is the ubatch row dict; any further elements are subtrees (same
    shape) recording alternative continuations from just after that row. The mainline
    is yielded unless a subtree forks off its final entry, in which case it is a
    strict prefix of that subtree's paths and replaying it would add nothing.
    """
    rows = list(prefix)
    forks = []
    for entry in tree:
        rows.append(entry[0])
        for subtree in entry[1:]:
            forks.append((len(rows), subtree))
    if not any(n == len(rows) for n, _ in forks):
        yield rows
    for n, subtree in forks:
        yield from iter_paths(subtree, rows[:n])


@dataclasses.dataclass
class Case:
    stem: str
    batch_size: int
    rows: list
    label: str
    skip_reason: str | None


def _load_cases():
    cases = []
    for path in sorted(TRACES.glob("*.json")):
        data = json.loads(path.read_text())
        groups = data.get("traces", {})
        for batch_size in sorted(groups, key=int):
            group = groups[batch_size]
            skip_reason = group.get("skip-reason")
            paths = [rows for tree in group["traces"] for rows in iter_paths(tree)]
            for path_index, rows in enumerate(paths):
                label = f"{path.stem}:{batch_size}:{path_index}"
                cases.append(Case(path.stem, int(batch_size), rows, label, skip_reason))
    return cases


CASES = _load_cases()


def run_driver(rows):
    # One cycle per row: predict this row (out of sample), then observe it.
    lines = [
        f"{r['batch_tokens']} {r['prior_tokens']} "
        f"{r['batch_tokens']} {r['prior_tokens']} {r['batch_time']}"
        for r in rows
    ]
    stdin = "".join(line + "\n" for line in lines)
    proc = subprocess.run(
        [str(DRIVER)],
        input=stdin,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"driver exited {proc.returncode}"
    preds = [float(x) for x in proc.stdout.split()]
    assert len(preds) == len(rows), f"got {len(preds)} predictions for {len(rows)} rows"
    return preds


def _finite(x):
    # JSON has no Infinity/NaN; report non-finite values as null.
    return x if math.isfinite(x) else None


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_trace(case, json_sink):
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    assert DRIVER.exists(), f"driver not built: {DRIVER} (run ./run.sh first)"
    for r in case.rows:
        assert r["batch_tokens"] <= case.batch_size, (
            f"{case.label}: ubatch of {r['batch_tokens']} exceeds batch size {case.batch_size}"
        )

    preds = run_driver(case.rows)

    records = []
    failures = []
    for i, row in enumerate(case.rows):
        tol, regime = tolerance_for(row, case.rows[:i])
        actual = row["batch_time"]
        predicted = preds[i]
        rel = abs(predicted - actual) / actual
        skipped = math.isinf(tol)
        ok = skipped or rel <= tol
        if not ok:
            failures.append(
                f"row {i} (batch={row['batch_tokens']}, prior={row['prior_tokens']}): "
                f"predicted {predicted:.3f}s vs measured {actual:.3f}s, "
                f"relative error {rel:.1%} > tolerance {tol:.1%}"
            )
        records.append(
            {
                "index": i,
                "batch_tokens": row["batch_tokens"],
                "prior_tokens": row["prior_tokens"],
                "expected": actual,
                "predicted": _finite(predicted),
                "regime": regime,
                "tolerance": None if skipped else tol,
                "lower_bound": None if skipped else actual * (1.0 - tol),
                "upper_bound": None if skipped else actual * (1.0 + tol),
                "rel_error": _finite(rel),
                "status": "skip" if skipped else ("pass" if ok else "fail"),
            }
        )

    if json_sink is not None:
        json_sink.append(
            {
                "trace": case.label,
                "batch_size": case.batch_size,
                "rows": records,
            }
        )

    assert not failures, "\n".join(failures)
