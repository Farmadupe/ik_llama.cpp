#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["unidiff"]
# ///
"""Find net-zero "churn" across a chain of jj commits: work that cancels out.

A change introduced by one commit and later undone by another never reaches the
tip, so it is absent from the cumulative ``base..tip`` diff even though it is
present in the per-commit diffs. That gap is wasted edit work -- the thing this
script surfaces, so a stack can be slimmed before a long rebase rather than after
errors pop up.

THE METHOD (and what is exact vs approximate)

For each touched file the script builds four multisets of *line content*:

  A_total / R_total  every added / removed line summed over all per-commit diffs
                     (the work that was done)
  A_net   / R_net    the added / removed lines of the single base..tip diff
                     (the work that survived)

Then ``churn_added = A_total - A_net`` (multiset difference, clamped at zero) is
the content that some commit added but that did NOT survive to the tip, and
``churn_removed = R_total - R_net`` is content some commit removed that came back
by the tip. A classic net-zero pair -- add foo() in one commit, delete it in a
later one -- lands foo()'s lines in both churn sets.

This counting is EXACT, duplicates included: the multiset difference respects
multiplicity. The one thing it cannot do precisely is *attribute* an ambiguous
duplicate line (a lone brace, a blank ``return;``) to the exact commit and spot
that produced it. So for every churned line the script reports the change ids
that added it (``added_by``) and removed it (``removed_by``) as CANDIDATES, and
leaves confirming the real partner -- which needs reading the code -- to a human
or an agent. This is a triage tool, not a prover; it deliberately does not try to
track line identity across the chain (the genuinely hard part git itself punts
on).

THREE REPORTS

  net_zero_files   files touched by some commit in the range but byte-identical
                   between base and tip: every commit that touched them cancels
                   out. EXACT (a file differs in base..tip iff it appears in that
                   diff; absence means identical content).
  files[path].churn_lines
                   per-file churn volume = surviving-blind work. A high ratio of
                   churn_lines to net_lines is the "look here first" signal.
  files[path].churn_added / churn_removed
                   the churned line contents with their candidate add/remove
                   change ids, for closing the gap by hand.

Pure-whitespace churned lines are dropped from the per-file detail and its
churn_lines count: blank-line / indentation churn is a whitespace matter, not the
wasted work this tool is after, and counting it would only inflate the headline
with noise. A lone ``}`` is not whitespace and is kept.

SCOPE / jj CONVENTIONS

This script takes a jj revset directly with -r, the way you would name any span
of commits to jj.
The work commits are exactly that set; the base is the parent of the set's root
(``(roots(R))-``) and the tip is its head (``heads(R)``). Both must resolve to a
single commit, so the range has to be linear (one root, one head); a fork or a
merge in the set is an error. Examples:

  churn_hunks.py -r 'mybookmark::@'        a bookmark up to the working copy
  churn_hunks.py -r 'base..tip'            jj's exclusive-of-base span
  churn_hunks.py                           default: mutable() & ::@ (your stack)

Change ids (not commit ids) are emitted throughout, because they survive the
rewrites that editing or reordering a commit forces onto its descendants, so they
stay valid ``jj edit`` / ``jj rebase`` targets after you act on the output.

Output is JSON on stdout. Exit code is 0 when nothing is flagged, 1 when any
net-zero file or churn is found, so it can gate a "clean before rebase" check.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict

from unidiff import PatchSet

# Files under these path prefixes are vendored / third-party and never inspected:
# they are not our work and not ours to slim.
EXCLUDED_PREFIXES = ("farmadupe/kimi_k2.5/reference_implementation/",)

# jj template emitting one tab-separated row per commit: change id, then the
# first line of the description. \\t / \\n reach jj as \t / \n for it to expand.
#
# Commit ids are deliberately NOT emitted. They are not needed for any operation
# here (every diff is taken by change id), and a commit id in the output is a
# divergence footgun: editing or reordering any commit rewrites the commit ids of
# all its descendants, so a commit id read from this report goes stale the moment
# you act on it, and reusing a stale one creates a divergent change. Change ids
# survive those rewrites and stay valid jj targets.
ROW_TEMPLATE = 'change_id.short() ++ "\\t" ++ description.first_line() ++ "\\n"'


def run(*args: str) -> str:
    """Run a command and return stdout, exiting on failure."""
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(args)}\n{proc.stderr}")
        sys.exit(2)
    return proc.stdout


def parse_rows(out: str) -> list[dict]:
    """Parse ROW_TEMPLATE output into [{change_id, description}]."""
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        change_id, _, description = line.partition("\t")
        rows.append({"change_id": change_id, "description": description})
    return rows


def resolve_single(revset: str, label: str) -> dict:
    """Resolve a revset that must name exactly one commit, or exit with guidance."""
    rows = parse_rows(run("jj", "log", "-r", revset, "--no-graph", "-T", ROW_TEMPLATE))
    if len(rows) != 1:
        sys.stderr.write(
            f"{label} revset {revset!r} resolved to {len(rows)} commits; need "
            f"exactly 1. The range must be linear (one root, one head) -- a fork "
            f"or merge in the commit set has no single base/tip to diff against.\n"
        )
        sys.exit(2)
    return rows[0]


def dedup(seq: list[str]) -> list[str]:
    """Order-preserving de-duplication of change ids."""
    seen: set[str] = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def collect_lines(diff_text: str):
    """Yield (path, line) for every added/removed line in a --git diff.

    Excluded paths and context lines are skipped; the caller decides what to do
    with each changed line. unidiff gives binary files no hunks, so they drop out.
    """
    if not diff_text.strip():
        return
    for pfile in PatchSet(diff_text):
        path = pfile.path
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        for hunk in pfile:
            for line in hunk:
                if line.is_added or line.is_removed:
                    yield path, line


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-r",
        "--revisions",
        default="mutable() & ::@",
        help="jj revset naming the work commits (default: 'mutable() & ::@', "
        "i.e. your whole mutable stack up to the working copy). The base is "
        "(roots(R))- and the tip is heads(R); both must be a single commit.",
    )
    args = parser.parse_args()

    work = parse_rows(run("jj", "log", "-r", args.revisions, "--no-graph", "-T", ROW_TEMPLATE))
    if not work:
        sys.stderr.write(f"revset {args.revisions!r} names no commits.\n")
        return 2
    work.reverse()  # jj log is newest-first; report base-most first

    base = resolve_single(f"(roots({args.revisions}))-", "base")
    tip = resolve_single(f"heads({args.revisions})", "tip")

    # Per-commit work: A_total / R_total content multisets per file, plus the
    # change ids that added / removed each line so the output can name candidates.
    a_total: dict[str, Counter] = defaultdict(Counter)
    r_total: dict[str, Counter] = defaultdict(Counter)
    added_by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    removed_by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    touched: set[str] = set()

    for commit in work:
        for path, line in collect_lines(run("jj", "diff", "-r", commit["change_id"], "--git")):
            touched.add(path)
            if line.is_added:
                a_total[path][line.value] += 1
                added_by[path][line.value].append(commit["change_id"])
            else:
                r_total[path][line.value] += 1
                removed_by[path][line.value].append(commit["change_id"])

    # Surviving work: the single base..tip diff. A file present here changed; a
    # touched file absent here is net-zero (base and tip contents are identical).
    a_net: dict[str, Counter] = defaultdict(Counter)
    r_net: dict[str, Counter] = defaultdict(Counter)
    net_files: set[str] = set()
    for path, line in collect_lines(run("jj", "diff", "--from", base["change_id"], "--to", tip["change_id"], "--git")):
        net_files.add(path)
        if line.is_added:
            a_net[path][line.value] += 1
        else:
            r_net[path][line.value] += 1

    def candidates(counter: Counter, path: str) -> list[dict]:
        """Render a churn multiset as candidate records, biggest count first.

        Each record names every change id that added the line and every one that
        removed it -- the leads for finding the real add/undo partner by reading
        the code. Pure-whitespace lines are dropped: that is a whitespace matter,
        not churn, and would only pad the count.
        """
        out = []
        for value, count in counter.most_common():
            if not value.strip():
                continue
            out.append(
                {
                    "value": value,
                    "count": count,
                    "added_by": dedup(added_by[path][value]),
                    "removed_by": dedup(removed_by[path][value]),
                }
            )
        return out

    files: dict[str, dict] = {}
    for path in touched:
        churn_added = a_total[path] - a_net[path]
        churn_removed = r_total[path] - r_net[path]
        added = candidates(churn_added, path)
        removed = candidates(churn_removed, path)
        churn_lines = sum(c["count"] for c in added) + sum(c["count"] for c in removed)
        if churn_lines == 0:
            continue
        files[path] = {
            "net_zero": path not in net_files,
            "work_lines": sum(a_total[path].values()) + sum(r_total[path].values()),
            "net_lines": sum(a_net[path].values()) + sum(r_net[path].values()),
            "churn_lines": churn_lines,
            "churn_added": added,
            "churn_removed": removed,
        }

    # Worst-churn file first, so an agent or reader triages top-down.
    files = dict(sorted(files.items(), key=lambda kv: kv[1]["churn_lines"], reverse=True))

    result = {
        "range": {"revset": args.revisions, "base": base, "tip": tip, "commits": work},
        "net_zero_files": sorted(touched - net_files),
        "files": files,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if (result["net_zero_files"] or files) else 0


if __name__ == "__main__":
    sys.exit(main())
