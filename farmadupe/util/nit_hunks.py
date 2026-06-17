#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["unidiff"]
# ///
"""Find "nit" hunks across the jj work descending from a bookmark.

Two checks are available:

  --whitespace   flag hunks whose changed lines are *each* entirely whitespace
                 (blank-line / whitespace-line churn). A line with any real
                 character fails the test, so reindentation of code is not
                 flagged.
  --non-ascii    flag hunks that *add* a line containing any non-ASCII character.

If no check flag is given, the script defaults to --whitespace only.

By default the commit set is every mutable commit descending from --bookmark
(default ``farmadupe_start``), i.e. all of "my work" rooted at that trunk point.
Pass --this-commit instead to inspect only the working-copy commit (jj's ``@``);
this is the mode to use while editing a commit to fix nits, because it re-reads
that single commit and reports up-to-date line numbers after each edit. The two
scope flags are mutually exclusive. Each commit's own diff (vs its parent) is
inspected.

Output is JSON on stdout: a dict keyed by check name, each value a dict keyed by
jj change id, each value a dict keyed by file path, each value the list of
flagged hunks for that file. Each hunk is a list of its flagged lines, and each
line is unidiff's own fields passed straight through: {"source_line_no": N|null,
"target_line_no": N|null, "line_type": "+"|"-", "value": "..."}. The line is
located by whichever of source_line_no (old file) / target_line_no (new file) is
non-null, so it can be read and edited directly without a text search; line_type
records which side of the diff it is on. No hunk header is emitted.

VERBATIM TRANSPORT (unconditional requirement): "value" is the offending line's
exact content, every space and tab and trailing newline intact, since stripping
the characters this tool reports would defeat its purpose. It is satisfied simply
by not editing anything unidiff gives us: every field above is passed through
untouched, and faithful line-content transport is intrinsic to a diff parser.

Change ids (not commit ids) are emitted because
they are stable across rewrites: editing one commit rewrites the git commit ids
of all its descendants, but their change ids are unchanged, so a change id stays
valid as a target for ``jj edit`` even after earlier fixes. Exit code is 0 if no
check found anything, 1 if any did.
"""

import argparse
import json
import re
import subprocess
import sys

from unidiff import PatchSet

# Per-line whitespace test. l.value includes the trailing newline, and \s covers
# it; an empty or all-whitespace line matches, a line with any real char does not.
WHITESPACE = re.compile(r"\s*")

# Files under these path prefixes are vendored / third-party and never flagged:
# they should never be raised or edited by us
EXCLUDED_PREFIXES = ("farmadupe/kimi_k2.5/reference_implementation/",)


def run(*args: str) -> str:
    """Run a command and return stdout, exiting on failure."""
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(args)}\n{proc.stderr}")
        sys.exit(2)
    return proc.stdout


def enumerate_commits(bookmark: str) -> list[str]:
    """Return the change id of each work commit.

    Change ids are used (rather than commit ids) so the caller can act on the
    results even after rewriting some of the commits: a change id survives the
    rebase that editing an ancestor forces onto its descendants.
    """
    revset = f"({bookmark}:: ~ {bookmark}) & mutable()"
    out = run(
        "jj",
        "log",
        "-r",
        revset,
        "--no-graph",
        "-T",
        'change_id.short() ++ "\\n"',
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


# --- checks: each returns the flagged lines of a hunk ([] means not flagged) ---


def find_whitespace_hunks(hunk) -> list:
    """Changed lines of a hunk whose every added and removed line is whitespace.

    Context lines are ignored. all() over an empty group is True, so pure-add and
    pure-delete hunks are handled with no special-casing; the explicit guard skips
    a pure-context (no change) hunk.
    """
    added = [l for l in hunk if l.is_added]
    removed = [l for l in hunk if l.is_removed]
    if not added and not removed:
        return []
    adds_whitespace = all(WHITESPACE.fullmatch(l.value) for l in added)
    rems_whitespace = all(WHITESPACE.fullmatch(l.value) for l in removed)
    return added + removed if (adds_whitespace and rems_whitespace) else []


def find_non_ascii_hunks(hunk) -> list:
    """Added lines of a hunk that contain any non-ASCII character."""
    return [l for l in hunk if l.is_added and not l.value.isascii()]


# check name -> function that returns a hunk's flagged lines
CHECKS = {
    "whitespace": find_whitespace_hunks,
    "non-ascii": find_non_ascii_hunks,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--bookmark",
        default="farmadupe_start",
        help="base bookmark; work is its mutable descendants (default: farmadupe_start)",
    )
    scope.add_argument(
        "--this-commit",
        action="store_true",
        help="inspect only the working-copy commit (jj's @), not a whole stack; "
        "use after editing a commit to re-read it with up-to-date line numbers",
    )
    parser.add_argument(
        "--whitespace",
        action="store_true",
        help="flag hunks whose changed lines are all whitespace",
    )
    parser.add_argument(
        "--non-ascii",
        action="store_true",
        help="flag hunks that add non-ASCII characters",
    )
    args = parser.parse_args()

    enabled = [
        name
        for name, on in (("whitespace", args.whitespace), ("non-ascii", args.non_ascii))
        if on
    ]
    if not enabled:
        enabled = ["whitespace"]  # script default

    # Each commit is both the diff target and the key its results are grouped
    # under. --bookmark uses change ids (a stable jj-edit-able handle that survives
    # the rewrites fixing one commit forces on its descendants). --this-commit uses
    # the literal "@": you are already editing that commit, so "@" is a clearer
    # "you are in the right place" marker than a change id.
    if args.this_commit:
        commits = ["@"]
    else:
        commits = enumerate_commits(args.bookmark)
    if not commits:
        sys.stderr.write(f"No mutable work descending from '{args.bookmark}'.\n")

    # results[check][commit][path] = list of hunks, each hunk a list of its flagged
    #   lines, each {"source_line_no", "target_line_no", "line_type", "value"} copied
    #   straight from unidiff. Empty per-check dicts are kept so every enabled check
    #   appears in the output.
    results: dict[str, dict[str, dict[str, list]]] = {name: {} for name in enabled}

    for commit in commits:
        diff_text = run("jj", "diff", "-r", commit, "--git")
        if not diff_text.strip():
            continue
        patch = PatchSet(diff_text)
        for name in enabled:
            find_hunks = CHECKS[name]
            for pfile in patch:
                if pfile.path.startswith(EXCLUDED_PREFIXES):
                    continue
                for hunk in pfile:
                    flagged = find_hunks(hunk)
                    if not flagged:
                        continue
                    by_file = results[name].setdefault(commit, {})
                    line_fields = ("source_line_no", "target_line_no", "line_type", "value")
                    by_file.setdefault(pfile.path, []).append(
                        [
                            {f: getattr(l, f) for f in line_fields}
                            for l in flagged
                        ]
                    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    return 1 if any(results[name] for name in enabled) else 0


if __name__ == "__main__":
    sys.exit(main())
