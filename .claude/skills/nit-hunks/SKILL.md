---
name: nit-hunks
description: Find "nit" hunks (whitespace-only changes and/or added non-ASCII characters) across the user's jj work (commits descending from a base bookmark). Use when the user asks to check their work for whitespace nits or non-ASCII characters, or mentions "nit hunks".
---

# nit-hunks

Find "nit" hunks across the jj work descending from a base bookmark. The script
runs one or both of these checks:

- **whitespace** -- a hunk in which every changed (added or removed) line is
  entirely whitespace. A line with any non-whitespace character fails the test,
  so the hunk is not flagged.
- **non-ascii** -- a hunk that *adds* a line containing any non-ASCII character.

# ESSENTIAL RULES
* NEVER use grep during this skill. It is STRICTLY FORBIDDEN. Use your file read tool instead

## How to run it

By default, **supply both checks** unless the user asks for just one:

```sh
uv run farmadupe/util/nit_hunks.py --whitespace --non-ascii
```

It has a `uv` shebang and is executable, so this also works:

```sh
farmadupe/util/nit_hunks.py --whitespace --non-ascii
```

Note: the *script's* own default (no flags) is `--whitespace` only. The
*both-checks* default above is the agent's job to supply.

Point it at a different base bookmark with `--bookmark` (default
`farmadupe_start`):

```sh
farmadupe/util/nit_hunks.py --whitespace --non-ascii --bookmark some_other_start
```

The script inspects every mutable commit descending from the bookmark (revset
`(<bookmark>:: ~ <bookmark>) & mutable()`), diffing each commit against its
parent.

## Reading the output

Output is JSON on stdout: a dict keyed by check name (`whitespace`,
`non-ascii`), each value a dict keyed by jj **change id**, each value a dict
keyed by file path, each value a list of flagged hunks. Each hunk is
`{"header": "@@ ... @@", "lines": [...]}`, where `header` is the unified-diff
hunk header that locates the hunk in that commit, and `lines` are the offending
changed lines verbatim, each prefixed with `+` (added) or `-` (removed). A check
that flags nothing maps to an empty object. Exit code is `0` when nothing is
flagged and `1` when any check finds something, so it can gate other steps.

The keys are change ids, not commit ids, on purpose: a change id stays a valid
`jj edit` target even after you rewrite earlier commits (which changes their git
commit ids, and those of every descendant).

## Fixing what it finds

Any fixups are **limited in scope to the hunks the script identifies** -- only
the specific changed lines reported under each `check -> change id -> file ->
hunk`, not the rest of the tree (which may legitimately contain the same
characters in code you did not touch, vendored sources, etc.).

The fixes are to be made **in the jj commits where the hunks live**, not in the
working copy on top of them. Each reported hunk belongs to a specific commit
(its change id key). Always `jj edit` by **change id** from the output, never a
commit id: editing one commit rewrites the commit ids of all its descendants, so
a commit id read earlier may already be stale, and reusing it can create a
divergent change.

Once you are editing a commit, re-run the script scoped to just that commit with
**`--this-commit`** to see its remaining nits with line numbers that reflect the
edits you have already made:

```sh
farmadupe/util/nit_hunks.py --whitespace --non-ascii --this-commit
```

**Do not use `grep` (or any other text search) to find the offending lines.** Use
`--this-commit` for that. grep over the tree also matches identical characters in
code you did not touch (other commits, vendored sources) -- those are out of
scope -- and its line numbers drift as you edit, whereas `--this-commit` re-reads
the working-copy commit and reports the exact in-scope hunks and current line
numbers.

**Do not use `jj absorb`.** The fixes must be made one by one, in place: for each
reported hunk, `jj edit` the exact commit that owns it, fix the offending lines
in that commit directly, and move on to the next. No batching the edits into a
scratch commit and squashing them down.

**Stop immediately if you ever notice a divergent change.** jj reports
divergence as a `??` change-id marker, a `(divergent)` tag in `jj log`, or a
`change id ... is divergent` error from a command. 


# RULES
* NEVER use grep during this skill. It is STRICTLY FORBIDDEN. Use your file read tool instead.