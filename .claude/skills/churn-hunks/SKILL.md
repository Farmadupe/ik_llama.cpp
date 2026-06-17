---
name: churn-hunks
description: Report net-zero "churn" across a chain of jj commits -- work that a later commit undoes, so it never reaches the tip -- so a stack can be slimmed before a long rebase. The mechanical pass (farmadupe/util/churn_hunks.py) finds the cancelled work exactly; the agent reads the implicated commits and reports what is genuinely wasted versus structural noise or change-and-refine. This is a reporting skill: it surfaces the churn and does not modify the stack. Use when the user asks to find or report churn, net-zero / cancelled / wasted work, or to assess a stack before rebasing.
---

# churn-hunks

A change introduced by one commit and later undone by another never reaches the
tip of a stack, so it is invisible in the cumulative `base..tip` diff yet present
in the per-commit diffs. That gap is wasted edit work. Finding it before a long
rebase lets the stack be slimmed deliberately, rather than discovering the dead
work as conflicts mid-rebase.

This skill has two layers, and the second is the point:

1. **Mechanical (`churn_hunks.py`).** Exact detection by multiset subtraction:
   every added/removed line summed across the per-commit diffs, minus the lines
   of the single `base..tip` diff, is the churn. The *count* is exact. What it
   cannot do alone is tell genuinely wasted work apart from churn that is fine,
   nor pin an ambiguous duplicate line (a lone brace) to the exact commit.
2. **Agentic (you).** Read the commits the script implicates and close that gap:
   confirm real cancellations, dismiss noise, resolve ambiguous localization, and
   **report** what is genuinely wasted. This is judgment work; the script only
   hands you the leads.

You report -- you do not fix: telling genuine waste from structural noise is
judgment work.

## Agreeing the range

The script takes a jj revset naming the work commits, exactly as you would name a
span to jj. There is no single right answer for "the chain" -- confirm it with
the user if unstated. Common forms:

```sh
farmadupe/util/churn_hunks.py -r 'mybookmark::@'   # a bookmark up to the working copy
farmadupe/util/churn_hunks.py -r 'base..tip'       # jj's exclusive-of-base span
farmadupe/util/churn_hunks.py                       # default: mutable() & ::@ (your stack)
```

The base is the parent of the range's root (`(roots(R))-`), the tip is its head
(`heads(R)`). Both must resolve to a single commit, so the range must be linear
(one root, one head); a fork or merge in the set is a clean error, not a guess.

## Reading the output

JSON on stdout, exit 1 when anything is flagged (so it can gate a pre-rebase
check), exit 0 when clean. Three reports:

- **`net_zero_files`** -- files touched by some commit in the range but identical
  between base and tip. Every commit's work on them cancels. This is **exact**;
  trust it.
- **`files[path].churn_lines` / `work_lines` / `net_lines`** -- the per-file
  churn volume and its context. A high `churn_lines / net_lines` ratio is the
  "look here first" triage signal. Files are sorted worst-churn first.
- **`files[path].churn_added` / `churn_removed`** -- the churned line contents.
  Each carries `added_by` (change ids that added the line) and `removed_by`
  (change ids that removed it): your **candidates** for the real add/undo
  partner. Pure-whitespace churn is omitted (it is a whitespace matter, not churn).

Everything is keyed by **change id**; commit ids are deliberately absent (they go
stale across rewrites, change ids do not).

## Closing the gap (the agentic core)

The script gives leads, not verdicts. For each candidate, read the implicated
commits (`jj diff -r <change_id>`, `jj show -r <change_id>`) and sort it into one
of three buckets. Only the first is genuine waste:

1. **Genuine net-zero work** -- a commit adds a block (a function, a branch, a
   field) and a later commit deletes it, leaving no trace. The added and removed
   contents match and nothing replaced them. This is genuinely wasted work.
2. **Structural / mechanical churn** -- braces, `});`, blank scaffolding, a line
   that moved when surrounding code was reindented or relocated. The content
   "cancels" only because identical punctuation appears in many places. Usually
   harmless; dismiss it. Most high counts on real code are mostly this -- say so
   rather than inflating a finding.
3. **Change-and-refine** -- a commit adds X, a later commit *replaces* X with a
   better Y. The removal of X is churn by the line math, but the work led
   somewhere; it is not wasted. Do not report it as cancelled.

Resolve ambiguous localization the same way: when `added_by` or `removed_by`
lists several change ids for one line, the multiset cannot say which pairing is
real -- read those commits and decide from the code, not the counts.

### Discipline

- **Empty is a normal, common result.** A clean stack, or one whose only churn is
  structural noise, is a finding: report "no wasted work" and stop. Do not
  manufacture findings to look productive.
- **Evidence per claim.** Every "this work is cancelled" must cite the change-id
  pair and name what was added and where it was removed. No evidence, no claim.
- **Argue the opposite before committing to a finding.** For each block you would
  call wasted, check whether it is actually load-bearing (bucket 3) or whether
  the removal was a real revert (bucket 1). Default to not flagging it when
  unsure.

### The deliverable

Two parts, and the second is the substance:

- `net_zero_files` -- the exact, no-judgment wins, reported as-is.
- The churn you read and confirmed as genuine waste (bucket 1): each a change-id
  pair with one line of evidence (what was added, where it was undone), plus a
  one-line note on what you dismissed as noise.

The script's lists are leads, not the report. Do not skip the reading and just
echo `net_zero_files` and the raw candidates -- the report is what you concluded
after reading the commits behind them.

## Reads only

The jj you run is read-only: the script, and `jj diff -r <change_id>` /
`jj show -r <change_id>` to read the implicated commits while classifying. This
skill surfaces the churn and stops there.
