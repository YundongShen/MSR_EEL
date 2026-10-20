# EEL — RQ5 Engineer Retention Review
### Reviewer Handbook

Thanks for helping with this review. This document explains what you're doing and how to do it — please read it once before opening the spreadsheet (5 minutes).

---

## 1. What this is

We're studying whether an AI system's learned signal for "which code changes in a patch are actually needed" agrees with how engineers judge the same question. You'll be shown some real GitHub issues, each paired with a set of candidate code changes ("hunks") that were generated for that issue. Your job is to judge each hunk on its own merits — not to guess which ones came from a human vs. a model.

## 2. Where to find your work

Open `EXP5_engineer_review_survey.xlsx`. Find the sheet tab with your reviewer ID (e.g. `R01`). Your 10 assigned issues are already loaded there — nothing else to set up. Each issue is identified only by a task number (e.g. `Task T12`) and its repository.

## 3. What you'll see

Each issue starts with:

| Element | Meaning |
|---|---|
| Issue header | Task number and repository |
| Issue text | The original GitHub issue (the requirement) |
| 🟢 Green "Fail-to-pass tests" block | The tests a correct fix must make pass, i.e. the expected behaviour. Reference only — you do not label it. Long tests are shortened to their signature and assertion lines. |

Then each hunk block has, top to bottom:

| Element | Meaning |
|---|---|
| `H1`, `H2`, ... + filepath | Hunk ID and the file it touches |
| 🟠 Amber banner (sometimes) | Another hunk in this issue touches the same file/region — see §6 |
| Grey "Context" block | The full function the hunk belongs to (or nearby lines if we couldn't resolve the function) — reference only, not itself part of what you're judging |
| Colored diff | Green = added, red = removed, grey italic = the hunk's location header (`@@ ... @@`) |
| Yellow `label` cell | Where you record your judgment |

Hunks within an issue are shown in **random order** and are **not marked** as reference or generated. Judge each one only on whether it belongs in a correct fix for the stated issue.

## 4. How to judge — three categories

Use this test for each hunk:

- **retain** — if you deleted this hunk, the issue would still be broken, or the fix would be incomplete/incorrect.
- **not certain** — reasonable and plausibly helpful, but the issue is still correctly resolved without it (e.g. a nearby cleanup, a docstring update, a defensive check not strictly required by the report).
- **reject** — unrelated to the issue, duplicates a change made by another hunk, or looks actively wrong/inconsistent with the surrounding code.

If you're unsure between two categories, ask: *"would I block a code review on this hunk being missing?"* — yes → retain, no → not certain/reject depending on whether it's still a reasonable change.

## 5. Worked example

*(Practice only — not one of your assigned issues.)*

> **Issue:** `sort_records()` returns records in the wrong order when the sort key is missing for some entries.
>
> **H1** — in `sort_records()`: changes `key=lambda r: r["priority"]` to `key=lambda r: r.get("priority", 0)`.
> → **retain.** This is the direct fix: it's exactly what stops the crash/misordering described in the issue.
>
> **H2** — in the same function: renames the loop variable `r` to `record` for readability.
> → **not certain.** Harmless and arguably nicer, but the issue is fully resolved without it.
>
> **H3** — in a different file, `format_records()`: adds a new `verbose` flag to a print statement.
> → **reject.** Unrelated to the sort-order bug; nothing about the issue asks for this.

## 6. The amber "same region" banner

When two hunks in the same issue touch the same file at nearby lines, we flag both with an amber banner naming the other hunk. This usually means they are **two different implementations of the same underlying change** — only one is likely to belong in a correct fix. Read both before labeling either; don't assume the first one you see is right.

## 7. Ground rules

- **Work independently.** Each issue is reviewed by 3 people separately — please don't discuss your judgments with other reviewers until everyone is done, or the independence we rely on for the results breaks down.
- **No need to run any code** or check out the repositories. Judge from the issue text, the test block, the context block, and the diff as given.
- **Don't look up the original issue or pull request.** Please don't search GitHub (or elsewhere) by repository name, issue wording, or any number in the text — seeing the merged fix would defeat the purpose of the study.
- **Don't leave a hunk blank** — if you're genuinely torn, pick your best guess over leaving it empty.
- Budget roughly 4-6 minutes per issue; you have 10, so about an hour in total.

## 8. FAQ

**Do I need to know if a hunk is the "real" fix or AI-generated?**
No — and you can't tell from the sheet on purpose. Judge purely on whether the hunk is a reasonable, necessary part of fixing the described issue.

**What if the context block doesn't fully explain what a hunk does?**
That happens sometimes — make your best judgment from what's shown. Don't go looking up the repository on GitHub; the point is to judge from the same information available in the sheet.

**What if I think none of the categories fit well?**
Pick the closest one. The three categories map to a numeric scale in our analysis, so "closest fit" is more useful than leaving it blank.

---
Questions about the task (not about a specific issue's content) → contact the study organizer.
