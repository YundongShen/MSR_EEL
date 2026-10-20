# EEL — Engineer Retention Review
### Reviewer Handbook

This handbook describes the review task and labeling procedure. Please read it before completing the spreadsheet.

---

## 1. What this is

We compare reference-patch matching labels with engineers' judgments about generated code edits. Each task contains a GitHub issue and candidate code changes ("hunks") generated for that issue. Judge whether each hunk belongs in a correct fix using the requirement, failing tests, unmodified code, and diff provided.

## 2. Where to find your work

Open `engineer_review_survey.xlsx`. Find the sheet tab with your reviewer ID (e.g. `R01`). Your 10 assigned issues are already loaded there — nothing else to set up. Each issue is identified only by a task number (e.g. `Task T12`) and its repository.

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
| Grey "Context" block | Unmodified source around the change, shown for reference and not itself labeled |
| Colored diff | Green = added, red = removed, grey italic = the hunk's location header (`@@ ... @@`) |
| Yellow `label` cell | Where you record your judgment |

All candidate hunks are **LLM-generated** and are shown in **random order** within each issue. The reference patch, tier labels, and model scores are not shown. Judge each hunk on whether it belongs in a correct fix for the stated issue.

## 4. How to judge — three categories

Use this test for each hunk:

- **retain** — the hunk should be included in a correct fix for the stated issue, based on the provided requirement, tests, and source context.
- **not certain** — the provided information is insufficient to decide whether the hunk should be retained or rejected.
- **reject** — the hunk should not be included because it is unrelated to the issue, incorrect, or inconsistent with the required behavior.

Use **not certain** when the available evidence does not support a clear retain or reject decision. Assess each hunk independently; there is no required number of hunks in any category.

## 5. Worked example

*(Practice only — not one of your assigned issues.)*

> **Issue:** `sort_records()` raises `KeyError` when a record has no `priority`. Missing priorities should be treated as zero.
>
> **H1** — in `sort_records()`: changes `key=lambda r: r["priority"]` to `key=lambda r: r.get("priority", 0)`.
> → **retain.** This change implements the required behavior for missing priorities.
>
> **H2** — in the same function: converts string priorities to integers.
> → **not certain.** The provided information does not establish whether string priorities are valid input or whether this conversion is required.
>
> **H3** — in a different file, `format_records()`: adds a new `verbose` flag to a print statement.
> → **reject.** Unrelated to the sort-order bug; nothing about the issue asks for this.

## 6. The amber "same region" banner

When two hunks in the same issue touch the same file at nearby lines, an amber banner names the other hunk. The banner identifies nearby changes; it does not indicate their correctness or imply that only one can be retained. Read the related context and judge each hunk independently.

## 7. Ground rules

- **Work independently.** Each issue is reviewed by 3 people separately — please don't discuss your judgments with other reviewers until everyone is done, or the independence we rely on for the results breaks down.
- **No need to run any code** or check out the repositories. Judge from the issue text, the test block, the context block, and the diff as given.
- **Don't look up the original issue or pull request.** Please don't search GitHub (or elsewhere) by repository name, issue wording, or any number in the text — seeing the merged fix would defeat the purpose of the study.
- **Label every hunk.** Use **not certain** if the provided information does not support a clear decision.
- Budget roughly 4-6 minutes per issue; you have 10, so about an hour in total.

## 8. FAQ

**Will I see the reference patch, tier labels, or model scores?**
No. All displayed candidate hunks are generated. Judge them using only the requirement, tests, source context, and diff provided.

**What if the context block doesn't fully explain what a hunk does?**
Use **not certain** if the missing context prevents a clear decision. Do not look up the repository on GitHub; all reviewers should judge from the information provided in the sheet.

**What if I think none of the categories fit well?**
Use **not certain** when you cannot determine whether the hunk should be retained or rejected from the provided information.

---
Questions about the task (not about a specific issue's content) → contact the study organizer.
