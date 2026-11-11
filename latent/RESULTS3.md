# Latent question pilot 3: results

Design and pass rule: [PILOT3.md](PILOT3.md), written before any pilot 3 training. Earlier pilots: [RESULTS.md](RESULTS.md), [RESULTS2.md](RESULTS2.md). All rule figures are on the 1,182 development variants. Per-case outputs are in [results/](results/).

## Did not pass

Run on 11 November 2026, 18:57 to 19:42, one run. Every planned step executed with none skipped, at 0.04% of one host core. Nothing was retried.

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 1. Assessed twin pairs, both right | at least 385 of 428 | 325 | No |
| 2. All questions | at least 1,115 of 1,182 | 1,049 | No |
| 3. Free-form questions | at least 151 of 184 | 153 | Yes |
| 4. Single-prerequisite questions where the two prerequisites differ | at least 65 of 72 | 56 | No |
| 5. Pointer moved to the other prerequisite: answer follows it | at least 65 of 72 | 56 | No |
| 6. Zero question vectors: assessed twin pairs | below 10% | 0/428 | Yes |

## Where it fails

By type of question (`analyse3.py`, output in `results/analysis-pilot3.json`):

| Development questions | Text arm (pilot 1) | Canonical wording as text, untrained | Replayed span (pilot 2) | Canonical requests (pilot 3) |
| --- | --- | --- | --- | --- |
| Whole rule, free-form state (184) | 176 | 167 | 115 | 153 |
| Whole rule, templated state (282) | 282 | 282 | 273 | 272 |
| Whole rule, prompt asks "is it FALSE that…" (20) | 20 | 10 | 20 | 10 |
| At least one prerequisite (232) | 232 | 232 | 232 | 232 |
| Single prerequisite, plain (232) | 232 | – | 232 | 232 |
| Single prerequisite, negated (232) | 232 | – | 232 | 150 |

### What canonical requests fixed

Free-form questions rose from 115 to 153 of 184, past the rule's 151 and close to the 167 the same request reaches as text. That was the purpose of this pilot.

### What broke: negation on single-prerequisite questions

All 82 single-prerequisite errors are negated questions answered as if they were not negated. In 81 the prerequisite holds and the arm answered True. Plain single-prerequisite questions are 232/232.

This is why rules 1, 4 and 5 failed. In the 72 differing-prerequisite cases, the 16 misses are negated questions; an answer without the negation equals the other prerequisite's answer there, so the binding figures cannot tell the two apart. The same error occurs where the two prerequisites agree and binding is not involved (28 of 56 first-prerequisite negated questions wrong).

It happens on development's names only. On the 150 training states with distinct names the arm is 1,200/1,200, negated single-prerequisite questions included.

The cause is not established. Two differences from pilot 2, where the same questions were 232/232:

- The request no longer contains the target's name, so there are two single-prerequisite requests for the whole dataset instead of one pair per target.
- Development's names are themselves clauses ("the first documented prerequisite holds"), read right after the vectors. Training names are noun phrases ("kitchen booking for instructor").

### The 20 "is it FALSE that" questions

As the check before training predicted, the arm follows the state's own prompt on these (10/20). That prompt wording does not occur in training.

## Across the three pilots

| | Pilot 1, 16 vectors | Pilot 2, replayed span | Pilot 3, canonical requests |
| --- | --- | --- | --- |
| Development, all questions | 1,024 | 1,104 | 1,049 |
| 600 held-back questions on training states | 481 | 600 | 600 |

Since the pointer was added, the arms answer every held-back question on training-style states. Each fails somewhere on development, and in a different place each time: development differs from training in its prerequisite names, its free-form states and some of its prompts. The text arm does not have this problem (1,174), because the model already reads question text.

So the open question has changed. It is no longer whether a request of vectors plus a pointer can work, but whether an arm trained on one template carries over to states unlike it.

## What this does and does not show

- Six fixed requests plus a replayed pointer are enough for free-form questions on this panel (153/184, rule met).
- The same arm drops the negation on single-prerequisite questions with development's names, which pilot 2's arm did not.
- As stated in the design, six requests only require the vectors to tell six things apart. Nothing here shows that fixed-encoder vectors can carry a question's content.
- One run per design. How much of the difference between pilot 2 and pilot 3 on negated questions is the design and how much is run-to-run variation was not measured.

Under the rule nothing further is retried.
