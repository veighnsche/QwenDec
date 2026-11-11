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

## Follow-up, no training: development's prerequisites renamed

Development's two names are unseen in training, are clauses, and differ in one word. To see which of these matters, the 464 single-prerequisite questions were copied twice with other names (`rename.py`, twelve pairs each, none a training name) and the trained arms measured again (`run_renamed.sh`, `analyse_renamed.py`, output in `results/analysis-renamed.json`). Labels are unchanged.

| Single-prerequisite questions (464) | Development's names | Distinct noun phrases | Distinct clauses ending in "holds" |
| --- | --- | --- | --- |
| Pilot 2, pointer as one vector | 344 | 464 | 441 |
| Pilot 2, pointer as replayed span | 464 | 464 | 464 |
| Pilot 3, canonical requests | 382 | 464 | 462 |

| Of those, where the two prerequisites differ (72) | Development's names | Distinct noun phrases | Distinct clauses |
| --- | --- | --- | --- |
| Pilot 2, pointer as one vector | 36 | 72 | 65 |
| Pilot 2, pointer as replayed span | 72 | 72 | 72 |
| Pilot 3, canonical requests | 56 | 72 | 71 |

What this settles:

- Unseen names are not the problem. With distinct names never seen in training, every arm answers every question when the names are noun phrases.
- Pilot 3's dropped negation is specific to development's two near-identical names: 150/232 negated questions there, 232 and 230 with distinct names.
- The one-vector pointer does bind unseen names (72/72 and 65/72). It fails on development's pair because the two pooled pointers are nearly the same. This answers the question [RESULTS2.md](RESULTS2.md) left open.
- Clause-style names cost the one-vector pointer a little (441/464) and the replayed span nothing.

This is a diagnosis made after the runs, on edited states. It does not change any verdict: the rules were declared on development as it is.

## Across the three pilots

| | Pilot 1, 16 vectors | Pilot 2, replayed span | Pilot 3, canonical requests |
| --- | --- | --- | --- |
| Development, all questions | 1,024 | 1,104 | 1,049 |
| 600 held-back questions on training states | 481 | 600 | 600 |

Since the pointer was added, the arms answer every held-back question on training-style states. Each fails somewhere on development, and in a different place each time: development differs from training in its prerequisite names, its free-form states and some of its prompts. The text arm does not have this problem (1,174), because the model already reads question text.

The follow-up above narrows this for single-prerequisite questions: the arms carry over to unseen names, and what they miss is one pair of names that differ in a single word. The remaining development errors of pilot 3 are 31 free-form questions (17 for the same request as text) and the 10 questions whose prompt asks the opposite.

## What this does and does not show

- Six fixed requests plus a replayed pointer are enough for free-form questions on this panel (153/184, rule met).
- The same arm drops the negation on single-prerequisite questions with development's two near-identical names, which pilot 2's arm did not. With distinct unseen names it does not.
- As stated in the design, six requests only require the vectors to tell six things apart. Nothing here shows that fixed-encoder vectors can carry a question's content.
- One run per design. How much of the difference between pilot 2 and pilot 3 on negated questions is the design and how much is run-to-run variation was not measured.

Under the rule nothing further is retried.
