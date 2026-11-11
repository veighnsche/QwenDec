# Latent question pilot 4: results

Design and pass rule: [PILOT4.md](PILOT4.md), pushed before the run. Earlier pilots: [RESULTS.md](RESULTS.md), [RESULTS2.md](RESULTS2.md), [RESULTS3.md](RESULTS3.md). Per-case outputs are in [results/](results/).

## Passed

Run on 11 November 2026, 20:34 to 21:22, one run. Every planned step executed with none skipped, at 0.04% of one host core. Nothing was retried.

The panel is development with the prerequisites of the states that parse renamed to distinct new names (1,182 questions).

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 1. Assessed twin pairs, both right | at least 385 of 428 | 425 | Yes |
| 2. All questions | at least 1,115 of 1,182 | 1,162 | Yes |
| 3. Free-form questions | at least 151 of 184 | 165 | Yes |
| 4. Single-prerequisite questions where the two prerequisites differ | at least 65 of 72 | 72 | Yes |
| 5. Pointer moved to the other prerequisite: answer follows it | at least 65 of 72 | 72 | Yes |
| 6. Zero question vectors: assessed twin pairs | below 10% | 0/428 | Yes |

The text arm has 1,174 and 427 on the same panel.

## Reported separately, outside the rule

**Near-identical names.** The same arm on development as it is, where the two prerequisites differ in one word: 1,160/1,182, 425/428 assessed twin pairs, 72/72 where the prerequisites differ, and every single-prerequisite question right, negated ones included (464/464). The failure pilot 3 had on these names did not occur in this run.

**Held-back questions on training states:** 600/600.

**By type of question** (`analyse4.py`, output in `results/analysis-pilot4.json`):

| Questions | Text arm | Pilot 4 | Pilot 4, development's own names | Pilot 3's arm, same panel |
| --- | --- | --- | --- | --- |
| All (1,182) | 1,174 | 1,162 | 1,160 | 1,132 |
| Whole rule, free-form state (184) | 176 | 165 | 165 | 153 |
| Whole rule, templated state (282) | 282 | 281 | 281 | 272 |
| Whole rule, prompt asks "is it FALSE that…" (20) | 20 | 20 | 20 | 11 |
| At least one prerequisite (232) | 232 | 232 | 230 | 232 |
| Single prerequisite (464) | 464 | 464 | 464 | 464 |

Of pilot 4's 20 errors on the panel, 19 are free-form questions, 11 of them on states whose gold status is Unresolved (45/56 recognised, against 55/56 for text).

## Two runs of one design

Pilot 3 and pilot 4 are the same design, examples and settings. Only the order of the training steps differs.

| | Pilot 3's arm | Pilot 4's arm |
| --- | --- | --- |
| New panel, all questions | 1,132 | 1,162 |
| New panel, assessed twin pairs | 408 | 425 |
| Development's own names, all questions | 1,049 | 1,160 |
| Development's own names, negated single-prerequisite questions | 150/232 | 232/232 |
| "Is it FALSE that…" questions | 10 or 11 of 20 | 20/20 |

Pilot 3's arm was measured on the new panel after this run, for comparison only: it reaches the thresholds of parts 1 to 4 there (parts 5 and 6 were not measured on that panel). On development's own names the two runs are 111 questions apart.

So the step order alone moves the result a great deal on the cases that are unlike training. One pass over this data does not settle those cases reliably, even though this run got them right.

## What this does and does not show

- Six fixed requests as vectors, plus a pointer whose tokens are replayed, are enough for this panel: free-form states, item names never seen in training, negation, and binding that follows the pointer.
- The rule was met on the first run after it was declared. It is the fourth rule, and it was written after seeing where the third failed; near-identical names were moved out of it by decision. This run happened to get those right as well.
- With six distinct requests the vectors only have to tell six things apart. This does not show that fixed-encoder vectors can carry a question's content, which is what the build plan's gate asked.
- Two runs of the design differ widely on cases unlike training. Reliability across runs is not established.
- Noul only. The pointer is given, not produced by the native side. The cases are synthetic and labelled by rule.

## Correction, added after the run

Part 6 of the rule could not fail. With the vectors zeroed, a question and its negation have identical inputs, so they get the same answer and are never both right; 0/428 follows by construction, here and in every earlier pilot. The pass does not depend on it: that the vectors are used is shown by part 1, where 425 of 428 pairs that differ only in their vectors are both answered right.

## Accepted

Vince accepted the request format on 11 November 2026 as the basis for the next stage: fixed request kinds as vectors, plus a pointer whose tokens are replayed. The acceptance is for the format on Noul. It leaves open the reliability across runs and does not cover Choice, Score, or a pointer produced by the native side (`runs/pilot4-canonical/result.json`).
