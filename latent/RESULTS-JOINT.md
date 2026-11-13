# One adapter for Noul and Choice requests: results

Design and pass rule: [JOINT.md](JOINT.md), pushed before the run. The separate runs: [RESULTS4.md](RESULTS4.md) for Noul, [RESULTS-CHOICE.md](RESULTS-CHOICE.md) for Choice. Per-case outputs are in [results/](results/); the breakdowns below come from `joint_analyse.py` (output in `results/analysis-joint.json`).

## Passed, and below both separate arms

Run on 12 November 2026, 14:11 to 15:37, measurements included, one run. All 810 steps executed with none skipped, at 0.04% of one host core. Nothing was retried.

One adapter and one projection, trained from `runs/fork-v1/adapter` on both training sets together, were measured on both panels.

| Part | Needed | Measured | Met | Separate latent arm | Text arm |
| --- | --- | --- | --- | --- | --- |
| N1. Noul: assessed twin pairs, both right | at least 385 of 428 | 406 | Yes | 425 | 427 |
| N2. Noul: all questions | at least 1,115 of 1,182 | 1,128 | Yes | 1,162 | 1,174 |
| N3. Noul: free-form questions | at least 151 of 184 | 153 | Yes | 165 | 176 |
| N4. Noul: questions where the two prerequisites differ | at least 65 of 72 | 72 | Yes | 72 | |
| N5. Noul: pointer moved, answer follows it | at least 65 of 72 | 72 | Yes | 72 | |
| C1. Choice: all questions | at least 921 of 1,001 | 929 | Yes | 955 | 971 |
| C2. Choice: select and ruled-out pairs, both right | at least 331 of 374 | 342 | Yes | 352 | 367 |
| C3. Choice: assessed states right in every order | at least 93 of 112 | 105 | Yes | 109 | 103 |

Every part is met. Three are met narrowly: N3 by 2 questions, C1 by 8, N2 by 13. The joint adapter is at or below the separate arm on every part.

## Noul: where the 34 questions went

On the renamed panel, by type of question:

| Questions | Text arm | Pilot 4's arm | Joint | Pilot 3's arm |
| --- | --- | --- | --- | --- |
| All (1,182) | 1,174 | 1,162 | 1,128 | 1,132 |
| Whole rule, free-form state (184) | 176 | 165 | 153 | 153 |
| Whole rule, templated state (282) | 282 | 281 | 272 | 272 |
| Whole rule, prompt asks "is it FALSE that…" (20) | 20 | 20 | 7 | 11 |
| At least one prerequisite (232) | 232 | 232 | 232 | 232 |
| Single prerequisite (464) | 464 | 464 | 464 | 464 |
| Assessed twin pairs (428) | 427 | 425 | 406 | 408 |

- **Pointing is untouched.** Every single-prerequisite question is right, and the answer follows a moved pointer in all 72 cases.
- **The losses are whole-rule questions on states unlike training:** free-form states whose answer is Unresolved (7 of 26, pilot 4's arm 15), and the prompts that ask "is it FALSE that…" (7 of 20).
- **The joint run lands where pilot 3's arm did.** Pilot 3 and pilot 4 are the same Noul-only design in two step orders. So the drop from pilot 4 is no larger than what step order alone produced before.

## Choice: where the 26 questions went

| Questions | Text arm | Choice pilot's arm | Joint |
| --- | --- | --- | --- |
| All (1,001) | 971 | 955 | 929 |
| Select, an alternative is right (319) | 301 | 312 | 302 |
| Select, no alternative is feasible (100) | 100 | 90 | 91 |
| Select, Unresolved (69) | 63 | 56 | 52 |
| Ruled out (374) | 374 | 364 | 351 |
| Alternatives named by single letters (594) | 594 | 591 | 588 |
| Other names (407) | 377 | 364 | 341 |

- The two latent arms agree on 969 questions (926 both right, 43 both wrong). The Choice pilot's arm alone is right on 29, the joint adapter alone on 3.
- **More wrong alternatives.** On ruled-out questions the joint adapter picks the wrong alternative 15 times, against 3 for the Choice pilot's arm.
- **Names unlike training carry the loss:** 23 of the 26 questions are on alternatives not named by single letters.
- There is only one Choice-only run, so there is no measure of how much two Choice runs differ.

## Reported outside the rule

- **Noul on development's own near-identical names:** 1,136 of 1,182 (pilot 4's arm 1,160, pilot 3's arm 1,049), every single-prerequisite question right.
- **Training states with distinct names**, 150 states with all eight questions, half of them held back from training: 1,200 of 1,200.
- **Choice, select questions in listed order:** 212 of 230 (Choice pilot's arm 216, text arm 218).
- **Training loss** is uneven until step 750 and rounds to zero only in the last 60 of the 810 steps. Pilot 4's Noul run was similar: uneven until step 300 of 393. The Choice-only run was at zero from step 200.

## What this does and does not show

- One adapter with one projection handles eight requests across two operations and meets both rules its halves were accepted under.
- It does so with less room than either separate arm. Three parts are within 13 questions of failing.
- Whether joint training costs anything is not settled. For Noul the drop is the size of the difference between two Noul-only runs. For Choice there is no second run to compare with. A different step order could move the joint result in either direction, including below the rule.
- Everything the earlier results left open is still open: fixed requests only, pointers given rather than produced, synthetic cases, Score not included.

## Not accepted, and superseded

The joint vector adapter was never accepted. On 13 November 2026 Vince accepted instead the adapter that serves all three operations with fixed word-sequence requests ([RESULTS-WORDS.md](RESULTS-WORDS.md)), which is at or above this one on every Noul and Choice measure. The verdict above remains the mechanical application of this run's rule (`runs/joint/result.json`).
