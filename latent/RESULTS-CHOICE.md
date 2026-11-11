# Latent Choice request pilot: results

Design and pass rule: [CHOICE.md](CHOICE.md), pushed before the runs. Noul pilots: [RESULTS4.md](RESULTS4.md) and earlier. Per-case outputs are in [results/](results/); the breakdowns below come from `choice_analyse.py` (output in `results/analysis-choice.json`).

## Passed

Run on 11 November 2026, one run per arm: text 21:58 to 22:40, latent 22:40 to 23:17, measurements included. Every planned step executed with none skipped, at 0.04% of one host core. Nothing was retried.

The panel is 1,001 development questions on 230 cases: select and ruled-out questions, each asked with the alternatives in two or three orders.

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 0. Conclusive: text arm, all questions | at least 901 of 1,001 | 971 | Yes |
| 1. All questions | at least 921 of 1,001 (text arm less 5 points) | 955 | Yes |
| 2. Select and ruled-out question on the same state and order, both right | at least 331 of 374 (90% of the text arm's 367) | 352 | Yes |
| 3. Assessed states whose select question is right in every order | at least 93 of 112 (90% of the text arm's 103) | 109 | Yes |

The latent arm's request is 16 vectors for the kind of question and one replayed pointer per alternative. It never sees an alternative's properties as an option; the text arm does.

## The three arms side by side

"Starting adapter" is `fork-v1` before any Choice training, reading the fixed wordings as text with names and properties.

| Questions | Starting adapter, text | Text arm | Latent arm |
| --- | --- | --- | --- |
| All (1,001) | 889 | 971 | 955 |
| Select (627) | 592 | 597 | 591 |
| Ruled out (374) | 297 | 374 | 364 |
| Kind pairs, both right (374) | 296 | 367 | 352 |
| Assessed states right in every order (112) | 105 | 103 | 109 |
| Select, an alternative is right (319) | 305 | 301 | 312 |
| Select, no alternative is feasible (100) | 100 | 100 | 90 |
| Select, Unresolved (69) | 54 | 63 | 56 |
| Select, Conflicting (66) | 60 | 60 | 60 |
| Select, Insufficient (73) | 73 | 73 | 73 |

The two trained arms agree on 963 questions (944 both right, 19 both wrong). The text arm alone is right on 27 and the latent arm alone on 11.

## Where the latent arm is behind

- **It picks an alternative where a status was right.** 29 of its 46 errors are of this kind, against 12 of the text arm's 30. Ten are select questions where no alternative is feasible, nine of them on three states of one family: asked which room to book when none qualifies, it names the same room in every order. The text arm gets all nine right.
- **Ruled-out questions:** 364 of 374, where the text arm has all 374. Seven of the ten errors answer with a status.

## Where it is ahead

- **Picking the right alternative when there is one:** 312 of 319 select questions, against 301 for the text arm, and 109 of 112 states right in every order against 103.
- The text arm ends below the starting adapter on that last measure (103 against 105), so the threshold of part 3 was lower than it would have been against the starting adapter (95). The latent arm clears either.
- Before training, names without properties bound poorly once the order changed (84 of 114 assessed cases in reversed order, [CHOICE.md](CHOICE.md)). With pointers the arm is right in every order on 109 of 112.

## Reported outside the rule

**Names.** Every training alternative is a single letter with a one-token pointer. Development has other names, with pointers of up to seven tokens.

| Alternatives named by | Starting adapter, text | Text arm | Latent arm |
| --- | --- | --- | --- |
| Single letters (594 questions) | 564 | 594 | 591 |
| Other names, found by rule (314) | 241 | 287 | 271 |
| Other names, mapped by hand (93) | 84 | 90 | 93 |

Nearly all errors of both trained arms are on names unlike training: all 30 of the text arm's and 43 of the latent arm's 46. The hand-mapped alternatives, whose mapping no person has reviewed, are all answered right by the latent arm.

**Listed-order select questions:** 216 of 230 for the latent arm and 218 for the text arm. The starting adapter has 225 of 232 with each case's own question, and 213 of 230 with the fixed wording. Neither trained arm reaches the 225.

**Errors cluster.** The latent arm's 46 errors fall on 24 of the 230 states, the text arm's 30 on 14.

**Training loss** rounds to zero from step 150 (text) and step 200 (latent). The training cases are fitted early, so the result rests on development, whose names are unlike training.

## What this does and does not show

- A Choice request can be two fixed request kinds as vectors plus one replayed pointer per alternative, with the properties left in the state. On this panel it is within 16 questions of an arm that reads the question and the properties as words.
- The vectors steer: the same state and pointers get different right answers under the two requests in 352 of 374 pairs. An arm that ignored the vectors could get none of them.
- The pointers bind: the answer holds when the order of the alternatives changes.
- It is weaker than text at recognising that no alternative should be picked.
- One run per arm. The Noul pilots showed that two runs of one design can differ widely on cases unlike training, and that is not measured here.
- Two fixed requests only require the vectors to tell two things apart.
- Choice only, from `fork-v1`. The Noul requests were not trained alongside, so one adapter serving both is not shown.
- The request always points at every alternative, the pointers are given rather than produced by the native side, and the cases are synthetic.

## Open for the owner

Whether to accept this request format for Choice is Vince's decision. The verdict above is the mechanical application of the rule (`runs/choice-latent/result.json`).
