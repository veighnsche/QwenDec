# One adapter for Noul and Choice requests: design and pass rule

Written 11 November 2026, before any joint training. Vince approved building it the same day. The examples and scripts were built that day without the GPU. The run was started on 12 November at 14:13, after the two GPU checks passed.

## Question

Can one decision adapter, with one projection, serve both accepted request formats: the six Noul requests and the two Choice requests?

## Why this question

The build plan has a single decision adapter. The two formats were accepted from separate runs, each starting from `runs/fork-v1/adapter` and each training its own projection:

| Run | Requests | Training examples | Result on its panel |
| --- | --- | --- | --- |
| Pilot 4 ([RESULTS4.md](RESULTS4.md)) | 6 Noul requests, one pointer | 4,468 | 1,162 of 1,182 |
| Choice pilot ([RESULTS-CHOICE.md](RESULTS-CHOICE.md)) | 2 Choice requests, one pointer per alternative | 3,954 | 955 of 1,001 |

Neither adapter has seen the other's requests. Training one on Choice may undo what the other learned for Noul, and the reverse. That is what this run measures.

## The run

One latent arm. No new text arm: the text references are the two already measured.

- **Start:** `runs/fork-v1/adapter` and an identity projection, as in both pilots.
- **Examples:** the union of the two training sets, unchanged: the 4,468 Noul examples (selection hash `0dffd3a6…`) and the 3,954 Choice examples (selection hash `9df5422f…`). 8,422 examples, one pass. Each operation gets exactly the examples its separate arm had.
- **Requests:** the eight request vector sets from the fixed encoder, copied from the two existing files into one. Nothing is encoded again.
- **Batches:** examples of both operations are grouped by length together, so a step can hold both. The order of steps is drawn with a new seed (5). 810 steps: 265 hold both operations, 329 only Noul and 216 only Choice. Most Noul prompts are shorter than any Choice prompt, which is why many steps hold one operation.
- **Settings:** as in the pilots: learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels, the on-device loop.

Prompts, pointers and the loss are those of the two pilots. A Choice example still credits its acceptable alternatives as a set.

## Pass rule

The joint adapter is judged by the rules its two halves were accepted under, with the numbers now fixed. Both must be met.

**Noul**, on the renamed development panel (`noul-development-panel4.jsonl`, 1,182 questions):

| Part | Needed | Text arm | Pilot 4's arm |
| --- | --- | --- | --- |
| N1. Assessed twin pairs, both right | at least 385 of 428 | 427 | 425 |
| N2. All questions | at least 1,115 of 1,182 | 1,174 | 1,162 |
| N3. Free-form questions | at least 151 of 184 | 176 | 165 |
| N4. Single-prerequisite questions where the two prerequisites differ | at least 65 of 72 | | 72 |
| N5. Pointer moved to the other prerequisite: answer follows it | at least 65 of 72 | | 72 |

**Choice**, on the development variants (`choice-development.jsonl`, 1,001 questions):

| Part | Needed | Text arm | Choice pilot's arm |
| --- | --- | --- | --- |
| C1. All questions | at least 921 of 1,001 | 971 | 955 |
| C2. Select and ruled-out question on the same state and order, both right | at least 331 of 374 | 367 | 352 |
| C3. Assessed states whose select question is right in every order | at least 93 of 112 | 103 | 109 |

One run, the final step's weights are the only candidate, nothing is retried. The zero-vector part of the earlier Noul rules is not carried over; it could not fail ([CHOICE.md](CHOICE.md)).

Reported outside the rule:

- Each measure against the separate arm's number.
- Noul on development's own near-identical names (pilot 4's arm: 1,160 of 1,182), the 600 held-back questions on training states (600 of 600), and results by type of question.
- Choice by status, with select questions where no alternative is feasible singled out (Choice pilot's arm: 90 of 100), and by kind of name.

## What each outcome would mean

| Outcome | Reading |
| --- | --- |
| Both rules met | One adapter and one projection serve eight requests across two operations. The next stage can build on a single adapter. |
| One rule met, the other not | Either the operations interfere or this run was an unlucky one. One run cannot tell these apart: pilots 3 and 4 differed by 30 questions on this Noul panel from step order alone. The first follow-up is a joint text arm, to see whether joint training costs anything when the requests are words. |
| Neither met | Joint training in this form does not hold either format. Same follow-up. |

## Checks before training

Done on 11 November 2026 without the GPU (`joint_prepare.py`, which reads its output back; no model is run):

| Check | Result |
| --- | --- |
| Every training row equals its row in the separate arm's prepared file: tokens with pointers, where the adapter turns on, target, options, request vectors | all 8,422, each present once and trained once |
| The joint vector file equals the two existing files, end to end | equal; 6 Noul requests, then 2 Choice requests |
| The two selection hashes | unchanged (`0dffd3a6…`, `9df5422f…`) |
| The eight request vector sets differ | cosines from 0.57 to 0.85. The closest pair is the whole-rule request and its negation (0.85); no Noul request has a cosine above 0.66 with a Choice request |

Done on the GPU on 12 November 2026, before the run was started (`check_joint.sh`):

| Check | Result |
| --- | --- |
| FP32 loss, batched step against an independent per-case forward, one step per length, holding both operations where a length has both | worst relative gap 9.8e-06 |
| On-device loop against ordinary PyTorch, 6 steps | bit-identical |

The run itself is `run_joint.sh`: training, then the five measurements.

## Limits

- The option list already shows which operation a request is: true and false for Noul, the alternatives for Choice. So the vectors do not have to say which operation it is, only which request within it.
- Eight fixed requests. The vectors have to tell eight things apart, not carry a question's content.
- One run. A pass does not establish reliability across runs, and a failure does not establish interference.
- Score requests are not included. Score is still open from `fork-v1`.
- The pointers are given, not produced by the native side. The cases are synthetic.

## Cost

One training run of about 70 minutes and five measurements of about 5 minutes each on the RTX 3090. The two remaining checks take a few minutes more.

## A decision for the owner

Whether to train a joint text arm alongside, for about 75 more minutes. This design leaves it out: the thresholds are already fixed, and the text arm is only needed to read a failure.
