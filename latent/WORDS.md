# One adapter for Noul, Choice and Score, with fixed word-sequence requests: design and pass rule

Written 13 November 2026, before any training for it; the token counts come from the tokenizer alone. Vince decided the same day that part S3 stays in the rule. The rows and scripts were then built without the GPU. The two GPU checks and the run have not been done and wait for his go-ahead.

## Question

Can one decision adapter serve all three operations when every request is a fixed word sequence, picked by the kind of request, plus pointers into the state?

## Why this question

- **Score did not carry over as vectors.** With two fixed requests as 16 vectors the adapter worked out few in-between values (28 of 134). With the same two requests as words it got 59 ([RESULTS-SCORE.md](RESULTS-SCORE.md)).
- **Fixed word sequences are allowed.** Vince decided on 13 November that a fixed word sequence per kind of request, never generated, is within "requests cross as vectors, never text".
- **One mechanism is simpler than two.** If words work for all three operations, the fixed encoder and the trained projection are no longer needed, and the build plan's single decision adapter has one request form.

Noul and Choice passed with vectors, separately and in one adapter ([RESULTS-JOINT.md](RESULTS-JOINT.md)). They have never been trained with word sequences and pointers. Score has never been trained alongside them.

## The request

A request is one of ten fixed token sequences. Each is the same token ids every time and is inserted as it is, where the 16 vectors went before.

| Operation | Fixed wording | Tokens |
| --- | --- | --- |
| Noul | Is this proposition true? What the supplied rule allows or defines holds in this case. Criterion: the supplied rule applied to the supplied facts. | 28 |
| Noul | the same, with "Negation of:" | 32 |
| Noul | Is this proposition true? At least one prerequisite of the supplied rule holds. Criterion: either prerequisite alone is enough. | 23 |
| Noul | the same, with "Negation of:" | 27 |
| Noul | Is this proposition true? The indicated prerequisite holds. Criterion: the indicated prerequisite alone. | 17 |
| Noul | the same, with "Negation of:" | 21 |
| Choice | Which of the indicated alternatives should be selected under the supplied rule? Criterion: the supplied rule applied to the supplied facts. | 24 |
| Choice | Which of the indicated alternatives does the supplied rule rule out? Criterion: … | 23 |
| Score | What score between 0 and 1 applies under the supplied rule? Criterion: … | 25 |
| Score | What is one minus the score that applies under the supplied rule? Criterion: … | 24 |

Pointers are as accepted: for a single-prerequisite Noul question the prerequisite's tokens from the state are replayed after the sequence; for Choice each alternative is the replayed tokens of its name; Score has none. Properties and rubrics stay in the state.

These are the wordings the fixed encoder read in the earlier pilots. Here the adapter reads them directly.

## The run

One arm.

- **Start:** `runs/fork-v1/adapter`. Only the adapter is trained; there are no vectors and no projection.
- **Examples:** the three training sets, unchanged: 4,468 Noul (selection hash `0dffd3a6…`), 3,954 Choice (`9df5422f…`), 2,596 Score (`14aad566…`). 11,018 examples, one pass.
- **How the rows are made:** each row of the three existing prepared files is taken as it is, and its 16 vector positions are replaced by the fixed sequence. State, pointers, options and targets are not derived again.
- **Batches:** all three operations grouped by length together, steps in a new seeded order (7). 1,210 steps: 641 hold two or three operations, 307 only Noul, 195 only Score and 67 only Choice. Most Noul prompts are shorter than the others, and most Score prompts longer, which is why many steps hold one operation.
- **Settings:** as in every pilot: learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels, the on-device loop.

## Pass rule

Each operation is judged by the rule its request format was judged by, with fixed numbers. All eleven parts must be met.

**Noul**, on the renamed development panel (1,182 questions):

| Part | Needed | Text arm | Vectors, Noul only | Vectors, joint adapter |
| --- | --- | --- | --- | --- |
| N1. Assessed twin pairs, both right | at least 385 of 428 | 427 | 425 | 406 |
| N2. All questions | at least 1,115 of 1,182 | 1,174 | 1,162 | 1,128 |
| N3. Free-form questions | at least 151 of 184 | 176 | 165 | 153 |
| N4. Questions where the two prerequisites differ | at least 65 of 72 | | 72 | 72 |
| N5. Pointer moved: answer follows it | at least 65 of 72 | | 72 | 72 |

**Choice**, on the development variants (1,001 questions):

| Part | Needed | Text arm | Vectors, Choice only | Vectors, joint adapter |
| --- | --- | --- | --- | --- |
| C1. All questions | at least 921 of 1,001 | 971 | 955 | 929 |
| C2. Select and ruled-out pairs, both right | at least 331 of 374 | 367 | 352 | 342 |
| C3. Assessed states right in every order | at least 93 of 112 | 103 | 109 | 105 |

**Score**, on the development variants (398 questions), the Score pilot's rule with its numbers now fixed:

| Part | Needed | Text arm with rubric | Words, no rubric, Score only | Vectors, Score only |
| --- | --- | --- | --- | --- |
| S1. All questions | at least 291 of 398 | 310 | 294 | 254 |
| S2. Steering pairs, both right | at least 59 of 110 | 65 | 66 | 47 |
| S3. In-between values | at least 63 of 134 | 69 | 59 | 28 |

One run, the final step's weights are the only candidate, nothing is retried.

**Part S3 is the hard one.** The Score-only run in this same word form got 59, four short. For the rule to be met, this run has to do slightly better on in-between values than that one did. Whether S3 belongs in the rule or is reported beside it was put to the owner before the run. Decided by Vince on 13 November 2026: S3 stays in the rule.

Stated in advance, outside the rule: in-between values count as working only at 67 of 134 or more.

Reported outside the rule: every measure against the joint vector adapter; Noul on development's own names and on training states with distinct names; Choice where no alternative is feasible and by kind of name; Score by status, by rule wording, and distance to the target.

## What each outcome would mean

| Outcome | Reading |
| --- | --- |
| All parts met | One adapter, one request form, three operations. The fixed encoder and the projection can be dropped. |
| Noul and Choice met, only S3 missed | The word form holds for the two accepted operations and for telling Score requests apart. Score values remain the open problem, as they were before any request work. |
| Noul or Choice not met | Either words are a worse form for that operation than vectors, or adding Score cost it, or this was an unlucky run. This run cannot tell these apart, because the form and the third operation change together. The vector joint adapter is the comparison, and a words-only Noul and Choice run would be the follow-up. |
| S1 or S2 not met | Score suffers from being trained with the others; the Score-only run met both. |

## Checks before training

Done on 13 November 2026 without the GPU (`words_prepare.py`, which reads its output back, and the tokenizer; no model is run):

| Check | Result |
| --- | --- |
| Every request position holds exactly its fixed sequence | all 11,018 rows |
| Outside the request, every row equals its source row: state, pointers, options, target | all 11,018, each present once and trained once |
| Every training row is what the measurement scripts build in word mode for a training question | all 11,018 |
| The three source files are the ones the vector arms trained on | hashes match |

Still to do on the GPU before training (`run_words.sh check`):

- FP32 loss, batched step against an independent per-case forward, one step per prompt length, the one holding the most operations.
- On-device loop against ordinary PyTorch, 6 steps: bit-identical.

The run itself is `run_words.sh`: training, then the six measurements.

## Limits

- Ten fixed sequences are a menu. The adapter can learn each as a label for a task; this says nothing about a request that carries new content.
- A request takes 17 to 32 positions instead of 16.
- Two things change together against the vector joint run: the form of the request, and Score being added. A difference on Noul or Choice cannot be put on one of them.
- The Score-only run in this form tokenised its question in place. A fixed sequence differs from that by one token at its end.
- One run. Runs of one design have differed widely on cases unlike training.
- The Score complement is an artificial request. The pointers are given, not produced by the native side. The cases are synthetic.

## Cost

1,210 steps, about 100 minutes of training, six measurements of 3 to 5 minutes each, and the two GPU checks: roughly two and a quarter hours on the RTX 3090. Building the rows and scripts needs no GPU.

## Decisions for the owner

1. **Is S3 part of the rule?** This design puts it in, as the Score pilot declared it. The alternative is to judge Score on S1 and S2 and report in-between values beside the rule, as was done for near-identical names in pilot 4. Decided: it stays in.
2. **What this run would replace.** If it passes, the open decision on the vector joint adapter falls away: this adapter would be the candidate for the next stage instead.
