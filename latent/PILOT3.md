# Latent question pilot 3 (Noul only): design and pass rule

Written 11 November 2026, before any pilot 3 training. Approved by Vince the same day; started at 18:57. Outcome: did not pass, see [RESULTS3.md](RESULTS3.md).

## Question

Does a request made of a fixed kind of question plus a pointer work on every development state, including the free-form ones?

## Why this question

Pilot 2 ([RESULTS2.md](RESULTS2.md)) showed that a pointer whose tokens are replayed binds a question to the right item (72/72, and it follows the pointer when moved). It still missed its rule, because 184 development questions are free-form and no training question is. Their vectors restated a wording the arm had never met (115/184).

Vince decided two things after pilot 2: replaying a few of the state's own tokens counts as "never text", and the next design should keep requests canonical rather than add free-form questions to training.

This follows the build plan: the state carries the content, and a request only has to identify which question is asked and about what.

## What crosses

Every request is one of three fixed wordings, read by pilot 1's fixed encoder into 16 vectors. No name from the state appears in any of them.

| Kind | Wording read by the encoder | Pointer |
| --- | --- | --- |
| Whole rule | "What the supplied rule allows or defines holds in this case. Criterion: the supplied rule applied to the supplied facts." | none |
| At least one | "At least one prerequisite of the supplied rule holds. Criterion: either prerequisite alone is enough." | none |
| One prerequisite | "The indicated prerequisite holds. Criterion: the indicated prerequisite alone." | the prerequisite's name in the rule sentence, replayed after the vectors |

Each has a negated form ("Negation of: …"), so there are six distinct requests in all. Free-form questions use the whole-rule request: each free-form state has one outcome that its rules decide, and the question asks about it.

## Check already made, without training

The whole-rule wording was written after reading the development states. It was fixed before this check, measured once and not revised.

Pilot 1's text arm was trained on each question's own wording. Reading the canonical wording instead, with no further training (`check_canonical.py`, output in `results/check-canonical-text-breakdown.json`):

| Development questions | Own wording | Canonical wording |
| --- | --- | --- |
| All | 1,174/1,182 | 1,155/1,182 |
| Whole rule, free-form state (184) | 176 | 167 |
| Whole rule, templated state (282) | 282 | 282 |
| Whole rule, templated state whose prompt asks "is it FALSE that…" (20) | 20 | 10 |
| At least one prerequisite (232) | 232 | 232 |

So the canonical request says enough for nearly every question when it is delivered as text. The 20 questions on states whose own prompt asks the opposite are the exception: that prompt wording does not occur in training.

## Arm

One arm, from `runs/fork-v1/adapter`, on the same 4,468 examples (selection hash `0dffd3a6…`), one pass, learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels. These are pilot 2's settings; only the question vectors change.

References, all already measured: pilot 1's text arm (1,174/1,182, 427/428 assessed twin pairs), the canonical wording as text (above), and pilot 2's replayed-span arm (1,104/1,182).

## Checks before training

All passed on 11 November 2026, before the run was started (`check_canonical.py vectors`, `check_pointer.py`, `check.py`):

| Check | Result |
| --- | --- |
| Every question of both splits maps to one of six requests | 8,276 training and 1,182 development questions |
| The six vector sets differ | largest cosine between two requests 0.29 after removing the mean |
| Pointer ranges decode to the prerequisite's name | unchanged from pilot 2, all questions |
| FP32 loss, batched step against an independent per-case forward | worst relative gap 3.3e-06 |
| On-device loop against ordinary PyTorch, 6 steps | bit-identical |

## Pass rule, on the development variants

1. Assessed twin pairs with both right: at least 385 of 428.
2. All questions: at least 1,115 of 1,182.
3. Free-form questions: at least 151 of 184 (90% of the 167 the same request reaches as text).
4. Binding: single-prerequisite assessed questions in states whose two prerequisites differ: at least 65 of 72.
5. With the pointer moved to the other prerequisite, the answer follows it in at least 65 of those 72.
6. With zero question vectors, assessed twin pairs fall below 10%.

One run, the final step's weights are the only candidate, nothing is retried.

Also reported, outside the rule: each type of question in the table above, and the 600 held-back questions on 150 training states.

## What a pass would and would not show

- A pass would show that six fixed requests plus a replayed pointer are enough for this panel, free-form states included.
- It would not show that vectors from a fixed encoder can carry a question's content. With six distinct requests the vectors only have to tell six things apart, which a typed index could also do. That is a narrower claim than the plan's gate for this pilot.
- The whole-rule request points at nothing. It works when a state has one outcome its rules decide. A state with several would need a pointer to the proposition, and this data has no such case to test.

## Limits

- Same cases as before: synthetic, labelled by rule, and development has few distinct free-form states (92 states in 38 families).
- The pointer is given, not produced by the native side. Noul only.

## Cost

One training run of about 33 minutes and four measurements of about 5 minutes each on the RTX 3090.
