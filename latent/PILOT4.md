# Latent question pilot 4 (Noul only): design and pass rule

Written 11 November 2026, before the pilot 4 run. Vince decided the same day that near-identical item names are reported separately instead of being part of the pass rule; this run follows from that decision. Started at 20:34. Outcome: passed, see [RESULTS4.md](RESULTS4.md).

## Question

Does pilot 3's design pass when it is run afresh and judged on a panel whose item names are distinct?

## Why a fresh run

Pilot 3 ([RESULTS3.md](RESULTS3.md)) failed its rule on one thing: negated single-prerequisite questions about development's two prerequisite names, which differ in a single word. A check afterwards showed the same arm answers those questions when the names are distinct.

That check was made after the run, on the same arm. Re-judging that arm on a friendlier panel would not be a test. So this pilot changes two things that have not been seen:

- **A new training run.** Same design, examples and settings as pilot 3, with the steps in a different order (order seed 4). Training here is deterministic, so without a new order the run would reproduce pilot 3's arm exactly.
- **A new panel.** Development with the prerequisites of the states that parse renamed to twelve pairs of names used nowhere before: not in training and not in the earlier check (`rename.py`, version `panel4`). Six pairs are noun phrases and six are clauses. States that do not parse are unchanged, so the 184 free-form questions and the 20 "is it FALSE that" questions are still in it.

## Arm

As pilot 3: six fixed requests as 16 vectors each, the pointer replayed after them, from `runs/fork-v1/adapter`, the same 4,468 examples (selection hash `0dffd3a6…`), one pass, learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels.

Reference: pilot 1's text arm measured on the new panel, before this run: 1,174/1,182, 427/428 assessed twin pairs, 72/72 where the prerequisites differ. These are the same figures as on the original panel.

## Checks before the run

All passed on 11 November 2026 (`rename.py`, `check.py`):

| Check | Result |
| --- | --- |
| No new name occurs in training or in the earlier check | all 24 names |
| Pointer ranges decode to the new names | all 464 questions with a pointer, both prerequisites |
| Same examples as pilot 3, different step order | selection hash identical, order differs |
| FP32 loss, batched step against an independent per-case forward | worst relative gap 6.6e-06 |
| On-device loop against ordinary PyTorch, 6 steps | bit-identical |

## Pass rule, on the new panel

1. Assessed twin pairs with both right: at least 385 of 428.
2. All questions: at least 1,115 of 1,182.
3. Free-form questions: at least 151 of 184.
4. Binding: single-prerequisite assessed questions in states whose two prerequisites differ: at least 65 of 72.
5. With the pointer moved to the other prerequisite, the answer follows it in at least 65 of those 72.
6. With zero question vectors, assessed twin pairs fall below 10%.

One run, the final step's weights are the only candidate, nothing is retried.

## Reported separately, outside the rule

- **Near-identical names.** The same arm on development as it is, where the two prerequisites differ in one word: all questions, assessed twin pairs, and single-prerequisite questions plain and negated.
- The 600 held-back questions on 150 training states.
- Each type of question, as in pilot 3.

## What a pass would and would not show

- A pass would show that six fixed requests plus a replayed pointer are enough on this panel, for free-form states and for item names never seen in training, in a second run of the design.
- It would not cover items whose names differ in one word. That case is measured and reported, not required.
- As in pilot 3, six requests only require the vectors to tell six things apart. A pass would not show that fixed-encoder vectors can carry a question's content.
- The panel was built after seeing where pilot 3 failed. The names are new and the run is new, but the choice to leave near-identical names out of the rule was made with that knowledge.

## Cost

One training run of about 33 minutes and five measurements of about 5 minutes each on the RTX 3090.
