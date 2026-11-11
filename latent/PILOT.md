# Latent question pilot (Noul only): design and pass rule

Written 11 November 2026, before any pilot training.

## Question

Can the decision adapter decide a Noul when the question arrives as a few vectors instead of text?

## Why new question variants were needed

In the imported cases every state has exactly one question and the state implies it. A model could pass without reading the question. So each state with two named prerequisites and yes/no facts gets eight questions whose answers differ: the rule is met, at least one prerequisite holds, the first holds, the second holds, and the negation of each. Labels follow from the stated facts by exact rule (`cases.py`); no model wrote or reviewed them. The state text is never changed.

## References measured before training (development variants, adapter `fork-v1`)

| Question given as | All questions | Assessed twin pairs, both right |
| --- | --- | --- |
| Text | 773/1182 (65%) | 41/428 (10%) |
| Nothing | 646/1182 (55%) | 0/428 (0%) |

With text, the plain variants are all right (116/116 each) and the negated ones mostly wrong: the adapter reads the question's content but not the negation.

## Arms

Both start from `runs/fork-v1/adapter`, use the same 4,468 examples (selection hash `0dffd3a6…`), one pass, learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels.

| Arm | Question slot | Trained |
| --- | --- | --- |
| Text | the question as words | LoRA |
| Latent, 4 vectors | 4 positions holding vectors from the fixed encoder | LoRA and one 2560 x 2560 projection, initialised to identity |

The fixed encoder is the frozen base model reading the question text alone: final hidden states averaged over 4 equal spans, rescaled to typical embedding length. It never sees a label, a state or a decision and is never updated.

## Pass rule for the latent arm, on the development variants

1. Assessed twin pairs with both right: at least 90% of the text arm's count.
2. All-questions accuracy: within 5 points of the text arm.
3. The vectors are used: with zero vectors, assessed twin pairs fall below 10%; with the twin question's vectors swapped in, the answer follows the swapped question in at least 90% of assessed cases.

If the text arm itself does not reach 90% of assessed twin pairs, the pilot is inconclusive about the interface, because the data or budget is then the limit.

If rule 1 or 2 fails, one further run with 16 vectors is allowed. Nothing else is retried, and the final step's weights are the only candidate.
