# Latent question pilot 2 (Noul only): design and pass rule

Written 11 November 2026, before any pilot 2 training. Not yet approved to run.

## Question

Can a request say which item in the state it is about, without the question being text?

## Why this question

Pilot 1 ([RESULTS.md](RESULTS.md)) showed that 16 vectors from a fixed encoder carry the kind of question: whole-rule questions were 464/464. They did not bind a named item to the state: when the two prerequisites of a state differ, the arm was at chance (35/72 on development, 86/172 on training states with distinct names).

A different pooling of the frozen model's states does not change this. Measured without training, no candidate separates the one-word pair by more than 0.33 of the typical distance between questions, and the arm failed even where the vectors were far apart. So this pilot does not try another encoder for the name.

It tests the split the build plan describes: the vectors say which question, and a pointer says which item. The state already holds the item's name, so the request only has to point at it.

## What crosses

For a single-prerequisite question the request is:

- **16 question vectors**, from pilot 1's fixed encoder reading a question with no name in it: "Is this proposition true? The indicated prerequisite holds for {target}. Criterion: the indicated prerequisite alone." Negated questions add "Negation of:" as before.
- **One pointer**: the token range of the prerequisite's name where the rule sentence mentions it. Every parsed state has that mention, whether or not a fact about it is stated.

Whole-rule questions and states that do not parse carry no pointer. Their questions and vectors are as in pilot 1.

## Arms

Both start from `runs/fork-v1/adapter`, use the same 4,468 examples (selection hash `0dffd3a6…`), one pass, learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels. The only difference is how the pointer is handed to the decision fork.

| Arm | Pointer handed over as | Positions |
| --- | --- | --- |
| One vector | The mean input embedding of the pointed tokens, not passed through the trained projection | 16 + 1 |
| Replayed span | The pointed tokens themselves, read again after the question vectors | 16 + the span's length (2 to 9 tokens, median 5) |

The reference is pilot 1's text arm (same examples and settings): 1174/1182, 427/428 assessed twin pairs, 72/72 on the differing-prerequisite questions.

Neither pointer form can leak the answer. Both are built from token identities only, never from the model's states at the mention, which could already reflect the facts read before it.

**A decision for the owner.** The replayed-span arm puts a few of the state's own tokens in the fork. The request itself still carries only a range, and no text is generated. Whether that satisfies "requests cross as vectors, never text" is a judgement, not a measurement. If it does not, only the one-vector arm counts.

## Checks before training

- Every pointer range decodes to exactly the prerequisite's name, in every parsed state of both splits.
- Single-prerequisite questions about the two prerequisites of one state have identical question vectors, so only the pointer can tell them apart.
- The on-device loop is bit-identical to ordinary PyTorch on the new layout, and the FP32 loss check passes, as in pilot 1.

## Pass rule for each arm, on the development variants

1. Assessed twin pairs with both right: at least 385 of 428 (90% of the text arm's 427).
2. All-questions accuracy: at least 1,115 of 1,182 (within 5 points of the text arm).
3. Binding: single-prerequisite assessed questions in states whose two prerequisites differ: at least 65 of 72.
4. The pointer does the binding: with the pointer moved to the other prerequisite, the answer follows the pointer in at least 65 of those 72 cases.
5. The vectors are still used: with zero question vectors, assessed twin pairs fall below 10%.

Removing the pointer is not a test here: the two questions then have identical inputs, so exactly half are right by construction.

Each arm is judged on its own. One run per arm, the final step's weights are the only candidate, and nothing is retried. If both pass, the one-vector pointer is preferred because it costs one position.

Also reported, outside the rule: the 600 held-back questions on 150 training states with distinct names, where pilot 1's 16-vector arm had 180/292 on single-prerequisite questions.

## What each outcome would mean

| Outcome | Reading |
| --- | --- |
| One vector passes | A request can be a few question vectors plus one pointer vector per item. |
| Only replayed span passes | Pointing works, and the fork needs the item's tokens to find it. One pooled vector is not enough. |
| Neither passes | Question vectors and a pointed item do not combine in one pass of training. The interface needs more than a redesign of the request. |

## Limits

- Development has only two prerequisite names, differing in one word, and neither occurs in training. That is a hard case for a pooled pointer and says little about names in general; the held-back training questions cover distinct names.
- One pointer per request. Requests about several items, and Choice or Score, are not covered.
- The pointer is given, not produced. The native side learning where to point is a later stage.
- Same cases as before: synthetic, templated, labelled by rule.

## Cost

Two training runs of about 33 minutes each and six measurements of about 5 minutes each on the RTX 3090.
