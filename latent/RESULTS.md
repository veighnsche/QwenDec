# Latent question pilot: results

Design and pass rule: [PILOT.md](PILOT.md), written before any pilot training. All figures are on the 1,182 development variants (116 states with eight questions each, plus states that keep one question and its negation). Per-case outputs are in [results/](results/).

## 4 vectors: did not pass

Run on 11 November 2026, 12:30 to 13:43. Both arms executed every planned step with none skipped, at 0.04% of one host core.

| Measure | Before pilot training (`fork-v1`, text) | Text arm | Latent arm, 4 vectors |
| --- | --- | --- | --- |
| All questions | 773/1182 (65%) | 1174/1182 (99%) | 985/1182 (83%) |
| Assessed twin pairs, both right | 41/428 (10%) | 427/428 (100%) | 337/428 (79%) |
| States with all eight right | 10/116 | 116/116 | 47/116 |
| Assessed questions | 469/856 | 855/856 | 760/856 (89%) |
| Insufficient recognised | 195/210 | 204/210 | 127/210 (60%) |
| Unresolved recognised | 49/56 | 55/56 | 38/56 (68%) |
| Conflicting recognised | 60/60 | 60/60 | 60/60 |

Against the pass rule:

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 1. Assessed twin pairs, both right | at least 385 (90% of 427) | 337 | No |
| 2. All-questions accuracy | within 5 points of 99% | 83%, 16 points below | No |
| 3a. Zero vectors: assessed twin pairs | below 10% | 0/428 | Yes |
| 3b. Twin's vectors swapped in: answer follows the swapped question | at least 90% of assessed | 760/856 (89%) | No, by 11 cases |

The text arm reached 427/428 assessed twin pairs, so the pilot is conclusive: the data and the training budget are enough for this task when the question is text.

### What the measurements show

- **The vectors carry the question.** With zero vectors no assessed twin pair is right. With the twin's vectors swapped in, the answer changes to the twin's answer in 760 of 856 assessed cases and stays with the original question in 73.
- **Rule 3b is capped by accuracy.** The 760 cases that follow the swapped question are the same count the latent arm gets right unswapped. The arm cannot follow a swapped question it would have answered wrongly anyway.
- **The loss is on single-prerequisite questions and on status.** By question kind, unnegated then negated: rule met 107 and 116 of 116, at least one prerequisite 111 and 116, first prerequisite 85 and 109, second prerequisite 87 and 64.
- **The latent arm had not converged.** The text arm's training loss reached 0.0004 by step 100. The latent arm's was 0.39 at its last step and still falling (0.52 at step 250, 0.45 at step 300).

## 16 vectors: did not pass

The pass rule allows one further run with 16 vectors when rule 1 or 2 fails, and nothing else. It ran on 11 November 2026, 13:44 to 14:26, with the same examples (selection hash `0dffd3a6…`), the same settings and one pass: 394 steps, none skipped.

| Measure | Text arm | 4 vectors | 16 vectors |
| --- | --- | --- | --- |
| All questions | 1174/1182 (99%) | 985/1182 (83%) | 1024/1182 (87%) |
| Assessed twin pairs, both right | 427/428 | 337/428 | 345/428 |
| States with all eight right | 116/116 | 47/116 | 56/116 |
| Rule met, at least one prerequisite (all four kinds) | 464/464 | 450/464 | 464/464 |
| First or second prerequisite (all four kinds) | 464/464 | 345/464 | 346/464 |
| Insufficient recognised | 204/210 | 127/210 | 189/210 |
| Unresolved recognised | 55/56 | 38/56 | 41/56 |

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 1. Assessed twin pairs, both right | at least 385 | 345 | No |
| 2. All-questions accuracy | within 5 points of 99% | 87%, 12 points below | No |
| 3a. Zero vectors: assessed twin pairs | below 10% | 0/428 | Yes |
| 3b. Twin's vectors swapped in | at least 90% of assessed | 734/856 (86%) | No |

The pilot is closed as not passed. Nothing further is retried under this rule.

## Where it fails

Both latent arms fail at one thing: telling the first prerequisite from the second.

Assessed single-prerequisite questions, split by what the state says about the two prerequisites (`analyse.py`, output in `results/analysis-prerequisites.json`):

| State | 4 vectors | 16 vectors |
| --- | --- | --- |
| Both prerequisites have the same answer | 224/224 | 224/224 |
| The two prerequisites differ | 36/72 | 35/72 |
| The other prerequisite is not stated | 64/84 | 15/84 |

When the two prerequisites differ, both arms are at chance. When the other prerequisite is not stated, the wrong answers are all "Insufficient", which is the right answer to the other prerequisite's question.

The cause is visible in the fixed encoder's output, before any training. The two questions differ in one word, "first" against "second". After removing the mean over all questions, the vectors for the two questions of one state have cosine similarity 0.98 (range 0.96 to 0.99, same for 4 and 16 vectors). Vectors for the same kind of question about different states have 0.5. Averaging hidden states over spans keeps the kind of question and the subject and nearly erases the one word that matters.

So the number of vectors was not the limit. More vectors fixed the questions about the whole rule (450 to 464 of 464) and left the single-prerequisite questions unchanged (345 to 346 of 464).

## What this does and does not show

- A question delivered as vectors does steer the decision: zero vectors and swapped vectors both change the answers as they should.
- This encoder, span averages of the frozen model's final states, loses a distinction the task needs. That is a finding about the encoder, not about vectors as an interface.
- Neither latent arm had converged after one pass (training loss 0.39 and 0.32 at the last step, against 0.0004 for text by step 100). Whether more training would separate vectors that are 0.98 alike was not tested.
- In the target design the request comes from the native model's own state, not from an encoder reading question text. This pilot does not test that.
