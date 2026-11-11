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

Interpretation, not measured: four span averages keep whether the question is about the whole rule or one prerequisite better than they keep which prerequisite is named.

## 16 vectors

The pass rule allows one further run with 16 vectors when rule 1 or 2 fails, and nothing else. It was started on 11 November 2026 at 13:44 with the same examples (selection hash `0dffd3a6…`), the same settings and one pass. Results will be added here.
