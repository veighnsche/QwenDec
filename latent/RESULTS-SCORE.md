# Latent Score request pilot: results

Design and pass rule: [SCORE.md](SCORE.md), pushed before the runs. Earlier pilots: [RESULTS4.md](RESULTS4.md) for Noul, [RESULTS-CHOICE.md](RESULTS-CHOICE.md) for Choice. Per-case outputs are in [results/](results/); the breakdowns below come from `score_analyse.py` (output in `results/analysis-score.json`).

## Not passed

Run on 13 November 2026, one run per arm: references 10:59 to 11:03, text arm to 11:42, latent arm to 12:18, measurements included. Every planned step executed with none skipped, at 0.04% of one host core. Nothing was retried.

The panel is 398 development questions on 199 cases: each case asked for its score and for one minus its score.

| Rule | Needed | Measured | Met |
| --- | --- | --- | --- |
| 0. Conclusive: text arm, questions whose answer is a status or an end of the scale | at least 238 of 264 | 241 | Yes |
| 1. All questions | at least 291 of 398 (text arm less 5 points) | 254 | No |
| 2. Steering pairs, both right | at least 59 of 110 (90% of the text arm's 65) | 47 | No |
| 3. In-between values | at least 63 of 134 (text arm less 5 points) | 28 | No |

The latent request is 16 vectors for the kind of question, with no rubric. As a Score request it does not carry over.

## The three arms side by side

"Starting adapter" is `fork-v1` before any training here, reading the fixed wordings and the rubric as text.

| Questions | Starting adapter, text | Text arm | Latent arm |
| --- | --- | --- | --- |
| All (398) | 217 | 310 | 254 |
| Answer is a status (160) | 141 | 142 | 132 |
| Answer is an end of the scale (104) | 55 | 99 | 94 |
| Answer is in between (134) | 21 | 69 | 28 |
| Steering pairs at the ends, both right (52) | 3 | 47 | 42 |
| Steering pairs in between, both right (58) | 3 | 18 | 5 |
| In-between answers: mean distance to the target | 0.29 | 0.09 | 0.25 |

## What the latent arm does and does not do

- **It tells the two requests apart.** At the ends of the scale it answers the complement question right in 42 of 52 cases; the starting adapter has 3. In between, its two answers on one state differ in 56 of 67 pairs and add up to 1, within 0.1, in 41.
- **It does not work out in-between values.** 28 of 134, hardly above the starting adapter's 21. Its picks are nearly as far from the target as before training (0.25 against 0.29).
- **This holds even on rules worded as in training:** 10 of 54 in-between questions there, against 31 for the text arm.

So the predeclared reading of part 2, that the vectors are not steering, does not fit what was measured. Part 2 fails because both answers of a pair must be right, and in between they are not. The failure is in the value, not in telling the requests apart.

## The text arm repaired a good part of Score

This was not expected. `fork-v1` reading its own Score question gets 11 of 67 in-between cases. The text arm, trained here on Score alone with the complement questions added, gets:

| In-between questions (134) | Starting adapter, text | Text arm |
| --- | --- | --- |
| Within 0.05 of the target | 21 | 69 |
| Within 0.1 | 45 | 101 |
| Within 0.2 | 65 | 115 |
| Rule worded as in training, within 0.05 (54) | 9 | 31 |
| Rule worded unlike training, within 0.05 (80) | 12 | 38 |

By the threshold stated in advance, at least half of the 134, in-between values count as working for the text arm. It clears that by two questions, on one run, so it is a start and not a settled result.

## Why the two arms differ is not known

The text arm reads two things the latent arm does not: the rubric as words, and the question as words. Either could be what lets it work out a value.

- **The rubric.** It restates the rule, for example "completed bins divided by required bins", inside the part of the prompt where the adapter is on. The state itself is read with the adapter off.
- **The words.** The question and rubric take about 60 positions with the adapter on, where the latent request has 16.

Before training, neither mattered: `fork-v1` on the 199 cases has 133 with question and rubric, 130 without the rubric and 131 without any question (`results/score-refs.json`). After training they matter a great deal. One further arm would separate the two: the fixed wording as words, without the rubric. That arm was run next; see the follow-up below.

## Follow-up: a text arm without the rubric

Design and readings: [SCORE-NORUBRIC.md](SCORE-NORUBRIC.md), pushed before the run. Run on 13 November 2026, 12:28 to 13:06, one run, all 393 steps executed, nothing retried.

The arm reads the fixed wording as words and no rubric. It differs from the latent arm only in how the same two requests arrive: about 25 word positions instead of 16 vectors.

| Questions | Text arm | Text arm, no rubric | Latent arm |
| --- | --- | --- | --- |
| All (398) | 310 | 294 | 254 |
| Answer is a status (160) | 142 | 136 | 132 |
| Answer is an end of the scale (104) | 99 | 99 | 94 |
| Answer is in between (134) | 69 | 59 | 28 |
| In between, within 0.1 | 101 | 90 | 46 |
| In between, rule worded as in training (54) | 31 | 29 | 10 |
| In between, rule worded unlike training (80) | 38 | 30 | 18 |
| Steering pairs, both right (110) | 65 | 66 | 47 |

**Reading, as stated in advance:** 59 lies between 35 and 62, so both contribute.

They do not contribute equally. Of the 41 in-between questions between the text arm and the latent arm, 10 go with the rubric and 31 with the question arriving as words instead of vectors.

- **The rubric helps where the rule is unlike training** (38 against 30) and hardly at all where it is like training (31 against 29).
- **Most of the gap is the vectors.** With the same two requests and no rubric in either, words get 59 in-between values and vectors 28.
- Measured against the pilot's own rule, the no-rubric text arm would meet parts 1 and 2 (294 against 291 needed, 66 against 59) and miss part 3 by four questions (59 against 63).

Why vectors do worse here, when the same form passed for Noul and Choice, is not shown by this run. Two things differ between the arms and are not separated: the number of positions (25 against 16), and that the latent arm has to learn to read its vectors through a new projection in the same single pass. Its loss starts far higher (1.35 over the first 50 steps, against 0.23) and is still higher at the end (0.09 against 0.08).

## A question for the owner

The no-rubric text arm's question is not generated text. It is one of two fixed word sequences, chosen by the kind of request. A runtime could insert the fixed sequence for the chosen kind exactly as it inserts 16 vectors. Whether that is within "requests cross as vectors, never text" is a judgement, like the replayed pointer was.

## Reported outside the rule

- **Statuses:** latent arm 132 of 160, text arm 142. Unresolved is the weakest in both (34 and 38 of 52).
- **The 24 questions whose rubric adds a midpoint:** 12 for the latent arm, 9 for the text arm.
- **Arms compared, in-between questions:** both right 23, only the text arm 46, only the latent arm 5, both wrong 60.
- **Training loss** ends at 0.07 (text) and 0.09 (latent). It does not go to zero because an in-between target is spread over two levels.

## What this does and does not show

- A Score request as a fixed kind of question in 16 vectors, with the rubric left in the state, is not enough. The follow-up puts most of the shortfall on the 16 vectors, not on the missing rubric. This is the first of the three operations where the accepted format does not carry over.
- The vectors still do their part: the two requests are told apart.
- Score values are not out of reach for this adapter. A Score-only text run got half of the in-between values within 0.05 and three quarters within 0.1.
- One run per arm. The earlier pilots showed that two runs of one design can differ widely.
- Score only, from `fork-v1`; two fixed requests; no pointer; synthetic cases.
