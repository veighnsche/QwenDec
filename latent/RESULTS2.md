# Latent question pilot 2: results

Design and pass rule: [PILOT2.md](PILOT2.md), written before any pilot 2 training. Pilot 1: [RESULTS.md](RESULTS.md). All rule figures are on the 1,182 development variants. Per-case outputs are in [results/](results/).

## Neither arm passed

Run on 11 November 2026, 16:47 to 18:19, one run per arm. Both arms executed every planned step with none skipped, at 0.04% of one host core. Nothing was retried.

| Rule | Needed | Pointer as one vector | Pointer as replayed span |
| --- | --- | --- | --- |
| 1. Assessed twin pairs, both right | at least 385 of 428 | 384, not met by one pair | 407, met |
| 2. All questions | at least 1,115 of 1,182 | 983, not met | 1,104, not met by 11 |
| 3. Single-prerequisite questions where the two prerequisites differ | at least 65 of 72 | 36, not met | 72, met |
| 4. Pointer moved to the other prerequisite: answer follows it | at least 65 of 72 | 36, not met | 72, met |
| 5. Zero question vectors: assessed twin pairs | below 10% | 0/428, met | 0/428, met |
| **Verdict** | all five | **Not passed** | **Not passed** |

For reference: pilot 1's text arm had 1,174 and 427, its 16-vector arm 1,024 and 345.

## Where each arm fails

Only states that the rule parser can read have single-prerequisite questions and pointers. The other development states keep one question and its negation, delivered as 16 vectors with no pointer, exactly as in pilot 1 (`analyse2.py`, output in `results/analysis-pilot2.json`):

| Development questions | Text arm | 16 vectors (pilot 1) | One vector | Replayed span |
| --- | --- | --- | --- | --- |
| States that parse (928) | 928 | 810 | 798 | 927 |
| States that do not parse (254) | 246 | 214 | 185 | 177 |

### Replayed span: binding works, the miss is elsewhere

On states that parse, the replayed-span arm is at the text arm's level: 927/928, and all 72 differing-prerequisite questions right. Moving the pointer to the other prerequisite changed 240 of 464 answers, and in all 72 differing cases the answer became the other prerequisite's.

Its 77 errors on states that do not parse are what put it 11 questions below rule 2. Those questions carry no pointer. The arm does worse there than pilot 1's 16-vector arm (177 against 214). Most of that drop is on Insufficient: 12/42, against 33/42 in pilot 1 and 36/42 for text.

Most of those questions are worded in a way training never contains. All 8,276 training questions use the generated wording ("The eligibility rule is met for …", "Criterion: Both … and …"). In development, 184 of the 254 questions on states that do not parse are free-form, such as "Ana Popescu may use the rooftop greenhouse after hours this week", with a free-form criterion:

| Development, states that do not parse | Text arm | 4 vectors (pilot 1) | 16 vectors (pilot 1) | One vector | Replayed span |
| --- | --- | --- | --- | --- | --- |
| Templated wording (70) | 70 | 61 | 60 | 63 | 62 |
| Free-form wording (184) | 176 | 129 | 154 | 122 | 115 |

So 69 of the replayed-span arm's 78 errors are on free-form questions, delivered as vectors of a kind the arm never met in training. The text arm is not affected because the model already reads question text. Pilot 1's 16-vector arm carried more of them (154) than either pilot 2 arm.

The arm also fitted its training questions almost completely (mean training loss 0.0001 over the last 43 steps, against 0.32 for pilot 1's 16-vector arm). Whether that is why it carries fewer free-form questions than pilot 1 was not tested.

**A flaw in both pass rules.** The all-questions threshold counts these 184 questions, although no training question is free-form. On the other 998 questions the replayed-span arm has 989. The verdict stands as declared; the flaw is that rule 2 mixed "does the interface work" with "does it carry wordings it was never trained on", and the design did not say so.

### One vector: works on training names, not on development's

The one-vector arm answers every development single-prerequisite question as if it were about the first prerequisite (first 116/116, second 56/116). Moving the pointer changed no answer; the largest change in any probability was 0.04.

On the 150 training states with distinct names it does bind:

| 150 training states, distinct names | Text arm | 16 vectors (pilot 1) | One vector | Replayed span |
| --- | --- | --- | --- | --- |
| Whole-rule questions, held back | 308/308 | 301/308 | 299/308 | 308/308 |
| Single-prerequisite questions, held back | 292/292 | 180/292 | 290/292 | 292/292 |
| Single-prerequisite questions, trained on | 308/308 | 197/308 | 308/308 | 308/308 |
| Single assessed, the two prerequisites differ | 172/172 | 86/172 | 170/172 | 172/172 |

The pooled pointers explain the difference in part. A state's two pointer vectors have cosine similarity 0.88 in development (the names differ in one of five tokens) and at most 0.50 in training (median 0.18). All 192 training names were seen in training, though on other questions. So this pilot does not show whether one pooled vector fails on unseen names in general or only on near-identical ones. A later check with development's prerequisites renamed shows it is the near-identical ones: 72/72 on distinct unseen noun phrases ([RESULTS3.md](RESULTS3.md)).

## What this does and does not show

- A pointer to the item's tokens does bind a name-free question to the right item, where pilot 1's vectors alone were at chance. With the tokens replayed this holds on unseen, near-identical names (72/72) and follows the pointer when it is moved.
- One pooled vector per item is enough for names seen in training (170/172) and not for development's two names (36/72).
- The replayed-span arm still fails the rule, on questions that have no pointer. PILOT2.md's reading for "neither passes" was that question vectors and a pointed item do not combine. That reading does not fit: they combined on every state that has a pointer. What failed is mostly the delivery of free-form questions as vectors alone, a wording absent from training (115/184, against 154 in pilot 1 and 176 for text).
- Vince decided after these results that replaying the state's own tokens satisfies "never text". So the replayed span is an allowed way to hand over a pointer, and the open problem is free-form questions.
- As before: the pointer is given, not produced by the native side; Noul only; synthetic, templated cases.

Under the rule nothing further is retried. A third pilot with canonical requests is designed in [PILOT3.md](PILOT3.md).
