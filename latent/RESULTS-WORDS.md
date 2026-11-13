# One adapter for Noul, Choice and Score, with fixed word-sequence requests: results

Design and pass rule: [WORDS.md](WORDS.md), pushed before the run. Earlier results: [RESULTS4.md](RESULTS4.md) (Noul), [RESULTS-CHOICE.md](RESULTS-CHOICE.md), [RESULTS-JOINT.md](RESULTS-JOINT.md) (Noul and Choice in one adapter, as vectors), [RESULTS-SCORE.md](RESULTS-SCORE.md). Per-case outputs are in [results/](results/); the breakdowns below come from `words_analyse.py` (output in `results/analysis-words.json`).

## Passed

Run on 13 November 2026, 13:52 to 15:54, measurements included, one run. All 1,210 steps executed with none skipped, at 0.04% of one host core. Nothing was retried.

One adapter, trained from `runs/fork-v1/adapter` on all three training sets together. Every request is one of ten fixed word sequences, plus pointers. There are no vectors and no projection.

| Part | Needed | Measured | Met |
| --- | --- | --- | --- |
| N1. Noul: assessed twin pairs, both right | at least 385 of 428 | 415 | Yes |
| N2. Noul: all questions | at least 1,115 of 1,182 | 1,141 | Yes |
| N3. Noul: free-form questions | at least 151 of 184 | 157 | Yes |
| N4. Noul: questions where the two prerequisites differ | at least 65 of 72 | 72 | Yes |
| N5. Noul: pointer moved, answer follows it | at least 65 of 72 | 72 | Yes |
| C1. Choice: all questions | at least 921 of 1,001 | 960 | Yes |
| C2. Choice: select and ruled-out pairs, both right | at least 331 of 374 | 358 | Yes |
| C3. Choice: assessed states right in every order | at least 93 of 112 | 109 | Yes |
| S1. Score: all questions | at least 291 of 398 | 301 | Yes |
| S2. Score: steering pairs, both right | at least 59 of 110 | 62 | Yes |
| S3. Score: in-between values | at least 63 of 134 | 65 | Yes |

All eleven parts are met. The Score parts are met narrowly: S3 by 2 questions, S2 by 3, S1 by 10. Noul's free-form part is met by 6.

**In-between Score values still do not count as working.** The threshold stated in advance was 67 of 134; this run has 65.

## Against the earlier arms

**Noul**, renamed panel:

| Questions | Text arm | Vectors, Noul only | Vectors, joint adapter | Words, three operations |
| --- | --- | --- | --- | --- |
| All (1,182) | 1,174 | 1,162 | 1,128 | 1,141 |
| Assessed twin pairs (428) | 427 | 425 | 406 | 415 |
| Whole rule, free-form state (184) | 176 | 165 | 153 | 157 |
| Whole rule, templated state (282) | 282 | 281 | 272 | 278 |
| Prompt asks "is it FALSE that…" (20) | 20 | 20 | 7 | 10 |
| Single prerequisite (464) | 464 | 464 | 464 | 464 |

**Choice:**

| Questions | Text arm | Vectors, Choice only | Vectors, joint adapter | Words, three operations |
| --- | --- | --- | --- | --- |
| All (1,001) | 971 | 955 | 929 | 960 |
| Select and ruled-out pairs (374) | 367 | 352 | 342 | 358 |
| Assessed states right in every order (112) | 103 | 109 | 105 | 109 |
| Select, no alternative is feasible (100) | 100 | 90 | 91 | 93 |
| Alternatives with other names (407) | 377 | 364 | 341 | 366 |

**Score:**

| Questions | Text arm with rubric | Words, Score only | Vectors, Score only | Words, three operations |
| --- | --- | --- | --- | --- |
| All (398) | 310 | 294 | 254 | 301 |
| Answer is a status or an end of the scale (264) | 241 | 235 | 226 | 236 |
| Answer is in between (134) | 69 | 59 | 28 | 65 |
| In between, within 0.1 | 101 | 90 | 46 | 91 |
| Steering pairs, both right (110) | 65 | 66 | 47 | 62 |

- **At or above the vector joint adapter on every Noul and Choice measure**, with a third operation added.
- **Choice is at the level of the Choice-only vector arm or above it** (960 against 955), and close to the text arm that reads the properties as words.
- **Noul is below the Noul-only vector arm** (1,141 against 1,162). The loss is where it was in the joint vector run: free-form states whose answer is Unresolved (7 of 26) and prompts that ask "is it FALSE that…" (10 of 20).
- **Score is no worse for being trained with the other two.** It is between the Score-only word run and the text arm that reads the rubric.

## Reported outside the rule

- **Noul on development's own near-identical names:** 1,143 of 1,182, 417 of 428 twin pairs, every single-prerequisite question right.
- **Noul on training states with distinct names**, half of the questions held back from training: 1,199 of 1,200.
- **Choice, select questions in listed order:** 216 of 230. Every question with single-letter names is right (594 of 594).
- **Score by rule wording:** in-between values 29 of 54 where the rule is worded as in training, 36 of 80 where it is not. Statuses 137 of 160.
- **Training loss** falls below 0.04 by step 200 and stays between 0.02 and 0.05 from there. It does not reach zero because an in-between Score target is spread over two levels.

## What this does and does not show

- One adapter can take all three kinds of request in one form: a fixed word sequence picked by the kind of request, plus pointers into the state. The fixed encoder and the trained projection are not needed for this.
- Pointing is intact: every single-prerequisite Noul question is right and the answer follows a moved pointer in all 72 cases; Choice holds its answer in every order on 109 of 112 states.
- The run cannot say why it does better than the vector joint adapter. Two things changed together: the form of the request, and Score being added. It is also one run, and runs of one design have differed by more than these gaps.
- The Score parts were met by 2, 3 and 10 questions. A different step order could put them under.
- Score values are improved, not solved: just under half of the in-between values are within 0.05 and two thirds within 0.1.
- Ten fixed sequences are a menu. Nothing here shows that a request can carry new content.
- The pointers are given, not produced by the native side. The Score complement is an artificial request. The cases are synthetic.

## Accepted

Vince accepted this adapter and its request form on 13 November 2026 as the basis for the next stage: one adapter for Noul, Choice and Score, each request a fixed word sequence picked by the kind of request, plus replayed pointers.

The acceptance leaves open the reliability across runs and the in-between Score values. It does not cover a request or a pointer produced by the native side. The open decision on the vector joint adapter ([RESULTS-JOINT.md](RESULTS-JOINT.md)) falls away with it; that adapter was never accepted (`runs/words/result.json`).
