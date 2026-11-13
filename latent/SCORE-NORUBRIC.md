# Score pilot follow-up: a text arm without the rubric

Written 13 November 2026, after the Score pilot ([RESULTS-SCORE.md](RESULTS-SCORE.md)) and before this run. Approved by Vince the same day. Started at 12:28. Outcome: 59 of 134, so both contribute, the vectors far more than the rubric; see [RESULTS-SCORE.md](RESULTS-SCORE.md).

## Question

The Score pilot's text arm works out half of the in-between values and its latent arm hardly any. The text arm reads two things the latent arm does not: the rubric, and the question as words. Which of the two makes the difference?

## The arm

The text arm again, with one change: the question is the fixed wording as words and nothing else. No rubric.

Everything else is as in the pilot: from `runs/fork-v1/adapter`, the same 2,596 examples (selection hash `14aad566…`), one pass, the same settings, the same options. 393 steps.

So the three arms line up like this:

| Arm | Question | Rubric | Positions the question takes |
| --- | --- | --- | --- |
| Text | words | words | about 60 |
| Text, no rubric (this run) | words | none | about 25 |
| Latent | 16 vectors | none | 16 |

## Readings, stated in advance

On the 134 development questions with an in-between target, the text arm has 69 right and the latent arm 28.

| This arm gets | Reading |
| --- | --- |
| At least 63 (within 5 points of the text arm) | The rubric is not needed. The question as words is enough where 16 vectors are not, so the vector form is what falls short. |
| At most 34 (within 5 points of the latent arm) | The rubric is what lets the adapter work out a value. Leaving it in the state is what failed, and the vectors are not shown to be at fault. |
| 35 to 62 | Both contribute. |

Also reported: all questions, steering pairs, and the starting adapter reading the same no-rubric question before training.

One run, the final step's weights are the only candidate, nothing is retried.

## Limits

- If the rubric turns out to matter, this run cannot say whether it is what the rubric says or the extra positions it takes.
- The no-rubric question is still about 25 words against 16 vectors, so a result in the top row would not separate the form from the length.
- One run. The arms it is compared with are one run each.

## Checks before training

- Every question has the same state tokens and the same adapter start as the text arm, and no rubric in the question: all 2,994, checked without the GPU.
- The pilot's two prepared files are unchanged by the code change (hashes match).
- FP32 loss, batched step against an independent per-case forward: worst relative gap 2.1e-05. On-device loop against ordinary PyTorch, 6 steps: bit-identical. Both run on the GPU before training.
