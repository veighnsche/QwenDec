# Latent Score request pilot: design and pass rule

Written 12 November 2026, before any Score request training. Not yet approved. Nothing is built and no model was run for this design; the figures below come from the case files and from per-case outputs that already existed.

## Question

Can a Score request be a fixed kind of question as vectors, with the rubric left in the state?

## What this pilot cannot settle

Score itself is still open. The accepted adapter `fork-v1` reads a Score request as words and still gets few in-between values right. On the 199 development Score cases, counting a case right when the status is right and the picked level is within 0.05 of the target:

| Target | Right |
| --- | --- |
| A status (80 cases) | 70 |
| An end of the scale, 0 or 1 (52) | 52 |
| An in-between value (67) | 11 |

So a request format can only be compared with a text arm that is itself weak on in-between values. This pilot asks whether the request can cross as vectors without losing what the text arm has. It does not ask whether Score works, and a pass would not show that it does.

## What a Score request is today

The adapter gets the question and the rubric as words: "What score between 0 and 1 applies? … Criterion: … Rubric: conditions, dimension, direction, one, zero." The options are eleven levels, 0.0 to 1.0, and the status options. The adapter answers with one option's letter.

## What the cases hold

From the case files, no model involved:

- **The rubric repeats the rule.** Every one of the 1,298 training states has one rule, and that rule defines the score, for example "Completion score equals completed bins divided by required bins. Zero means none complete; one means all required units complete." The rubric says the same thing again.
- **Nothing to point at.** Each state defines one score. The 20 development states with a second rule use it as a side condition or to state the ends of the scale, never as a second score. So a pointer would have nothing to tell apart.
- **Not every prompt asks for a score.** 333 training prompts ask for another task, such as listing the observations, and 136 describe two situations, with the listed facts saying which one is meant. Development has neither.
- **Training is narrow.** Two rule templates. 779 assessed cases: 518 at 0 or 1, 79 at 0.5, and 182 spread over 16 other values.
- **Development is not.** 199 cases with many rule wordings training never has (one minus a fraction, a count divided by twice the requirement, the largest shortfall), 46 distinct in-between values, and some cases where the rubric says things the state does not, for example the 12 whose rubric adds a midpoint.

## What crosses

One of two fixed requests, as 16 vectors from the fixed encoder. No rubric and no pointer.

| Kind | Fixed wording | Right answer |
| --- | --- | --- |
| Score | "What score between 0 and 1 applies under the supplied rule?" | the case's own status, or its value |
| Complement | "What is one minus the score that applies under the supplied rule?" | the same status, or one minus the value |

Each is followed by "Criterion: the supplied rule applied to the supplied facts."

The complement is derived by rule from each case's label. It is there for the same reason as the negation in Noul and the ruled-out kind in Choice: with one kind of request the list of levels alone says what is wanted, so an arm could ignore the vectors. The two kinds share a state and differ only in the vectors, and for most assessed cases no single level is right for both.

## Variants

| | Cases | Questions | Assessed pairs with no level right for both |
| --- | --- | --- | --- |
| Training | 1,298 | 2,596 | 690 of 779 |
| Development | 199 | 398 | 110 of 119 (52 at an end of the scale, 58 in between) |

The complement also widens training a little: 33 distinct in-between values instead of 17.

Of the 398 development questions, 264 have a status or an end of the scale as their answer, and 134 an in-between value.

## Arms

Both start from `runs/fork-v1/adapter`, on the same 2,596 examples, one pass, with the settings of the earlier pilots (learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels).

| Arm | Question | Rubric |
| --- | --- | --- |
| Text (reference) | the fixed wording as words | as words, the adapter's own format |
| Latent | the fixed wording as 16 vectors (two distinct requests) | not given; the state's rule holds it |

Options are the same in both: the eleven levels, then the three status options. An assessed target is trained as in `fork-v1`: the two levels around it, weighted so the expected level equals the target.

## References before training

To be measured on the GPU with `fork-v1`, no training, on the 199 development cases. They do not enter the rule; they say what the request carries.

| Request | Shows |
| --- | --- |
| Question and rubric as words | the starting point (133 of 199 with the tolerance below) |
| Question as words, no rubric | what leaving the rubric in the state costs |
| No question, no rubric | whether the request matters at all on these cases |
| The two fixed wordings as words, on the 398 variants | the starting point of the text arm |

## Pass rule for the latent arm, on the development variants

A question is right when the status is right and, if assessed, the picked level is within 0.05 of the target (of one minus the target for a complement). The text arm sets the thresholds, so they are stated as formulas.

0. The pilot is conclusive only if the text arm gets at least 238 of the 264 questions whose answer is a status or an end of the scale (90%).
1. All questions: within 5 points of the text arm (20 questions of 398).
2. The vectors steer: assessed pairs with no level right for both, both questions right: at least 90% of the text arm's count.
3. In-between values are not lost: the 134 questions with an in-between target: within 5 points of the text arm (7 questions).

One run per arm, the final step's weights are the only candidate, nothing is retried.

Stated in advance, outside the rule: in-between values count as working for an arm only if it gets at least half of those 134 questions right. Below that, the finding on Score values stands as it is, whatever the rule says.

Also reported: each status, the ends of the scale, mean distance to the target, the cases whose rule wording training never has, and the 12 cases whose rubric adds a midpoint.

## What each outcome would mean

| Outcome | Reading |
| --- | --- |
| Rule met, in-between values under half in both arms | The request format carries over to Score. Score values remain the open problem, and it is not a problem of the request. |
| Rule met, in-between values at least half | The format carries over and this training also repaired Score values. |
| Part 2 fails | The two requests are not told apart; the vectors are not steering. |
| Part 1 or 3 fails, part 2 met | Something the text arm reads is missing from the latent request. The rubric is the first suspect; the no-rubric reference says how much it carries. |
| Part 0 fails | The text arm is too weak to compare against. Score needs work before a request pilot means anything. |

## Checks before training

- The two requests' vector sets differ (cosine reported), and both differ from the eight existing ones.
- Every complement target mirrors its case's target, and statuses are unchanged.
- FP32 loss, batched step against an independent per-case forward, both arms.
- On-device loop against ordinary PyTorch, 6 steps: bit-identical, both arms.

## A correction to an earlier figure

The earlier Score measure compared the distance to the target with 0.05 exactly. For three development targets that lie midway between two levels, such as 0.75, rounding puts a right answer a hair over 0.05 and counts it wrong. With a tolerance for that, `fork-v1` has 133 of 199 on Score, not the 130 in the README. This pilot uses the tolerance.

## Limits

- No pointer. A request that says which of several quantities to score is not tested, because no case has more than one.
- The complement is an artificial request. It tests that the vectors are used, not that anyone would ask it.
- Two fixed requests only require the vectors to tell two things apart.
- Score only, from `fork-v1`. It is not trained alongside Noul and Choice.
- Training has two rule templates; development's other wordings are a test of the adapter's reading, not of the request.
- The cases are synthetic and labelled by rule.

## Cost

About 25 minutes of training per arm, a few minutes for each measurement and for the references: roughly one hour on the RTX 3090. Building the variants and the two arms needs no GPU.

## Decisions for the owner

1. **Whether to run this before Score values are repaired.** The pilot is worth running either way: the text arm is also a fresh Score-only training run, so it reports on the value problem at no extra cost. But the result on values will most likely stay poor, and the pass would then cover the request only.
2. **Whether a pointer should be tested.** It would need states with two things to score, made by joining two cases into one state. That changes the cases, not just the request, so it is left out here.
3. **Where to start from.** This design starts from `fork-v1`, like every pilot so far. Starting from the joint adapter would test Score on top of Noul and Choice, but that adapter is not accepted yet.
