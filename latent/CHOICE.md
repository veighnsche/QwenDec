# Latent Choice request pilot: design and pass rule

Written 11 November 2026, before any Choice training. Not approved or run yet.

## Question

Can a Choice request be a fixed kind of question as vectors plus one pointer per alternative, with the alternatives' properties left in the state?

## What a Choice request is today

The decision adapter gets the question as words and a list of options. Each alternative is an option with its name and its properties written out, for example "Birch: seats 8, designated quiet". The status options follow. The adapter answers with one option's letter.

In the format accepted for Noul ([RESULTS4.md](RESULTS4.md)), the question becomes a fixed request kind as 16 vectors, and each alternative becomes a pointer: the tokens of its name in the state, replayed as the option. The properties are not restated. They are in the state.

## What the unchanged cases do not test

Measured without training, on the 232 development Choice cases with the accepted adapter `fork-v1` (`choice_refs.py`, output in `results/choice-refs.json`). Right means the status is right and, if assessed, an acceptable alternative is picked.

| Request | Listed order | Reversed order |
| --- | --- | --- |
| Question as words, names and properties | 225 | 224 |
| Question as words, names only | 225 | 194 |
| No question, names and properties | 221 | 221 |
| No question, names only | 216 | 183 |

Two things follow.

- **The question hardly matters.** Each state's own prompt already asks it, so an arm could ignore the request vectors and still score 221. The cases need a second kind of question on the same state.
- **Names alone bind poorly when the order changes.** With properties the adapter matches an option to its facts. With names only it partly goes by position (assessed cases 84/114 when reversed). This is what a pointer has to fix, and it is the hard part of this pilot.

## Variants, derived by rule

Every case has a reviewed `feasibility` label per alternative. From it:

| Kind | Asks | Right answer | Defined for |
| --- | --- | --- | --- |
| Select | which indicated alternative to select under the supplied rule | the case's own acceptable alternatives, or its status | every case |
| Ruled out | which indicated alternative the supplied rule rules out | any alternative labelled not feasible | assessed cases with one, and cases where no alternative is feasible: 727 training, 131 development |

The two kinds never share a right answer on one state, so an arm that ignores the vectors cannot get both.

Each question is asked with the alternatives in several orders: the listed order, the reverse, and for three alternatives one rotation.

- **Development:** 1,013 questions (633 select, 380 ruled out).
- **Training:** two orders per question drawn with a fixed seed, 3,954 examples (2,500 select, 1,454 ruled out), one pass.

## Arms

Both start from `runs/fork-v1/adapter`, on the same examples, with the settings of the Noul pilots (learning rate 5e-5, 50-step warm-up, weight decay 0.01, clip 1.0, stock numerics, deterministic kernels).

| Arm | Question | Each alternative |
| --- | --- | --- |
| Text (reference) | the fixed wording as words | name and properties as words, the adapter's own format |
| Latent | the fixed wording as 16 vectors from the fixed encoder (two distinct requests) | the name's tokens from the state, replayed; no properties |

The fixed wordings: "Which of the indicated alternatives should be selected under the supplied rule?" and "Which of the indicated alternatives does the supplied rule rule out?", each followed by "Criterion: the supplied rule applied to the supplied facts."

## Pointers

Each alternative needs the place in the state where it is named.

- All 3,750 training alternatives and 553 of the 633 development alternatives are found by rule (the name occurs in the state, allowing for case and underscores).
- The other 80, in 33 development cases, have ids that differ from the state's wording, such as `room-alder` for "Room Alder" or `nine` for "9:00". I will map these by hand to the state's words and publish the list. That mapping is model-made and not reviewed by a person.

## Checks before training

- Every pointer range decodes to the alternative's name, in both splits.
- The two requests' vectors differ.
- FP32 loss check and bit-identity of the on-device loop on both prepared files.

## Pass rule for the latent arm, on the development variants

The text arm sets the thresholds, so they are stated as formulas.

0. The pilot is conclusive only if the text arm gets at least 90% of all questions right.
1. All questions: within 5 points of the text arm.
2. The vectors steer: pairs of a select and a ruled-out question on the same state and order, both right: at least 90% of the text arm's count.
3. The pointers bind: assessed states whose select question is right in every order asked: at least 90% of the text arm's count.

One run per arm, the final step's weights are the only candidate, nothing is retried.

Reported outside the rule: each status, each kind, cases whose alternatives are named by single letters against the rest, and the listed-order select questions against the 225/232 above.

## A correction to the earlier rules

The Noul pilots each had a part "with zero question vectors, assessed twin pairs fall below 10%". That part could not fail. With the vectors zeroed, a question and its negation have identical inputs, so they get the same answer and are never both right. It measured 0/428 every time for that reason.

It is dropped here. What shows that the vectors are used is that paired questions, which differ only in their vectors, are both answered right (part 2 above, and part 1 of the Noul rules: 425/428 in pilot 4). Pilot 4's pass does not depend on the dropped part.

## Limits

- Choice only, from `fork-v1`. The Noul requests are not trained alongside; one adapter for both is a later step.
- Every training alternative is named by a single letter (A, B, C), the same letters that label the options. Development has 119 such cases and 113 with other names.
- The request always points at every alternative. Pointing at a subset is not tested.
- As for Noul, two fixed requests only require the vectors to tell two things apart.
- The pointers are given, not produced by the native side. The cases are synthetic.

## Cost

Two training runs of about 35 minutes each and two measurements of about 5 minutes each on the RTX 3090, after the variants and the hand mapping are built.
