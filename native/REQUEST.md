# Native side, step 1: forming a request from the model's own reading of the state

Written 13 November 2026; the figures come from the case files. Vince approved it the same day, with the narrower scope and the native model frozen. The labels, the head, its training loop and the scoring were then built without the GPU. Nothing that runs a model has been run: the head's code is untested until the GPU is free.

## Question

Can what a request needs be read off the native model's own states: which operation a state calls for, and where the items it can point at are?

## Where this sits

The decision side is accepted: one adapter that takes a request as a fixed word sequence plus pointers ([RESULTS-WORDS.md](../latent/RESULTS-WORDS.md)). Until now every request was handed to it. The operation came from the case's label. The pointers were found by a text rule, and for 55 development alternatives by a hand-made mapping.

The build plan has the model itself make the request, without generating text. This is the first step of that: produce the parts of a request from the states the native model already computes while reading.

## What is and is not chosen here

A request has three parts. This step produces two of them.

| Part | Example | Produced here |
| --- | --- | --- |
| The operation | this state calls for a Choice | Yes |
| The items that can be pointed at | the two prerequisites; the three alternatives | Yes |
| Which question to ask about them | select or rule out; this prerequisite or that one; negated or not | No, given |

The third part is left out because the cases hold no signal for it. One state supports eight Noul questions, or a select and a ruled-out question, and nothing in the state says which one is wanted. In use, that comes from what the model is trying to work out at that moment. That is the later stage of deciding when to call and what to ask.

So this is narrower than "the model chooses the request". It is: the model's states say what kind of decision the state is about and what can be pointed at.

## The native model is not changed

The reader is a small separate head on the frozen base model's final-layer states, taken from the pass that already reads the state with the adapter off. Nothing in the native model is trained, so its own thinking and answers stay exactly as they are.

The head, fixed in advance:

- **Operation:** one linear layer on the state's last position, three classes.
- **Items:** up to three slots, in the order the items appear in the state. For each slot, attention from the state's last position over the state's tokens gives where the item starts and where it ends, or that the slot is empty. Token states are first projected to 256 dimensions.

A pointer produced this way is a token range in the state, the same thing the adapter was trained to receive.

## What the cases supervise

Every state belongs to exactly one operation; no state is used by two.

| | Training states | Development states |
| --- | --- | --- |
| Noul | 1,282, of which 952 with two named prerequisites | 243, of which 116 with two named prerequisites |
| Choice | 1,250, three alternatives each | 230, two or three alternatives each |
| Score | 1,298, no items | 199, no items |
| Total | 3,830 | 672 |

Targets are the pointers the adapter was trained on: a prerequisite where the rule sentence names it, an alternative at its first exact mention.

Two things make development hard:

- **Names.** Every training alternative is a single letter ("Option A"). 111 of the 230 development Choice states use other names, such as "Birch" or "9:00". Training prerequisites have 192 distinct names; development's are renamed to ones training never has.
- **The operation is not written on the state.** A hand-written keyword rule on the rule text gets 604 of 674 development cases.

## Training

- All 3,830 training states, except a tenth held out by family (409 states, leaving 3,421) to tell "the head cannot learn this" from "it does not carry over to development".
- 20 passes in batches of 32 (2,140 steps), AdamW, learning rate 1e-3, weight decay 0.01, clip 1.0. The final weights are the only candidate; nothing is chosen on development.
- The native states are computed once on the GPU. The head's training runs on the GPU with the host out of the loop, as for the adapter.

## Built, and checked without the GPU

Done on 13 November 2026 (`labels.py`, `judge.py`, the tokenizer; no model is run):

| Check | Result |
| --- | --- |
| Every item's token range reads as its name and lies inside the state | all 5,654 training and 859 development items; 4 development ones carry a neighbouring mark, as before |
| The native pass is the same tokens whatever the operation | it is built from the state alone |
| Scoring with a made-up reader that produces exactly the labels | 672 of 672 operations, 346 of 346 item sets, and end to end exactly the adapter's own 1,141, 960 and 301 |
| With that reader, the rewritten panels give the adapter the same prompts as the handed panels | all 1,182 Noul questions in both pointer directions, all 1,001 Choice questions |
| Scoring with a reader spoiled in known ways | loses exactly the states and questions that were spoiled |

Still to do on the GPU: the loop check (`run_native.sh check`), then the run (`run_native.sh`). Score needs no new adapter run: a Score request has no pointers, so the adapter's answers are those it gave with handed requests, and a question counts only if the operation was produced right.

## Pass rule, on the development states

1. **Operation:** right on at least 639 of 672 states (95%).
2. **Items:** of the 346 states that have items (116 Noul, 230 Choice), every item found and none extra on at least 312 (90%). A produced range counts if it reads exactly as the item's name, at any of its mentions.
3. **End to end:** the accepted adapter `runs/words`, given the produced operation and pointers instead of the handed ones, on the same three panels. Which question to ask is still given. Within 5 points of the same adapter with handed requests:

| Panel | With handed requests | Needed with produced requests |
| --- | --- | --- |
| Noul, renamed panel (1,182) | 1,141 | at least 1,082 |
| Choice (1,001) | 960 | at least 910 |
| Score (398) | 301 | at least 282 |

In part 3 a question counts as wrong when its state's operation is produced wrong, or when it needs pointers and the state's items are not all right.

One run, nothing retried.

Reported outside the rule: the held-out tenth of training; Choice states with letter names against other names; the 55 alternatives that needed the hand mapping; items produced on states that have none.

## What each outcome would mean

| Outcome | Reading |
| --- | --- |
| All parts met | The native states carry what a request needs. The text rule and the hand mapping can be retired for these cases, and the path from state to decision runs without a handed request, apart from which question to ask. |
| Met on letter names, missed on other names | The training alternatives are too narrow, not the reading. The follow-up is to rename training alternatives by rule, as was done for development's prerequisites. |
| Part 1 missed | The final-layer state at the end of the state does not say what the state is about. The follow-up is other layers. |
| Missed on the held-out tenth as well | The head as specified cannot learn it; the design of the head is wrong. |

## Limits

- Which question to ask, and when to call at all, are not produced. Every state here gets a call.
- The states are those of the decision prompt's opening, not of a real conversation in the model's own chat format.
- A text rule already finds most of these pointers. This step shows whether the model's states can replace it, not that the task is hard.
- The final layer only. Three slots only.
- Synthetic cases, labelled by rule. One run.

## Cost

Computing the native states, training the head and three measurements: about half an hour on the RTX 3090. The labels and scripts need no GPU.

## Decisions for the owner

1. **Is the narrower scope acceptable as a first step?** The alternative is to wait for a design that also produces which question to ask, which needs cases where the model's own reasoning says what it wants to know. Those do not exist yet.
2. **Frozen native model with a separate head, or train the native model?** This design freezes it, so native behaviour cannot change. Training it could read more, at the cost of that guarantee.
