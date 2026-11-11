# Product requirements

The 20 GIVEN / WHEN / THEN clauses that define what the system must do. Taken from QwenOneTwo (`docs/blind-architecture-requirements-2026-11-09.md`), without its two assignment-framing clauses. They were written for Qwen3.5-0.8B with Qev; QwenDec uses Qwen3.5-4B with `decision-4b`, so read the model named in the mode clause accordingly.

GIVEN a user submits one text prompt,
WHEN the system handles it,
THEN it selects an initial thinking allowance, may perform internally chosen decisions, and produces a normal text answer within one inference job. There is one prompt prefill and one causal history; no second user action, separate decision request, prompt replay or second prefill is allowed.

GIVEN the chosen foundation is Qwen3.5-0.8B with a Qev decision LoRA,
WHEN selecting the initial allowance or evaluating an internal decision,
THEN Qev is active; WHEN producing native reasoning or answer text, THEN Qev is inactive. Preserve the pretrained base and original Qev release; trainable extensions and any aligned Qev derivative must be explicit and independently versioned. Do not silently change this mode requirement. If mode switching and a single correct history cannot coexist, identify the feasibility blocker and the experiment needed to resolve it before proposing costly fitting.

GIVEN the model can choose to continue, invoke Noul, invoke Choice, or invoke Score,
WHEN it reasons or answers,
THEN invocation timing and the proposition/question, evidence, target, candidate alternatives or rubric are learned model outputs. No invocation can be legitimate, and multiple serial invocations can be legitimate. Do not route using keywords, domains, hand-written semantic rules or a second translator/judge model in deployed inference. Initially support one pending request and no nested or asynchronous decision calls.

GIVEN an internal request or result crosses the native/decision boundary,
WHEN it is transported,
THEN its semantic interface consists of hidden vectors with typed indices, scalars and metadata, not generated question strings, JSON or textual tool calls. Offline text annotations and training-only semantic supervision are allowed. Exact identities, order, names, dates and quantities needed by a task must remain recoverable. Do not assume that all request-specific information can reside in weights.

GIVEN the user expects autonomous task completion,
WHEN the native model forms a request, receives a bound decision, and continues,
THEN the complete formation/invocation/result-consumption behavior must be trained and demonstrated. Training only a decision receiver or a bridge is insufficient. Do not claim general capability from forced calls or diagnostic fixtures.

GIVEN Noul receives a bound proposition and evidence,
WHEN it can assess that proposition,
THEN return a probability of that proposition being true. Preserve negation and criterion meaning; true must not universally mean approve or act. A probability near 0.5 is not automatically insufficient evidence.

GIVEN Choice receives a criterion and an ordered bank of alternatives,
WHEN a supported choice is possible,
THEN return a stable selected identity and its usable contents, preserving identity under reordered alternatives. Ranking does not prove feasibility. Unknown availability is not known unavailability. Equally valid alternatives may be accepted without invented preferences. Whether a known workable option suffices depends on whether the question asks for any workable option or a certified best option.

GIVEN Score receives a target and a supplied rubric,
WHEN an assessment is supported,
THEN return a continuous value in [0,1] attached to that exact target and rubric meaning, including conditions, anchors and direction. Interior values must have demonstrated meaning. Zero and 0.5 are ordinary valid assessed values, not absence or generic uncertainty markers.

GIVEN the evidence cannot support an assessment,
WHEN evaluating any operation,
THEN distinguish Insufficient evidence, materially Conflicting evidence, and Unresolved interpretation/assessment; Choice also supports No feasible choice for its supplied bank. Only Assessed has a decision value. Non-assessed returns identify relevant missing/conflicting facts or constraints through bound latent reasons. Irrelevant missing facts need not prevent a supported assessment. Never fabricate numeric/selection supervision for absent values.

GIVEN the native model receives a result,
WHEN it resumes the user's task,
THEN it uses the result's meaning, selected contents and reasons to answer, clarify, qualify or request alternatives appropriately. It must not blindly obey every result, ignore every result, invent withheld facts, or repeat identical calls as though they create evidence. Demonstrate content-sensitive responses under controlled result interventions without giving the native model gold answers or hidden facts.

GIVEN requests and results refer to user facts and alternatives,
WHEN a request is dispatched and a result is consumed,
THEN all relevant bindings, evidence, bank order and rubric versions are immutable and unambiguous. Duplicate, stale, foreign or late returns cannot be consumed twice or revive a cancelled job. Runtime performs structural/resource validation, not hidden semantic correction or gold-answer substitution. Structural errors, cancellation and execution/resource failures remain distinct from semantic statuses.

GIVEN the initial thinking allowance is predicted,
WHEN computing it,
THEN use one continuous scalar regression readout in an actual LM-head row, separate from the task Score readout, with s=clamp(raw,0,1) and B_pred=floor((3*s*s-2*s*s*s)*131072+0.5), using the specified ordered FP64 arithmetic. Do not substitute budget bins or a categorical average. Total context is 262144; grant at most the remaining capacity after prompt, answer and necessary control reservations. Show predicted, explicit override if present, and granted allowances separately.

GIVEN an inference job has limited context and thinking allowance,
WHEN it performs reasoning, internal request construction, decision evaluation or feedback delivery,
THEN charge all positions actually executed, including rejected/aborted attempts. Keep visible-answer and thinking allowances separate, preserve a normal answer at zero extra thinking allowance, do not reset allowance at answer start, do not let general Score implicitly change the allowance, and do not start a call that cannot finish within resource bounds. A vector input consumes a position; storage by itself does not. Account for any recomputation or additional processing honestly.

GIVEN a user cancels or provides new input,
WHEN a request is pending or computation is active,
THEN cancellation prevents later delivery/resumption and preserves correct accounting. New input queues a new job or cancels the old one; it does not mutate a sealed pending request. Preserve normal answer closure and termination behavior.

GIVEN model workloads run on one explicitly selected RTX 3090,
WHEN performing inference, training, extraction, fitting or real-model tests,
THEN execute model computation on that GPU with no CPU model fallback. Ordinary host orchestration, file I/O and static checks may use CPU. Keep external services and model source artifacts unchanged. The training design must fit the actual memory envelope, including optimizer state, gradients, activation graphs, temporary workspaces and checkpointing.

GIVEN training throughput is part of acceptance,
WHEN measuring representative steady-state steps,
THEN the CPU must keep up with the GPU. A saturated CPU thread causing GPU idle gaps fails the performance objective. Do not substitute a CPU cap, cosmetic utilization target or reduced work for efficient execution. Specify profiling to establish where time goes, numerical/gradient checks for optimization changes, meaningful workload throughput, memory peaks and end-to-end accounting. Do not assume a new programming language alone removes host overhead.

GIVEN training must establish actual semantic learning and native use,
WHEN specifying data and objectives,
THEN name the labels and matched/counterfactual supervision each stage requires, identify missing supervision as a blocker, preserve legitimate absence masks, and distinguish engineering fixtures from reviewed semantic evidence. Do not invent label coverage, train on evaluation cases, derive labels from the answer under test, or assume data exists because a schema exists. Propose data requirements without creating data.

GIVEN a proposed finite training stage,
WHEN it is authorized and run in a future implementation,
THEN fix its owners, objectives, exposure/order, checkpoint/recovery protocol and acceptance criteria before fitting; preserve complete-window model/optimizer/RNG/sampler state and frozen-owner invariants. Separate execution completion from useful-learning acceptance. No automatic extra passes, architecture searches or deployment follow a failure.

GIVEN a trained candidate is evaluated,
WHEN deciding whether it is useful,
THEN test joint status and delivered value/content, Noul against training-defined constant baselines, continuous interior Score and rubric-direction changes, Choice identity/order/feasibility retention, native no-call and multi-call behavior, reasons and bindings, numeric/accounting/cancellation correctness, and ordinary answering retention. Use genuinely held-out families and wording/operator/range interventions, keep all cases/failures in denominators, report both-correct pairs, and forbid development fitting or best-checkpoint selection. Require evidence per stage before dependent native training; explicitly report failed or untestable acceptance.

GIVEN some requirements may be mutually difficult or infeasible on this model/hardware,
WHEN you identify a conflict,
THEN explain it plainly, offer the minimum decisive proof or falsification test, and identify any product requirement change that would need user approval. Do not silently relax one requirement to make an elegant architecture look feasible.
