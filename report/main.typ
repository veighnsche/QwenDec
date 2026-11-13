#import "design.typ": *
#import "motif.typ": *
#import "diagrams.typ": *
#import "data.typ": d

#show: report

#let section(txt) = text(font: serif-t, style: "italic", size: 15pt, fill: ink, txt)

// A line in a list: status mark, name, a figure at the right, a hairline under it.
#let item(kind, name, value: none, sub: none) = {
  grid(
    columns: (7mm, 1fr, auto),
    align: (left + top, left + horizon, right + horizon),
    pad(top: 5.2pt, mark(kind)),
    {
      text(font: serif-t, size: 14pt, name)
      if sub != none { v(-2.6mm); note(sub) }
    },
    if value != none { value },
  )
  v(-1.5mm)
  hairline()
  v(-1.5mm)
}

#let head(num, lab, title) = {
  row(label(num, lab), headline(title))
  v(9mm)
}

#let opener(numeral, title, sub, art) = dark-page({
  kicker(fill: paper.transparentize(30%), "Chapter " + numeral)
  v(26mm)
  {
    set text(font: serif-d, size: 56pt, style: "italic", fill: paper)
    set par(leading: 0.36em)
    title
  }
  v(7mm)
  box(width: 112mm, {
    set par(leading: 0.55em)
    text(font: serif-t, size: 14pt, fill: paper.transparentize(10%), sub)
  })
  place(bottom + left, dy: -8mm, art)
})

#let verdict(passed) = if passed { status("working", txt: "Passed") } else {
  box(baseline: -0.5pt, line(length: 6pt, stroke: 1pt + soft))
  h(5pt)
  text(size: 8pt, fill: soft)[Not passed]
}

// ---------------------------------------------------------------- 01 Cover
#dark-page({
  text(font: sans, size: 14pt, weight: "medium")[QwenDec]
  v(16mm)
  {
    set text(font: serif-d, size: 56pt, style: "italic", fill: paper)
    set par(leading: 0.36em)
    [Thinking fast \ and slow. \ In one model.]
  }
  place(bottom + left, dy: -16mm, motif-fan(162mm, dark: true))
  place(bottom + left, text(font: sans, size: 10pt, fill: paper.transparentize(15%))[Progress report / November 2026])
})

// ---------------------------------------------------------------- 02 The idea
#head("01", "The idea")[One mind, \ two tempos.]
#row([], lede[QwenDec is building a language model that thinks at two speeds with one set of weights.])
#v(5mm)
#row(
  {
    v(1pt)
    note(fill: petroleum)[*Slow.* Reasoning, one token after another.]
    v(3mm)
    note(fill: mineral.darken(20%))[*Fast.* Noul, Choice and Score: a decision in one step.]
  },
  [
    A language model reasons slowly. It writes one token, then the next. Some questions inside that reasoning do not need a paragraph: is this true, which of these, how much.

    QwenDec gives the model a fast way to settle them. These are JEV-style decisions, answered in one step as a value and made inside the model while it thinks.

    It stays one model. The prompt is read once. When a decision is needed, the request and the answer pass inside the model as vectors, never as written text.
  ],
)
#v(7mm)
#row(label(none, "One model"), {
  one-model()
  v(3mm)
  status-key()
  v(2mm)
  note[The goal, marked by its state today. Noul and Choice decide from requests that arrive as vectors. Score is open. The model does not yet ask for a decision or take the result back by itself.]
})

#pagebreak()

// ---------------------------------------------------------------- 03 Summary
#head("02", "Summary")[What works. \ What comes next.]
#row(section[Working], {
  item("working", [Noul], value: frac(d.noul.right, d.noul.of, size: 21pt), sub: [Is it true? Asked as vectors, not text.])
  item("working", [Choice], value: frac(d.choice.right, d.choice.of, size: 21pt), sub: [Which one? Asked as vectors, not text.])
  item("working", [Deciding from what the model already read], value: frac("69", "81", size: 21pt), sub: [Recognises an unresolved question. Before: 2 of 81.])
  item("working", [Training without the CPU], value: fig(size: 21pt, d.loop.cpu), sub: [Share of one CPU core during a full training run.])
})
#v(7mm)
#row(section[Pending], {
  item("pending", [One adapter for both], value: fig(size: 21pt, fill: amber.darken(15%))[#d.joint.noul #h(3pt)·#h(3pt) #d.joint.choice], sub: [Passed its rule narrowly. Acceptance pending.])
})
#v(7mm)
#row(section[Still to build], {
  item("tobuild", [Reliable Score], sub: [In-between values are poor.])
  item("tobuild", [Model asks + uses answers], sub: [Today the request is handed to the model.])
  item("tobuild", [Decision budget prediction])
  item("tobuild", [Runtime + accounting])
})
#v(1fr)
#row([], note[Fractions are questions answered correctly on development tests. All cases are synthetic. Pages 10 to 12 and 19 say what the figures do and do not show.])

// ---------------------------------------------------------------- 04 Opener I
#opener("I", [The fast \ tempo.], [Three decisions, and how they came to read what the model has already read.], motif-three(162mm, dark: true))

// ---------------------------------------------------------------- 05 Three decisions
#head("03", "The decisions")[Three decisions.]
#row([], lede[Each answers one bounded question with a value, in one step.])
#v(8mm)
#row([], grid(
  columns: (36mm, 36mm, 36mm), column-gutter: 6mm,
  ..(
    (noul-glyph(u: 4.4mm), "Noul", "Is it true?", [The probability that a statement is true, given the evidence.]),
    (choice-glyph(u: 4.4mm), "Choice", "Which one?", [One of several alternatives. The same one, whatever order they are listed in.]),
    (score-glyph(u: 4.4mm), "Score", "How much?", [A value between 0 and 1, measured against a rubric.]),
  ).map(((g, name, q, body)) => {
    box(height: 9mm, align(left + horizon, g))
    v(1mm)
    text(font: serif-d, size: 24pt, name)
    v(-2mm)
    text(font: serif-t, style: "italic", size: 12pt, fill: soft, q)
    v(0mm)
    hairline()
    v(0mm)
    body
  }),
))
#v(12mm)
#row(section[A decision \ can decline], {
  [Only a decision that can be assessed carries a value. Otherwise it says why not.]
  v(3mm)
  grid(
    columns: (1fr, 1fr), column-gutter: 8mm, row-gutter: 6mm,
    [*Insufficient* \ #note[The evidence does not reach the question.]],
    [*Conflicting* \ #note[The evidence points both ways.]],
    [*Unresolved* \ #note[The question or rule can be read in more than one way.]],
    [*No feasible choice* \ #note[Choice only: none of the alternatives qualifies.]],
  )
  v(6mm)
  hairline()
  v(1mm)
  note[Knowing when not to answer was the largest gap in the adapter QwenDec started from, and the first thing it repaired. The next page shows by how much.]
})

#pagebreak()

// ---------------------------------------------------------------- 06 Fork
#head("04", "The model's own reading")[Deciding from what \ was already read.]
#row(
  {
    kicker("Unresolved questions recognised")
    v(2mm)
    evidence()
  },
  {
    frac("69", "81", size: 54pt, fill: mineral.darken(12%))
    v(-3mm)
    lede[Before: 2 of 81. The decision now works from the model's own reading of the prompt.]
  },
)
#v(6mm)
#row([], [
  The published decision adapter expects to read the whole prompt itself. In QwenDec the model reads the prompt natively, as it always does, and nothing is read twice. The decision adapter switches on only for the question and its options.

  One training run on 10 November taught the adapter to decide from that state.
])
#v(6mm)
#row(label(none, "Before and after"), {
  tbl(
    (1fr, 16mm, 16mm, 14mm),
    ([Development cases], [Before], [After], [Of]),
    d.fork.rows.map(r => (r.at(0), num(text(fill: soft, r.at(1))), num(r.at(2)), num(text(fill: soft, r.at(3))))),
    align: (left + horizon, right + horizon, right + horizon, right + horizon),
  )
  v(4mm)
  status("working", txt: [Working. Accepted on 11 November for Noul, Choice and recognising when not to answer.])
  v(2mm)
  status("pending", txt: [Score did not follow. In-between values: #d.fork.score-between-after of #d.fork.score-between-of right, against #d.fork.score-between-before before. See page 17.])
})

// ---------------------------------------------------------------- 07 Opener II
#opener("II", [Asking \ without words.], [A request that crosses as vectors, not text. It took four attempts.], motif-select(162mm, dark: true))

// ---------------------------------------------------------------- 08 Request
#head("05", "The request")[How a request \ crosses.]
#row([], lede[A request is 16 vectors and a pointer. No sentence is written.])
#v(5mm)
#row(
  {
    v(1pt)
    note(fill: mineral.darken(20%))[*The kind of request.* One of a few fixed kinds: six for Noul, two for Choice. Each is 16 vectors.]
    v(3mm)
    note(fill: mineral.darken(20%))[*The item.* A pointer to the item's own name in what the model has read.]
  },
  [
    The request says two things. What kind of question is being asked, and which item it is about.

    The kind crosses as vectors. The item is not described again: the request points at the place where the model already read its name.
  ],
)
#v(8mm)
#row(label(none, "One request"), request-crossing())
#v(6mm)
#row([], {
  hairline()
  v(1mm)
  note[Why a pointer. Vectors alone carried the kind of question but not the item: with two items in play, the answer was right no more often than chance. Pointing at the item's own tokens fixed that.]
})

#pagebreak()

// ---------------------------------------------------------------- 09 Four attempts
#head("06", "The road")[Four attempts.]
#row([], lede[The format passed on the fourth attempt. The first three did not, and each failure said what to change.])
#v(7mm)
#{
  let what = (
    [*Vectors alone.* Right kind of question, wrong item: #d.pilot1-chance of #d.pilot1-chance-of where two items differ, which is chance.],
    [*A pointer added.* The item is now found: #d.pilot2-binding of 72. It fell short of the #d.pilot2-needed needed, on questions worded freely.],
    [*Six fixed requests.* Free wording repaired: #d.pilot3-free of #d.pilot3-free-of. A negation was lost on two names that differ in one word.],
    [*The same design, run again,* judged on distinct names. All six parts of its rule met.],
  )
  for (i, p) in d.pilots.enumerate() {
    row(
      {
        text(font: serif-d, size: 30pt, fill: if p.passed { mineral.darken(12%) } else { soft }, p.n)
      },
      {
        grid(
          columns: (1fr, auto), align: (left + bottom, right + bottom),
          frac(p.right, "1,182", size: 19pt), verdict(p.passed),
        )
        v(-1.5mm)
        track(p.v, c: if p.passed { mineral } else { petroleum })
        v(-1.5mm)
        note(fill: ink, what.at(i))
      },
    )
    v(5.5mm)
  }
}
#v(1fr)
#row(evidence(), {
  hairline()
  v(1mm)
  note[Bars run from 0 to 1,182 questions. The tick marks 1,174, the score when the same questions are read as text. Attempts 1 to 3 were judged on the original item names, attempt 4 on distinct names; the text score is 1,174 on both.]
  v(2mm)
  note[Attempts 3 and 4 are one design. Only the order of the training steps differs. On the same questions they score #d.pilot3-same-panel and 1,162: how far two runs can differ is still an open question.]
})

#pagebreak()

// ---------------------------------------------------------------- 10 Noul
#head("07", "Result")[Noul.]
#row(
  {
    kicker("Questions right")
    v(2mm)
    evidence()
  },
  {
    frac(d.noul.right, d.noul.of, size: 54pt, fill: mineral.darken(12%))
    v(-3mm)
    lede[Against #d.noul.text when the same questions are read as text.]
    v(4mm)
    field(1182, 20, cols: 83)
    v(-1mm)
    note[One square for each question. The 20 hollow ones, in the last row, were answered wrong.]
  },
)
#v(7mm)
#row(label(none, "What holds"), {
  metric([A question and its negation, both right], d.noul.pairs, of: d.noul.pairs-of, text-ref: d.noul.pairs-text)
  hairline()
  metric([The answer follows the pointer when it moves], d.noul.pointer, of: d.noul.pointer-of)
  hairline()
  metric([Questions held back from training], d.noul.held-back, of: d.noul.held-back-of)
  hairline()
  metric([Two item names that differ in one word], d.noul.own-names, of: d.noul.of)
  hairline()
})
#v(6mm)
#row(label(none, "Where it is weaker"), {
  metric([Questions worded freely], d.noul.free, of: d.noul.free-of, text-ref: d.noul.free-text, kind: "pending")
  hairline()
  v(2mm)
  note[#d.noul.errors-free of its #d.noul.errors errors are on freely worded questions, a wording the training cases do not contain.]
})
#v(8mm)
#row([], status("working", txt: [Working. The request format was accepted for Noul on 11 November, as the basis for the next stage. Reliability across runs was left open.]))

#pagebreak()

// ---------------------------------------------------------------- 11 Choice
#head("08", "Result")[Choice.]
#row(
  {
    kicker("Questions right")
    v(2mm)
    evidence()
  },
  {
    frac(d.choice.right, d.choice.of, size: 54pt, fill: mineral.darken(12%))
    v(-3mm)
    lede[Against #d.choice.text when the question and the alternatives are read as text.]
    v(4mm)
    field(1001, 46, cols: 85)
    v(-1mm)
    note[One square for each question. The 46 hollow ones, in the last row, were answered wrong.]
  },
)
#v(7mm)
#row(label(none, "What holds"), {
  metric([The same answer in every order of the alternatives], d.choice.order, of: d.choice.order-of, text-ref: d.choice.order-text)
  hairline()
  metric([Picks the right alternative when there is one], d.choice.pick, of: d.choice.pick-of, text-ref: d.choice.pick-text)
  hairline()
  metric(["Which to pick" and "which is ruled out", both right], d.choice.pairs, of: d.choice.pairs-of, text-ref: d.choice.pairs-text)
  hairline()
})
#v(6mm)
#row(label(none, "Where it is weaker"), {
  metric([Recognises that no alternative qualifies], d.choice.nopick, of: d.choice.nopick-of, text-ref: d.choice.nopick-text, kind: "pending")
  hairline()
  v(2mm)
  note[Asked which to pick when none qualifies, it sometimes names one anyway. The text version does not make this mistake.]
})
#v(6mm)
#row([], {
  [The request never restates an alternative's properties. It only points at each alternative's name; the properties stay where the model read them. On two measures this does better than text.]
  v(5mm)
  status("working", txt: [Working. The request format was accepted for Choice on 11 November, on the same terms as Noul.])
})

#pagebreak()

// ---------------------------------------------------------------- 12 Joint
#head("09", "Result")[One adapter \ for both.]
#row([], lede[One adapter now serves both kinds of request. It passed its rule, narrowly.])
#v(6mm)
#row(
  {
    kicker("Questions right", fill: amber.darken(18%))
    v(2mm)
    evidence()
  },
  grid(
    columns: (1fr, 1fr), column-gutter: 8mm,
    [#note[Noul] \ #frac(d.joint.noul, d.noul.of, size: 24pt, fill: amber.darken(15%)) \ #note[Needed #d.joint.noul-needed. Separate adapter: #d.noul.right.]],
    [#note[Choice] \ #frac(d.joint.choice, d.choice.of, size: 24pt, fill: amber.darken(15%)) \ #note[Needed #d.joint.choice-needed. Separate adapter: #d.choice.right.]],
  ),
)
#v(7mm)
#row(label(none, "The rule, part by part"), {
  tbl(
    (1fr, 15mm, 18mm, 14mm),
    ([Part], [Needed], [Measured], [Margin]),
    d.joint.parts.map(p => {
      let narrow = p.at(3) == 2 or p.at(3) == 8 or p.at(3) == 13
      (
        [#mark(if narrow { "pending" } else { "working" }, r: 2.4pt) #h(4pt) #p.at(0)],
        num(text(fill: soft, p.at(1))), num(p.at(2)),
        num(text(fill: if narrow { amber.darken(22%) } else { ink })[+#p.at(3)]),
      )
    }),
    align: (left + horizon, right + horizon, right + horizon, right + horizon),
  )
  v(2mm)
  note[All eight parts are met. Three by small margins: 2, 8 and 13 questions, marked with a ring.]
})
#v(6mm)
#row([], {
  [It scores below both separate adapters. Whether that is a cost of training the two together is not shown: two Noul-only runs of one design differ by about as much.]
  v(5mm)
  status("pending", txt: [Acceptance pending. The decision is on page 18.])
})

// ---------------------------------------------------------------- 13 Opener III
#opener("III", [Training \ without the CPU.], [The training loop runs on the graphics card from the first step to the last.], motif-loop(162mm, dark: true))

// ---------------------------------------------------------------- 14 Loop
#head("10", "The training loop")[The loop lives \ on the GPU.]
#row([], lede[Training no longer waits for the CPU.])
#v(5mm)
#row(
  {
    v(1pt)
    note[In the first attempt at this project, one CPU core ran at full load throughout training and held the graphics card back. It could not be removed afterwards.]
  },
  [
    QwenDec therefore started from the other end. First a training loop that does not involve the CPU, then everything else.

    In ordinary training the CPU sends the GPU every small operation, one at a time. Here one whole training step is recorded once: pick the batch, compute, correct the weights, write the log. The GPU then repeats that step by itself.
  ],
)
#v(7mm)
#row(label(none, "Who drives the loop"), loop-diagram())
#v(4mm)
#row([], {
  hairline()
  v(1mm)
  note[Everything a step needs lives on the GPU: the data, their order, the learning-rate schedule, the optimiser and the logs. The CPU wakes once every 50 steps to save a checkpoint.]
})

#pagebreak()

// ---------------------------------------------------------------- 15 Measured
#head("11", "The training loop")[What was \ measured.]
#row(
  {
    kicker("Share of one CPU core")
    v(2mm)
    note[During a full training run of #d.loop.steps-first steps.]
  },
  {
    fig(size: 54pt, fill: mineral.darken(12%), d.loop.cpu)
    v(-3mm)
    lede[With the graphics card busy #d.loop.gpu of the time.]
  },
)
#v(8mm)
#row(label(none, "Side by side"), {
  tbl(
    (1fr, 20mm, 20mm, 20mm),
    ([A test of 16 steps], [CPU core], [GPU busy], [Per step]),
    (
      ([Ordinary training], num(d.loop.eager-cpu), num(d.loop.eager-gpu), num(d.loop.eager-ms)),
      ([The loop on the GPU], num(d.loop.loop-cpu), num(d.loop.loop-gpu), num(d.loop.loop-ms)),
    ),
    align: (left + horizon, right + horizon, right + horizon, right + horizon),
  )
})
#v(6mm)
#row(label(none, "What holds"), {
  item("working", [The same result, bit for bit], sub: [The loop ends with exactly the weights ordinary training gives, step for step.])
  item("working", [Stops and resumes exactly], sub: [A run continued in a fresh process ends in the identical state.])
  item("working", [Used for every run in this report], sub: [#d.loop.steps-first steps for the first adapter, #d.loop.steps-joint for the single adapter, and each attempt in between.])
})
#v(5mm)
#row(label(none, "To know"), {
  note[On the real training data the loop is not faster than ordinary training; the step is bound by the graphics card either way. What it removes is the CPU core held at 100%.]
  v(2mm)
  note[It relies on a feature of PyTorch that is still in beta and undocumented.]
})

// ---------------------------------------------------------------- 16 Opener IV
#opener("IV", [What \ comes next.], [Four pieces still to build, and two decisions waiting.], motif-select(162mm, dark: true, unfinished: true))

// ---------------------------------------------------------------- 17 Still to build
#head("12", "Ahead")[Still to build.]
#row([], lede[The decisions work when they are asked. The model does not ask yet.])
#v(7mm)
#row(section[Pending], {
  item("pending", [One adapter for both Noul and Choice], sub: [Passed its rule on one run. Acceptance pending, page 18.])
})
#v(6mm)
#row(section[To build], {
  item("tobuild", [Reliable Score], sub: [
    Score recognises when not to answer (#d.score.status of #d.score.status-of) and the ends of the scale (#d.score.ends of #d.score.ends-of). In-between values are poor: #d.score.between of #d.score.between-of. The training cases hold only 19 distinct values. A test of a Score request as vectors is designed, not run.
  ])
  item("tobuild", [Model asks + uses answers], sub: [
    Today each request is handed to the model. It has to learn to decide when to ask, to form the request and its pointers, and to use the result as it continues. This is the largest piece.
  ])
  item("tobuild", [Decision budget prediction], sub: [
    A prediction, made at the start, of how much thinking a prompt deserves.
  ])
  item("tobuild", [Runtime + accounting], sub: [
    One inference job that runs all of it and counts every position it computes.
  ])
})
#v(1fr)
#row([], note[In the repository these are called the joint adapter, Score, the native side, the allowance predictor and the runtime.])

#pagebreak()

// ---------------------------------------------------------------- 18 Decisions
#head("13", "Ahead")[Two open \ decisions.]
#row([], lede[Both are the owner's to take. Nothing is running until they are.])
#v(8mm)
#row(text(font: serif-d, size: 30pt, fill: amber.darken(15%))[1], {
  section[Accept the single adapter, or run it once more.]
  v(3mm)
  [It met both rules on one run, three parts by small margins. A second run with a different order of training steps would say whether that pass is stable.]
  v(3mm)
  grid(
    columns: (1fr, 1fr), column-gutter: 8mm,
    [*Accept* \ #note[The next stage builds on one adapter now. Its stability stays unknown.]],
    [*Run once more* \ #note[About 95 minutes of GPU time. The rule and its numbers stay as they are.]],
  )
})
#v(4mm)
#row([], hairline())
#v(4mm)
#row(text(font: serif-d, size: 30pt, fill: amber.darken(15%))[2], {
  section[Approve the Score request test.]
  v(3mm)
  [It asks whether a Score request can cross as vectors, with the rubric left where the model read it. The design and its pass rule are written. It needs about one hour of GPU time.]
  v(3mm)
  note[It is not expected to repair Score values. It would show whether the request is the problem. Three points in the design are also open: whether to run it before values are repaired, whether to test a pointer, and which adapter to start from.]
})
#v(1fr)
#row([], {
  hairline()
  v(1mm)
  note[No GPU time has been assigned to QwenDec since 12 November.]
})

#pagebreak()

// ---------------------------------------------------------------- 19 Evidence
#head("14", "Reading the evidence")[How to read \ the evidence.]
#row(section[How the work \ was run], {
  item("working", [Rule first], sub: [Each test's design and pass rule were published before its run.])
  item("working", [One run each], sub: [The last step's weights are the only candidate. Nothing is retried after a failure. Three attempts are recorded as not passed.])
  item("working", [Passing is not accepting], sub: [Applying the rule is mechanical. Accepting a result as the basis for the next stage is the owner's decision.])
  item("working", [Open records], sub: [The code, the cases, the logs and the answer to every single question are public. No weights are.])
})
#v(7mm)
#row(section[What it does \ not show], {
  set text(size: 9.5pt)
  [*A few fixed kinds of request.* Six for Noul, two for Choice. The vectors only have to tell these apart. They do not yet carry a question's content.]
  v(1mm)
  [*Pointers are given.* The model does not yet choose what to ask about.]
  v(1mm)
  [*Reliability across runs is open.* Two runs of one design differed by 30 questions out of 1,182.]
  v(1mm)
  [*The cases are synthetic,* built from templates and labelled by rule. A model reviewed the labels, not people.]
  v(1mm)
  [*General skill is unmeasured.* Nothing here tests whether the adapter lost ability elsewhere.]
  v(1mm)
  [*Development figures only.* A locked evaluation set exists and has never been read.]
})

#pagebreak()

// ---------------------------------------------------------------- 20 Sources
#head("15", "Sources")[Sources.]
#row([], lede[Every figure in this report comes from the public repository.])
#v(3mm)
#row([], link("https://github.com/veighnsche/QwenDec")[#text(fill: petroleum)[github.com/veighnsche/QwenDec]])
#v(7mm)
#row(label(none, "By chapter"), {
  let src(a, b) = (a, text(font: "Red Hat Mono", size: 8pt, fill: petroleum, b))
  grid(
    columns: (54mm, 1fr), row-gutter: 3mm,
    ..src([Overview and limits], [README.md]),
    ..src([What the system must do], [docs/requirements.md]),
    ..src([I. The model's own reading], [train/RESULTS.md, baseline/]),
    ..src([II. Noul, four attempts], [latent/RESULTS.md, RESULTS2.md, RESULTS3.md, RESULTS4.md]),
    ..src([II. Choice], [latent/CHOICE.md, RESULTS-CHOICE.md]),
    ..src([II. One adapter for both], [latent/JOINT.md, RESULTS-JOINT.md]),
    ..src([III. The training loop], [gpu_loop/RESULTS.md, train/RESULTS.md]),
    ..src([IV. Score request test], [latent/SCORE.md]),
  )
  v(1mm)
  hairline()
})
#v(6mm)
#row(label(none, "Models"), {
  note(fill: ink)[The language model is Qwen3.5-4B. The decision adapter QwenDec started from is the published decision-4b. Neither is redistributed.]
})
#v(4mm)
#row(label(none, "Measures"), {
  note(fill: ink)[A question counts as right when its status is right and, where it has a value, the value is right. Score figures use the picked level, within 0.05 of the target.]
})
#v(1fr)
#row([], motif-select(120mm, dark: false))
#v(1fr)
#row(label(none, "Colophon"), {
  hairline()
  v(1mm)
  note[State of 12 November 2026. Development tests on synthetic cases. \ Set in Newsreader, Red Hat Text and Red Hat Mono. Typeset with Typst.]
})
