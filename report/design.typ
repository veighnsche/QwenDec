// QwenDec progress report: the design system.
// "One mind, two tempos": petroleum for slow reasoning, mineral for decisions
// and working pieces, amber for pending work, grey dashes for what is not built.

#let paper = rgb("#F4F0E7")
#let ink = rgb("#172A2E")
#let petroleum = rgb("#234D58")
#let mineral = rgb("#2C9B8F")
#let amber = rgb("#C28C35")
#let grey = rgb("#646C68")
#let hair = color.mix((ink, 26%), (paper, 74%))
#let soft = color.mix((ink, 62%), (paper, 38%))
#let mineral-light = rgb("#5CC4B7")
#let ivory-blue = rgb("#C9DBDC")

#let serif-d = "Newsreader 72pt"
#let serif-t = "Newsreader 16pt"
#let sans = "Red Hat Text"

#let side = 24mm
#let label-w = 36mm
#let gutter = 6mm
#let body-w = 120mm

#let two(n) = if n < 10 { "0" + str(n) } else { str(n) }

#let footer = context {
  let n = counter(page).get().first()
  set text(font: serif-t, size: 8.5pt, fill: soft)
  line(length: 100%, stroke: 0.4pt + hair)
  v(-4pt)
  grid(
    columns: (1fr, auto),
    text(style: "italic")[Development tests / 12 Nov 2026],
    text(number-type: "lining", number-width: "tabular")[#two(n)],
  )
}

#let report(body) = {
  set document(title: "QwenDec: progress report, November 2026", author: "QwenDec")
  set page(
    paper: "a4",
    fill: paper,
    margin: (left: side, right: side, top: 26mm, bottom: 26mm),
    footer: footer,
    footer-descent: 9mm,
  )
  set text(font: sans, size: 10.5pt, fill: ink, lang: "en", number-type: "lining")
  set par(leading: 0.7em, spacing: 1.35em, justify: false)
  body
}

// Small tracked capitals with a number, as on the board ("01  COVER").
#let label(num, txt) = {
  set text(font: serif-t, size: 8.2pt, fill: soft, tracking: 0.16em)
  if num != none {
    text(tracking: 0.06em, size: 9.6pt, number-type: "lining")[#num]
    h(7pt)
  }
  upper(txt)
}

#let kicker(txt, fill: soft) = text(font: serif-t, size: 8.2pt, fill: fill, tracking: 0.16em, upper(txt))

#let headline(body, size: 38pt) = {
  set text(font: serif-d, size: size, weight: "regular", fill: ink)
  set par(leading: 0.42em, spacing: 0pt)
  body
}

#let lede(body) = {
  set text(font: serif-t, size: 15.5pt, fill: ink)
  set par(leading: 0.52em)
  body
}

#let note(body, fill: soft) = {
  set text(font: sans, size: 9pt, fill: fill)
  set par(leading: 0.62em)
  body
}

#let hairline(c: hair, w: 100%) = line(length: w, stroke: 0.4pt + c)

// A label column and a reading column.
#let row(side-content, body, gap: 0pt) = {
  grid(
    columns: (label-w, body-w),
    column-gutter: gutter,
    side-content, body,
  )
  v(gap)
}

// Evidence figures: serif, lining, tabular.
#let fig(body, size: 30pt, fill: ink) = text(
  font: serif-d, size: size, fill: fill, number-type: "lining", number-width: "tabular", body,
)
#let frac(a, b, size: 30pt, fill: ink) = fig(size: size, fill: fill)[#a#h(0.22em)/#h(0.22em)#b]
#let num(body) = text(font: serif-t, size: 12.5pt, number-type: "lining", number-width: "tabular", body)

#let evidence(txt: "Evidence / synthetic cases") = text(font: serif-t, style: "italic", size: 9.5pt, fill: soft, txt)

// Status marks: shape and word, never colour alone.
#let mark(kind, r: 3pt) = box(width: 2 * r, height: 2 * r, {
  if kind == "working" {
    place(center + horizon, circle(radius: r, fill: mineral, stroke: none))
  } else if kind == "pending" {
    place(center + horizon, circle(radius: r - 0.55pt, fill: none, stroke: 1.1pt + amber))
  } else {
    place(center + horizon, circle(radius: r - 0.3pt, fill: none, stroke: (paint: grey, thickness: 1.05pt, dash: (2.3pt, 1.9pt))))
  }
})

#let status-word(kind) = if kind == "working" { "Working" } else if kind == "pending" { "Pending" } else { "To build" }

#let status(kind, txt: none) = {
  mark(kind)
  h(5pt)
  text(font: sans, size: 9pt, fill: if kind == "pending" { amber.darken(22%) } else { soft })[
    #if txt == none { status-word(kind) } else { txt }
  ]
}

#let status-key() = {
  grid(
    columns: 3, column-gutter: 16pt,
    status("working"), status("pending"), status("tobuild"),
  )
}

// A page on ink, without footer.
#let dark-page(body) = page(
  fill: ink, footer: none,
  margin: (left: side, right: side, top: 26mm, bottom: 22mm),
  {
    set text(fill: paper)
    body
  },
)

// A measure with its figure and what the text reference scored.
#let metric(name, value, of: none, text-ref: none, kind: "working") = {
  grid(
    columns: (6mm, 1fr, 31mm, 14mm),
    align: (left + horizon, left + horizon, right + horizon, right + horizon),
    mark(kind),
    name,
    num[#value#if of != none [ #text(fill: soft)[of] #of]],
    if text-ref != none { note[text #text-ref] } else { [] },
  )
}

#let hgrid(..args) = grid(stroke: none, ..args)

// A small table: header, hairline, rows, hairline.
#let tbl(columns, header, rows, align: left) = {
  grid(columns: columns, align: align, column-gutter: 3mm, ..header.map(h => note(h)))
  v(-2.2mm)
  hairline()
  v(-1.4mm)
  grid(columns: columns, align: align, column-gutter: 3mm, row-gutter: 3mm, ..rows.flatten())
  v(-1.4mm)
  hairline()
}
