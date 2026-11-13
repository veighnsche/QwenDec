// The motif: one fine boundary encloses a slow path of stepped points and a
// decision path that branches and resolves. Drawn in units of 1/100 of the width.
#import "design.typ": *

#let pt(s, p) = (p.at(0) * s, p.at(1) * s)

#let bez(p0, p1, p2, p3, t) = {
  let u = 1 - t
  (
    u * u * u * p0.at(0) + 3 * u * u * t * p1.at(0) + 3 * u * t * t * p2.at(0) + t * t * t * p3.at(0),
    u * u * u * p0.at(1) + 3 * u * u * t * p1.at(1) + 3 * u * t * t * p2.at(1) + t * t * t * p3.at(1),
  )
}

#let seg(s, p0, p1, p2, p3, stroke) = place(top + left, curve(
  stroke: stroke, fill: none,
  curve.move(pt(s, p0)), curve.cubic(pt(s, p1), pt(s, p2), pt(s, p3)),
))

#let dot(s, p, r, fill: none, stroke: none) = place(
  top + left, dx: p.at(0) * s - r, dy: p.at(1) * s - r,
  circle(radius: r, fill: fill, stroke: stroke),
)

#let closed(s, pts, stroke) = {
  // pts: start, then triples (c1, c2, end)
  let items = (curve.move(pt(s, pts.at(0))),)
  let i = 1
  while i + 2 < pts.len() + 1 and i + 2 <= pts.len() {
    items.push(curve.cubic(pt(s, pts.at(i)), pt(s, pts.at(i + 1)), pt(s, pts.at(i + 2))))
    i += 3
  }
  place(top + left, curve(stroke: stroke, fill: none, ..items))
}

#let steps(s, p0, p1, p2, p3, ts, r, fill) = for t in ts { dot(s, bez(p0, p1, p2, p3, t), r, fill: fill) }

#let palette(dark) = if dark {
  (bound: 0.55pt + paper.transparentize(25%), slow: 0.85pt + ivory-blue, step: paper,
   fast: 1.15pt + mineral-light, node: mineral-light, end: paper, bg: ink,
   ghost: (paint: paper.transparentize(45%), thickness: 0.7pt, dash: (2.2pt, 2pt)))
} else {
  (bound: 0.5pt + petroleum, slow: 0.75pt + petroleum, step: petroleum,
   fast: 0.95pt + mineral, node: mineral, end: ink, bg: paper,
   ghost: (paint: grey, thickness: 0.7pt, dash: (2.2pt, 2pt)))
}

// Cover: the slow path arrives at a node and fans into three symbols.
#let motif-fan(w, dark: true) = {
  let s = w / 100
  let c = palette(dark)
  box(width: w, height: 70 * s, {
    closed(s, ((3, 44), (1, 30), (22, 31), (40, 20), (58, 9), (70, 2), (84, 3), (99, 4), (100, 22), (99, 35),
      (98, 53), (80, 63), (56, 67), (30, 71), (5, 62), (3, 44)), c.bound)
    let (a, b, d, e) = ((3, 44.5), (24, 43), (40, 33), (63, 34))
    seg(s, a, b, d, e, c.slow)
    steps(s, a, b, d, e, (0.14, 0.27, 0.40, 0.53, 0.65, 0.76, 0.87), 1.9pt, c.step)
    seg(s, (63, 34), (73, 34), (71, 19), (85, 19), c.fast)
    seg(s, (63, 34), (70, 34), (78, 34.6), (85.2, 34), c.fast)
    seg(s, (63, 34), (73, 34), (71, 49), (85, 49), c.fast)
    dot(s, (63, 34), 2.9pt, fill: c.node)
    // three symbols: square, circle, triangle
    place(top + left, dx: 85 * s, dy: 19 * s - 2.1 * s, rect(width: 4.2 * s, height: 4.2 * s, fill: none, stroke: 0.9pt + c.end))
    dot(s, (87.4, 34), 2.2 * s, fill: c.end)
    place(top + left, dx: 85 * s, dy: 49 * s - 2.3 * s, polygon(
      fill: none, stroke: 0.9pt + c.end, (2.3 * s, 0pt), (4.6 * s, 4.2 * s), (0pt, 4.2 * s),
    ))
  })
}

// Select and rejoin: the decision path branches, one branch is taken, the paths meet again.
#let motif-select(w, dark: false, unfinished: false) = {
  let s = w / 100
  let c = palette(dark)
  let back = if unfinished { c.ghost } else { c.fast }
  box(width: w, height: 62 * s, {
    closed(s, ((2, 34), (2, 14), (30, 13), (48, 6), (66, -1), (96, 3), (98, 24), (100, 45), (84, 60), (56, 60),
      (28, 60), (2, 54), (2, 34)), c.bound)
    let (a, b, d, e) = ((2.2, 37), (18, 35), (30, 29.5), (50, 29.5))
    seg(s, a, b, d, e, c.slow)
    steps(s, a, b, d, e, (0.16, 0.31, 0.46, 0.60, 0.73, 0.86), 1.9pt, c.step)
    seg(s, (50, 29.5), (58, 29.5), (60, 14), (70, 14), c.fast)
    seg(s, (50, 29.5), (58, 29.5), (60, 45), (70, 45), c.fast)
    seg(s, (50, 29.5), (57, 29.5), (63, 29.5), (70, 29.5), c.fast)
    seg(s, (70, 14), (80, 14), (84, 29.5), (92, 29.5), back)
    seg(s, (70, 45), (80, 45), (84, 29.5), (92, 29.5), back)
    seg(s, (70, 29.5), (77, 29.5), (85, 29.5), (92, 29.5), back)
    dot(s, (50, 29.5), 2.9pt, fill: c.node)
    dot(s, (70, 14), 2.1pt, fill: c.bg, stroke: 0.9pt + c.node)
    dot(s, (70, 45), 2.1pt, fill: c.bg, stroke: 0.9pt + c.node)
    dot(s, (70, 29.5), 2.6pt, fill: c.node)
    dot(s, (92, 29.5), 2.3pt, fill: c.bg, stroke: if unfinished { c.ghost } else { 0.9pt + c.end })
  })
}

// A loop of stepped points with one entry: the training loop.
#let motif-loop(w, dark: true) = {
  let s = w / 100
  let c = palette(dark)
  box(width: w, height: 62 * s, {
    closed(s, ((2, 30), (2, 10), (34, 4), (56, 5), (80, 6), (99, 14), (98, 34), (97, 54), (70, 60), (48, 59),
      (22, 58), (2, 50), (2, 30)), c.bound)
    seg(s, (2, 32), (10, 32), (20, 32), (30.5, 32), c.fast)
    dot(s, (30.5, 32), 2.9pt, fill: c.node)
    // the loop: four arcs
    let k = 0.5523
    let (cx, cy, rx, ry) = (58, 32, 27, 17)
    let arcs = (
      ((cx - rx, cy), (cx - rx, cy - k * ry), (cx - k * rx, cy - ry), (cx, cy - ry)),
      ((cx, cy - ry), (cx + k * rx, cy - ry), (cx + rx, cy - k * ry), (cx + rx, cy)),
      ((cx + rx, cy), (cx + rx, cy + k * ry), (cx + k * rx, cy + ry), (cx, cy + ry)),
      ((cx, cy + ry), (cx - k * rx, cy + ry), (cx - rx, cy + k * ry), (cx - rx, cy)),
    )
    for (i, arc) in arcs.enumerate() {
      seg(s, ..arc, c.slow)
      steps(s, ..arc, if i == 0 { (0.3, 0.55, 0.8) } else { (0.05, 0.3, 0.55, 0.8) }, 1.9pt, c.step)
    }
  })
}

// The three decisions, close: a short slow path, then the fan and its three symbols.
#let motif-three(w, dark: true) = {
  let s = w / 100
  let c = palette(dark)
  box(width: w, height: 66 * s, {
    closed(s, ((2, 36), (2, 16), (30, 8), (54, 5), (80, 2), (99, 12), (98, 34), (97, 56), (72, 64), (48, 63),
      (24, 62), (2, 56), (2, 36)), c.bound)
    let (a, b, d, e) = ((2, 36), (10, 36), (18, 34), (30, 34))
    seg(s, a, b, d, e, c.slow)
    steps(s, a, b, d, e, (0.22, 0.48, 0.74), 1.9pt, c.step)
    seg(s, (30, 34), (47, 34), (45, 14), (66, 14), c.fast)
    seg(s, (30, 34), (42, 34), (54, 34), (66, 34), c.fast)
    seg(s, (30, 34), (47, 34), (45, 54), (66, 54), c.fast)
    dot(s, (30, 34), 2.9pt, fill: c.node)
    place(top + left, dx: 66 * s, dy: 14 * s - 3.4 * s, rect(width: 6.8 * s, height: 6.8 * s, fill: none, stroke: 1pt + c.end))
    dot(s, (69.6, 34), 3.6 * s, fill: c.end)
    place(top + left, dx: 66 * s, dy: 54 * s - 3.7 * s, polygon(
      fill: none, stroke: 1pt + c.end, (3.7 * s, 0pt), (7.4 * s, 6.8 * s), (0pt, 6.8 * s),
    ))
  })
}
