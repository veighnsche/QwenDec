// Diagram language: token squares for slow reasoning, three glyphs for the
// decisions, dashes for connections that do not exist yet.
#import "design.typ": *

#let at(x, y, body) = place(top + left, dx: x, dy: y, body)
#let centred(x, y, body) = place(top + left, dx: x, dy: y, context {
  let m = measure(body)
  move(dx: -m.width / 2, dy: -m.height / 2, body)
})
#let ln(stroke, ..pts) = {
  let p = pts.pos()
  place(top + left, curve(stroke: stroke, fill: none, curve.move(p.at(0)), ..p.slice(1).map(q => curve.line(q))))
}
#let solid = 0.95pt + mineral
#let open-dash = (paint: amber, thickness: 0.95pt, dash: (3.2pt, 2.2pt))
#let build-dash = (paint: grey, thickness: 0.8pt, dash: (1.6pt, 1.6pt))
#let small(body, fill: ink) = text(font: sans, size: 9.5pt, fill: fill, body)
#let tiny(body, fill: soft) = text(font: sans, size: 8pt, fill: fill, body)

#let token(size, filled: true, c: petroleum) = rect(
  width: size, height: size, radius: 0.7pt,
  fill: if filled { c } else { none }, stroke: if filled { none } else { 0.7pt + c },
)

// Noul: two states. Choice: one of three. Score: a marker on a scale.
#let noul-glyph(u: 3.4mm, c: mineral) = box(width: 2.4 * u, height: u, {
  at(0pt, 0pt, rect(width: u, height: u, radius: 1pt, fill: c))
  at(1.4 * u, 0pt, rect(width: u, height: u, radius: 1pt, fill: none, stroke: 0.9pt + ink))
})
#let choice-glyph(u: 3.4mm, c: mineral) = box(width: 3.8 * u, height: u, {
  at(0pt, 0pt, circle(radius: u / 2, fill: none, stroke: 0.9pt + ink))
  at(1.4 * u, 0pt, circle(radius: u / 2, fill: c))
  at(2.8 * u, 0pt, circle(radius: u / 2, fill: none, stroke: 0.9pt + ink))
})
#let score-glyph(u: 3.4mm, c: mineral, value: 0.72) = box(width: 5.4 * u, height: u, {
  ln(0.8pt + ink, (0pt, u / 2), (5.4 * u, u / 2))
  for f in (0, 0.25, 0.5, 0.75, 1) {
    let h = if f == 0 or f == 1 { u * 0.9 } else { u * 0.45 }
    ln(0.8pt + ink, (5.4 * u * f, (u - h) / 2), (5.4 * u * f, (u + h) / 2))
  }
  at(5.4 * u * value - u * 0.42, u * 0.08, circle(radius: u * 0.42, fill: c))
})

// The goal and its state: one model, one row of tokens, three decisions.
#let one-model(w: 120mm) = {
  let h = 63mm
  let u = 3.4mm
  let row-y = 15mm
  let g-y = 41mm
  let (nx, cx, sx) = (22mm, 56mm, 92mm)
  box(width: w, height: h, {
    at(0pt, 0pt, rect(width: w, height: h, radius: 3pt, fill: none, stroke: 0.6pt + petroleum))
    at(7mm, 5.5mm, small[LLM])
    ln(0.6pt + petroleum, (7mm, row-y + u / 2), (88mm, row-y + u / 2))
    for i in range(6) { at(26mm + i * 8mm, row-y, token(u)) }
    for i in range(3) { at(92mm + i * 2.6mm, row-y + u / 2 - 0.5mm, circle(radius: 0.5mm, fill: petroleum)) }
    // requests go down as vectors: working for Noul and Choice, open for Score
    ln(solid, (60mm, row-y + u + 1.2mm), (60mm, 28mm), (nx, 28mm), (nx, g-y - 3mm))
    ln(solid, (cx, 28mm), (cx, g-y - 3mm))
    ln(open-dash, (68mm, row-y + u + 1.2mm), (68mm, 28mm), (sx, 28mm), (sx, g-y - 3mm))
    at(7mm, 22.8mm, tiny[request, as vectors])
    centred(nx, g-y, noul-glyph(u: u))
    centred(cx, g-y, choice-glyph(u: u))
    centred(sx, g-y, score-glyph(u: u, c: amber))
    centred(nx, g-y + 7.5mm, small[Noul])
    centred(cx, g-y + 7.5mm, small[Choice])
    centred(sx, g-y + 7.5mm, small[Score])
    // the way back into reasoning is not built
    ln(build-dash, (nx, g-y + 11.5mm), (nx, 55mm), (112mm, 55mm), (112mm, row-y + u / 2), (102mm, row-y + u / 2))
    ln(build-dash, (cx, g-y + 11.5mm), (cx, 55mm))
    ln(build-dash, (sx, g-y + 11.5mm), (sx, 55mm))
    ln(0.8pt + grey, (103.6mm, row-y + u / 2 - 1.2mm), (102mm, row-y + u / 2), (103.6mm, row-y + u / 2 + 1.2mm))
    at(nx + 2mm, 56.3mm, tiny[result, back into reasoning])
  })
}

// A request: 16 vectors for the kind of question, and a pointer to the item's own name.
#let request-crossing(w: 120mm) = {
  let u = 3.2mm
  let pitch = 5.5mm
  let x0 = 0mm
  let row-y = 7mm
  let px = x0 + 9 * pitch
  box(width: w, height: 66mm, {
    at(0pt, 0pt, tiny[What the model has read])
    ln(0.6pt + petroleum, (0mm, row-y + u / 2), (w, row-y + u / 2))
    for i in range(20) {
      at(x0 + i * pitch, row-y, rect(width: u, height: u, radius: 0.7pt, fill: if i >= 9 and i <= 11 { mineral } else { petroleum }))
    }
    // the pointer
    ln(solid, (px - 0.6mm, row-y + u + 1.6mm), (px - 0.6mm, row-y + u + 3mm), (px + 2 * pitch + u + 0.6mm, row-y + u + 3mm), (px + 2 * pitch + u + 0.6mm, row-y + u + 1.6mm))
    at(px + 3 * pitch + 2mm, row-y + u + 1.4mm, tiny[the pointer: the item's own name])
    ln(solid, (px + pitch + u / 2, row-y + u + 3mm), (px + pitch + u / 2, 22mm), (96mm, 22mm), (96mm, 28mm))
    // the kind of request
    at(0pt, 27.2mm, tiny[16 vectors: the kind of request])
    for i in range(16) { at(i * 3.5mm, 32mm, rect(width: 2.5mm, height: 2.5mm, radius: 0.5pt, fill: mineral)) }
    ln(solid, (15 * 3.5mm + 2.5mm + 1.5mm, 33.25mm), (78mm, 33.25mm))
    // the decision
    at(78mm, 28mm, rect(width: 36mm, height: 14mm, radius: 3pt, fill: none, stroke: 0.9pt + mineral))
    centred(96mm, 35mm, small[Decision])
    ln(solid, (96mm, 42mm), (96mm, 49mm))
    centred(96mm, 52mm, small[a value, or a reason to decline])
    // back into reasoning: not built
    ln(build-dash, (96mm, 55.5mm), (96mm, 61mm), (12mm, 61mm))
    ln(0.8pt + grey, (13.6mm, 59.8mm), (12mm, 61mm), (13.6mm, 62.2mm))
    at(12mm, 62.6mm, tiny[result, back into reasoning: to build])
  })
}

// One attempt on a full scale of 1,182 questions; the tick is the text reference.
#let track(value, total: 1182, ref: 1174, w: 120mm, c: mineral) = box(width: w, height: 5mm, {
  ln(0.5pt + hair, (0pt, 2.5mm), (w, 2.5mm))
  at(0pt, 1.5mm, rect(width: w * value / total, height: 2mm, fill: c))
  ln(0.9pt + ink, (w * ref / total, 0.2mm), (w * ref / total, 4.8mm))
})

// Who drives the training loop.
#let loop-diagram(w: 120mm) = {
  let bw = 34mm
  let bh = 24mm
  let panel(title, caption, ours) = box(width: w, height: bh + 13mm, {
    at(0pt, 0pt, kicker(title))
    let y0 = 6mm
    at(0pt, y0, rect(width: bw, height: bh, radius: 3pt, fill: none, stroke: 0.6pt + petroleum))
    at(w - bw - 14mm, y0, rect(width: bw + 14mm, height: bh, radius: 3pt, fill: none, stroke: 0.6pt + petroleum))
    at(4mm, y0 + 3.5mm, small[CPU])
    at(w - bw - 10mm, y0 + 3.5mm, small[GPU])
    if ours {
      at(4mm, y0 + 15mm, tiny[asleep])
      ln(solid, (bw, y0 + bh / 2), (w - bw - 14mm, y0 + bh / 2))
      at(bw + 6mm, y0 + bh / 2 - 4.6mm, tiny[one launch])
      at(bw - 1.1mm, y0 + bh / 2 - 1.1mm, circle(radius: 1.1mm, fill: mineral))
      let (cx, cy, rx, ry) = (w - 19mm, y0 + bh / 2 + 1.5mm, 13mm, 7mm)
      at(cx - rx, cy - ry, ellipse(width: 2 * rx, height: 2 * ry, fill: none, stroke: 0.75pt + petroleum))
      for k in range(10) {
        let a = k * 36deg
        at(cx + rx * calc.cos(a) - 0.75mm, cy + ry * calc.sin(a) - 0.75mm, circle(radius: 0.75mm, fill: petroleum))
      }
    } else {
      at(4mm, y0 + 15mm, tiny[one core at 100%])
      for k in range(11) {
        ln(0.5pt + petroleum, (bw, y0 + 2.5mm + k * 1.9mm), (w - bw - 14mm, y0 + 2.5mm + k * 1.9mm))
      }
      for k in range(5) { at(w - bw - 4mm + k * 6mm, y0 + bh / 2 + 1mm, token(3mm)) }
    }
    at(0pt, y0 + bh + 2.2mm, note(caption))
  })
  panel("Ordinary training", [The CPU sends the GPU every operation of every step, one by one.], false)
  v(5mm)
  panel("QwenDec", [The CPU starts the loop and sleeps. Every step runs on the GPU.], true)
}

// One square per question; the hollow ones were answered wrong.
#let field(total, wrong, cols: 60, w: 120mm, c: mineral) = {
  let pitch = w / cols
  let size = pitch * 0.68
  let rows = calc.ceil(total / cols)
  box(width: w, height: rows * pitch, {
    for i in range(total) {
      let x = calc.rem(i, cols) * pitch
      let y = calc.quo(i, cols) * pitch
      if i < total - wrong {
        at(x, y, rect(width: size, height: size, fill: c))
      } else {
        at(x, y, rect(width: size, height: size, fill: none, stroke: 0.5pt + ink))
      }
    }
  })
}
