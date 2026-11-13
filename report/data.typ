// Every figure printed in the report, with the repository file it comes from.
// Pages take their numbers from here only.

#let d = (
  // latent/RESULTS4.md (pilot 4, renamed panel of 1,182 questions)
  noul: (
    right: "1,162", of: "1,182", text: "1,174",
    pairs: "425", pairs-of: "428", pairs-text: "427",
    pointer: "72", pointer-of: "72",
    free: "165", free-of: "184", free-text: "176",
    own-names: "1,160",
    held-back: "600", held-back-of: "600",
    errors: "20", errors-free: "19",
  ),
  // latent/RESULTS.md, RESULTS2.md, RESULTS3.md, RESULTS4.md
  pilots: (
    (n: "1", v: 1024, right: "1,024", passed: false),
    (n: "2", v: 1104, right: "1,104", passed: false),
    (n: "3", v: 1049, right: "1,049", passed: false),
    (n: "4", v: 1162, right: "1,162", passed: true),
  ),
  pilot1-chance: "35", pilot1-chance-of: "72",
  pilot2-needed: "1,115", pilot2-binding: "72",
  pilot3-free: "153", pilot3-free-of: "184",
  pilot3-same-panel: "1,132",
  // latent/RESULTS-CHOICE.md (1,001 development questions on 230 cases)
  choice: (
    right: "955", of: "1,001", text: "971",
    order: "109", order-of: "112", order-text: "103",
    pairs: "352", pairs-of: "374", pairs-text: "367",
    pick: "312", pick-of: "319", pick-text: "301",
    nopick: "90", nopick-of: "100", nopick-text: "100",
  ),
  // latent/RESULTS-JOINT.md, latent/JOINT.md (one run, 12 Nov 2026)
  joint: (
    noul: "1,128", choice: "929", noul-needed: "1,115", choice-needed: "921",
    parts: (
      ("Noul: a question and its negation", "385", "406", 21),
      ("Noul: all questions", "1,115", "1,128", 13),
      ("Noul: free-form questions", "151", "153", 2),
      ("Noul: the two items differ", "65", "72", 7),
      ("Noul: answer follows the pointer", "65", "72", 7),
      ("Choice: all questions", "921", "929", 8),
      ("Choice: pick and rule out, both right", "331", "342", 11),
      ("Choice: right in every order", "93", "105", 12),
    ),
  ),
  // README.md and baseline/results-fork-fork-v1.json (run fork-v1, 10 Nov 2026);
  // Score rows use the measure of latent/SCORE.md (picked level, with tolerance).
  fork: (
    rows: (
      ("Recognises \"unresolved\"", "2", "69", "81"),
      ("Recognises \"conflicting\"", "47", "80", "85"),
      ("Noul: status and value right", "194", "228", "243"),
      ("Choice: status and value right", "165", "225", "232"),
      ("Score: status and value right", "111", "133", "199"),
    ),
    score-between-before: "15", score-between-after: "11", score-between-of: "67",
  ),
  // latent/SCORE.md
  score: (status: "70", status-of: "80", ends: "52", ends-of: "52", between: "11", between-of: "67"),
  // train/RESULTS.md, gpu_loop/RESULTS.md
  loop: (
    cpu: "0.04%", gpu: "99.7%",
    eager-cpu: "99.5%", eager-gpu: "56%", eager-ms: "733 ms",
    loop-cpu: "0.4%", loop-gpu: "96%", loop-ms: "397 ms",
    steps-first: "1,034", steps-joint: "810",
  ),
)
