#!/bin/bash
# Build the progress report. Needs Typst (0.15) on PATH; fonts are in ./fonts and on the system (Red Hat Text, Red Hat Mono).
cd "$(dirname "$0")" && typst compile --font-path fonts main.typ QwenDec-progress-report.pdf
