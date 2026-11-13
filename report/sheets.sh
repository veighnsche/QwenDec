#!/bin/bash
# Render the PDF to contact sheets for checking. Usage: sheets.sh OUTDIR [DPI]
out="${1:?output folder}"; dpi="${2:-50}"
mkdir -p "$out"
pdftoppm -r "$dpi" -png QwenDec-progress-report.pdf "$out/p" 2>/dev/null
python3 - "$out" <<'PY'
import sys, glob
from PIL import Image
out = sys.argv[1]
ims = [Image.open(f) for f in sorted(glob.glob(out + '/p-*.png'))]
w, h = ims[0].size
for k in range(0, len(ims), 8):
    sheet = Image.new('RGB', (w * 4 + 50, h * 2 + 30), 'white')
    for i, im in enumerate(ims[k:k + 8]):
        sheet.paste(im, ((i % 4) * (w + 10) + 10, (i // 4) * (h + 10) + 10))
    sheet.save(f'{out}/sheet{k // 8}.png')
print(len(ims), 'pages')
PY
