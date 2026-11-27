# Inference paper — LaTeX sources

Compile on Overleaf with pdfLaTeX and BibTeX. Upload this whole folder: `main.tex`,
`refs.bib`, `generated/` and `figures/`. IEEEtran ships with Overleaf.

Every number in the text and every table is generated from the raw CSVs. Do not edit
`generated/` or `figures/` by hand. After any change to the results, regenerate them:

```bash
.venv/bin/python tools/plot_inference.py      # figures -> results/figures/inference/ and paper/figures/
.venv/bin/python tools/inference_tables.py    # macros and tables -> paper/generated/
```

Comments starting `% RAVIV:` mark points that need his ruling. Bibliography entries marked
`VERIFY` in `refs.bib` were written from memory and should be checked before submission.
