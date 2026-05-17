# Companion paper (LaTeX skeleton)

A 3-4 page arXiv companion describing the algorithms in `cvar-leximin-fair`.
Build with:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The text intentionally tracks the v0.1.0 API; for substantive theoretical
results we plan a separate writeup. Contributions welcome via PRs.
