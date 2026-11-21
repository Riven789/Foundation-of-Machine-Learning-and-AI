# Homework 3

Data 37711 — boosting, hyper-parameter selection and generalization. The assignment is `assignment.pdf`.

| Folder | Contents |
|---|---|
| `Problem1/` | Poisson boosting with regression trees (theory): `problem1.tex` / `problem1.pdf` |
| `Problem2/` | (a) test-set selection bound + proof, (b) what goes wrong when the same test set is reused, (c) wine-quality grid search with a 70/10/20 train/validation/test split (RandomForest and KNN, 600 settings each): `problem2.py`, `problem2.tex` / `problem2.pdf`, CSV results and figures |
| `Problem3/` | Gradient boosting with 1–1000 trees on `dset0` / `dset1`, plus label-noise and learning-curve diagnostics: `problem3.py`, `problem3.tex` / `problem3.pdf`, CSV results and figures |
| `data/` | `winequality-white.csv`, `dset0.csv`, `dset1.csv` (copies of the provided files) |

## Reproduce

```bash
pip install numpy pandas scikit-learn matplotlib
cd homework_3/Problem2 && python problem2.py   # ~ several minutes (1,200 grid fits + 20 x 600 KNN fits)
cd ../Problem3 && python problem3.py           # ~ a few minutes
```

Each script writes its CSVs and PNGs next to itself. The console output of the runs used in the
write-ups is saved as `problem2_output.txt` / `problem3_output.txt`. The PDFs are built with
`pdflatex problemN.tex`.
