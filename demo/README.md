# Live demo: Fair Machine Unlearning

A one-page local demo of the paper (Oesterling et al., AISTATS 2024) built on this
repo's own code. No new dependencies: it uses the standard library plus torch, numpy,
pandas and scikit-learn, which the repo already needs.

## Run it

From the repo root:

```bash
./run_demo.sh            # uses .venv/bin/python if it exists
# or
.venv/bin/python demo/app.py
```

It trains on COMPAS and Adult once (about 5 to 15 s), then opens
<http://localhost:8257>. Stop it with Ctrl+C.

Options: `--port 9000`, `--no-browser`, and `--check` (prints one COMPAS run in the
terminal and exits, a quick sanity check before presenting).

The page loads the Sora and DM Mono fonts from Google Fonts. Offline it falls back to
system fonts and still works.

## What it does

1. Loads COMPAS or Adult with `src/data/datasets.load_dataset` (split seed 0).
2. Trains the fair model (BCE + L2 + gamma * Berk pairwise penalty, paper
   hyperparameters: COMPAS gamma = 10, Adult gamma = 1, lambda = 1e-4) and the regular
   model (gamma = 0) with `src/objectives/train.train_model`. Optional noise
   b ~ N(0, sigma^2 I) is added to the objective, the same b for training and retraining.
3. Deletes the chosen % of training rows (random, or only from one race group).
4. For each model: retrains from scratch on the remaining rows, and runs one Newton step
   with `src/unlearning/newton.newton_unlearn`. Reports time, test accuracy, test AEOD,
   and the weight gap to the retrained model.
5. For the fair model it also shows a "naive" step that removes the deleted rows from
   the BCE term but keeps the pairwise fairness penalty at its old value. Its large gap
   shows why the pairwise term cannot be unlearned by subtracting per-row gradients.
6. With sigma > 0 it reports epsilon from `src/unlearning/certificate.certificate`
   (data-dependent bound, delta = 1e-4), a post-hoc estimate as documented there.

Nothing is written to disk and no existing results are changed. All code is in
`demo/app.py` (backend) and `demo/index.html` (page).
