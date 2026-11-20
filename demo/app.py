#!/usr/bin/env python3
"""
Live demo: Fair Machine Unlearning (Oesterling et al., AISTATS 2024).

Run from anywhere:   python demo/app.py        (or ./run_demo.sh)
Then open            http://localhost:8257     (opens automatically)

Uses only the Python standard library on top of the repo's own dependencies
(torch, numpy, pandas, scikit-learn). Reuses the repo code unchanged:
    src.data.datasets.load_dataset        COMPAS / Adult preprocessing
    src.objectives.train.train_model      L-BFGS on BCE + L2 + gamma * fairness
    src.unlearning.newton.newton_unlearn  one Newton step on the remaining data
    src.unlearning.certificate.certificate  (epsilon, delta) estimate
    src.evaluation.metrics.calculate_metrics  accuracy and AEOD
Nothing is written to disk; existing results are untouched.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)                     # load_dataset reads data/raw relative to cwd
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.datasets import load_dataset  # noqa: E402
from src.evaluation.metrics import calculate_metrics  # noqa: E402
from src.objectives.fair_loss import fair_logistic_loss  # noqa: E402
from src.objectives.train import train_model  # noqa: E402
from src.unlearning.certificate import certificate  # noqa: E402
from src.unlearning.newton import newton_unlearn  # noqa: E402

HERE = Path(__file__).resolve().parent
DELTA = 1e-4
MAX_ITER = 500

# Paper Table 1 (App. A.6), same values as experiments/run_deletion_sweep.py
PAPER_HP = {
    "compas": {"lambda_reg": 1e-4, "gamma": 10.0},
    "adult": {"lambda_reg": 1e-4, "gamma": 1.0},
}
GROUP_NAMES = {
    "compas": {0: "African-American", 1: "Caucasian"},
    "adult": {0: "Black", 1: "White"},
}

_lock = threading.Lock()
_data_cache = {}
_model_cache = {}


def get_data(name):
    if name not in _data_cache:
        _data_cache[name] = load_dataset(name, random_state=0)
    return _data_cache[name]


def make_noise(d, sigma, seed):
    if sigma <= 0:
        return None
    g = torch.Generator().manual_seed(10_000 + seed)
    return sigma * torch.randn(d, generator=g, dtype=torch.float64)


def get_original(name, gamma, lam, sigma, seed):
    """Model trained on ALL training rows (cached: it is the 'deployed' model)."""
    key = (name, gamma, lam, sigma, seed)
    if key not in _model_cache:
        D = get_data(name)
        noise = make_noise(D["X_train"].shape[1], sigma, seed)
        t0 = time.perf_counter()
        theta = train_model(D["X_train"], D["y_train"], D["sensitive_train"],
                            lambda_reg=lam, gamma=gamma, max_iter=MAX_ITER,
                            noise=noise)
        _model_cache[key] = (theta, noise, time.perf_counter() - t0)
    return _model_cache[key]


def pick_deleted(s, pct, mode, seed):
    """Indices of training rows to delete."""
    n = len(s)
    m = max(1, int(round(n * pct / 100)))
    g = torch.Generator().manual_seed(seed)
    if mode == "random":
        pool = torch.arange(n)
    else:
        group = 0 if mode == "group0" else 1
        pool = torch.nonzero(s == group).flatten()
        # never delete a whole group: the fairness term needs both
        m = min(m, len(pool) - 10)
    perm = pool[torch.randperm(len(pool), generator=g)]
    return torch.sort(perm[:m]).values


def naive_unlearn(theta, X, y, s, keep, lam, gamma, noise):
    """
    The 'subtract their gradient' shortcut applied to the fair loss: one
    Newton step that removes the deleted rows from the per-sample terms (BCE)
    but keeps the pairwise fairness penalty at its full-data value, as if it
    decomposed per sample. It does not, which is the paper's point.
    """
    Xr, yr = X[keep], y[keep]
    n_rem = Xr.shape[0]

    def obj(t):
        bce = F.binary_cross_entropy_with_logits(Xr @ t, yr)
        _, comp = fair_logistic_loss(t, X, y, s, lambda_reg=0.0, gamma=gamma)
        total = bce + gamma * comp["fairness"] + lam / 2 * torch.sum(t ** 2)
        if noise is not None:
            total = total + torch.dot(noise, t) / n_rem
        return total

    t = theta.detach().clone().requires_grad_(True)
    grad = torch.autograd.grad(obj(t), t)[0]
    hess = torch.autograd.functional.hessian(obj, theta.detach())
    return theta.detach() - torch.linalg.solve(hess, grad)


def metrics(theta, D):
    m = calculate_metrics(theta, D["X_test"], D["y_test"], D["sensitive_test"])
    return {"accuracy": m["accuracy"], "aeod": m["aeod"]}


def finite(x):
    return x if (x is not None and math.isfinite(x)) else None


def run_one(name, gamma, lam, sigma, seed, deleted, keep):
    D = get_data(name)
    X, y, s = D["X_train"], D["y_train"], D["sensitive_train"]
    theta0, noise, t_train = get_original(name, gamma, lam, sigma, seed)
    Xr, yr, sr = X[keep], y[keep], s[keep]

    t0 = time.perf_counter()
    theta_rt = train_model(Xr, yr, sr, lambda_reg=lam, gamma=gamma,
                           max_iter=MAX_ITER, noise=noise)
    t_retrain = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta_un, _ = newton_unlearn(theta0, Xr, yr, sr, lambda_reg=lam,
                                 gamma=gamma, noise=noise)
    t_unlearn = time.perf_counter() - t0

    cert = certificate(Xr, s, deleted, theta0, theta_un, lam,
                       sigma=sigma, delta=DELTA, fair=gamma > 0)

    gap = lambda t: float(torch.linalg.norm(t - theta_rt))  # noqa: E731
    out = {
        "gamma": gamma,
        "train_time": t_train,
        "original": {**metrics(theta0, D), "gap": gap(theta0)},
        "retrain": {**metrics(theta_rt, D), "time": t_retrain, "gap": 0.0},
        "unlearn": {**metrics(theta_un, D), "time": t_unlearn,
                    "gap": gap(theta_un)},
        "speedup": t_retrain / max(t_unlearn, 1e-9),
        "eps_prime": finite(cert["eps_prime_data"]),
        "epsilon": finite(cert["epsilon"]),
        "theta_norm": float(torch.linalg.norm(theta_rt)),
    }
    if gamma > 0:
        t0 = time.perf_counter()
        theta_nv = naive_unlearn(theta0, X, y, s, keep, lam, gamma, noise)
        out["naive"] = {**metrics(theta_nv, D),
                        "time": time.perf_counter() - t0,
                        "gap": gap(theta_nv)}
    return out


def run_demo(params):
    name = params.get("dataset", "compas")
    if name not in PAPER_HP:
        raise ValueError(f"unknown dataset {name}")
    pct = min(max(float(params.get("pct", 5)), 0.1), 40)
    sigma = min(max(float(params.get("sigma", 0)), 0), 100)
    seed = int(params.get("seed", 0))
    mode = params.get("mode", "random")
    hp = PAPER_HP[name]
    gamma = float(params.get("gamma", hp["gamma"]))
    lam = hp["lambda_reg"]

    D = get_data(name)
    s = D["sensitive_train"]
    deleted = pick_deleted(s, pct, mode, seed)
    keep = torch.ones(len(s), dtype=torch.bool)
    keep[deleted] = False

    names = GROUP_NAMES[name]
    return {
        "dataset": name,
        "n_train": int(len(s)),
        "n_test": int(len(D["y_test"])),
        "d": int(D["X_train"].shape[1]),
        "n_deleted": int(len(deleted)),
        "deleted_by_group": {names[g]: int((s[deleted] == g).sum())
                             for g in (0, 1)},
        "group_names": [names[0], names[1]],
        "pct": pct, "sigma": sigma, "lambda_reg": lam, "delta": DELTA,
        "mode": mode, "seed": seed,
        "fair": run_one(name, gamma, lam, sigma, seed, deleted, keep),
        "regular": run_one(name, 0.0, lam, sigma, seed, deleted, keep),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif url.path == "/api/run":
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                with _lock:
                    t0 = time.perf_counter()
                    result = run_demo(params)
                    result["wall_time"] = time.perf_counter() - t0
                self._send(200, json.dumps(result), "application/json")
            except Exception as exc:  # show the error in the page
                traceback.print_exc()
                self._send(500, json.dumps({"error": str(exc)}),
                           "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--port", type=int, default=8257)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="run once on COMPAS in the terminal and exit")
    args = ap.parse_args()

    print("Loading COMPAS and training the fair + regular models ...",
          flush=True)
    t0 = time.perf_counter()
    warm = run_demo({"dataset": "compas"})   # warm-up so the first click is fast
    if not args.check:
        print("Loading Adult ...", flush=True)
        run_demo({"dataset": "adult"})
    print(f"  ready in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.check:
        for k in ("fair", "regular"):
            r = warm[k]
            print(f"  {k:8s} gamma={r['gamma']:<5} "
                  f"retrain {r['retrain']['time']*1e3:7.1f} ms  "
                  f"newton {r['unlearn']['time']*1e3:6.1f} ms  "
                  f"gap {r['unlearn']['gap']:.2e}  "
                  f"AEOD {r['unlearn']['aeod']:.3f}  "
                  f"acc {r['unlearn']['accuracy']:.3f}")
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Demo running at {url}   (Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
