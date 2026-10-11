"""
DRL Assignment 1 - Group 152
Demo code for: "A Survey on Practical Applications of Multi-Armed and
Contextual Bandits" (Bouneffouf & Rish, 2019, arXiv:1904.10040)

The paper is a SURVEY (no experiments of its own), so this demo reproduces the
core algorithms it describes, on synthetic data, one module per team member:

  Module 1 (mab)     Classical MAB   : epsilon-greedy, UCB1, Thompson Sampling   (Sec 1)
  Module 2 (cmab)    Contextual MAB  : LinUCB, Contextual Thompson Sampling      (Sec 1, 2.1, 2.4)
  Module 3 (nonstat) Non-stationary  : UCB1 vs Discounted-UCB vs Sliding-Window  (Sec 2.8, 2.10 / Table 1 gap)
  Module 4 (hpo)     Bandit for ML   : Successive Halving (Hyperband core) vs
                                       random search                             (Sec 3.2)

Usage:
    pip install numpy matplotlib
    python bandit_survey_demo.py                 # run everything
    python bandit_survey_demo.py --module mab    # run one module
    python bandit_survey_demo.py --seed 7 --out results

Outputs: PNG plots + printed summary tables (use these as screenshots/proof of work).
"""
import argparse
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Module 1: classical stochastic MAB (Bernoulli arms)
# ----------------------------------------------------------------------------
def run_mab(policy, probs, T, rng, eps=0.1):
    K = len(probs)
    counts = np.zeros(K)
    sums = np.zeros(K)
    alpha = np.ones(K)  # Thompson Sampling Beta posterior
    beta = np.ones(K)
    best = probs.max()
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        if policy == "eps-greedy":
            if rng.random() < eps or counts.min() == 0:
                a = rng.integers(K)
            else:
                a = int(np.argmax(sums / counts))
        elif policy == "UCB1":
            if t < K:
                a = t
            else:
                a = int(np.argmax(sums / counts + np.sqrt(2 * np.log(t + 1) / counts)))
        elif policy == "Thompson":
            a = int(np.argmax(rng.beta(alpha, beta)))
        else:
            raise ValueError(policy)
        r = float(rng.random() < probs[a])
        counts[a] += 1
        sums[a] += r
        alpha[a] += r
        beta[a] += 1 - r
        total += best - probs[a]
        regret[t] = total
    return regret


def module_mab(args):
    print("\n=== Module 1: Classical MAB (Bernoulli, K=10) ===")
    T, runs, K = 3000, 30, 10
    rng = np.random.default_rng(args.seed)
    probs = np.sort(rng.uniform(0.1, 0.9, K))[::-1]
    print("True arm probabilities:", np.round(probs, 2))
    results = {}
    for pol in ["eps-greedy", "UCB1", "Thompson"]:
        results[pol] = np.mean([run_mab(pol, probs, T, rng) for _ in range(runs)], axis=0)
    plt.figure(figsize=(7, 4.5))
    for pol, reg in results.items():
        plt.plot(reg, label=pol)
    plt.xlabel("Round t"); plt.ylabel("Cumulative regret")
    plt.title(f"Module 1: MAB regret (K={K}, mean of {runs} runs)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    path = os.path.join(args.out, "1_mab_regret.png"); plt.savefig(path, dpi=150); plt.close()
    print(f"{'Policy':<12}{'Final regret':>14}")
    for pol, reg in results.items():
        print(f"{pol:<12}{reg[-1]:>14.1f}")
    print("Saved:", path)


# ----------------------------------------------------------------------------
# Module 2: contextual bandits (linear rewards)
# ----------------------------------------------------------------------------
def run_cmab(policy, theta, T, rng, alpha=1.0, v=0.5, noise=0.1):
    K, d = theta.shape
    A = [np.eye(d) for _ in range(K)]
    b = [np.zeros(d) for _ in range(K)]
    # context-free UCB1 baseline stats
    counts = np.zeros(K); sums = np.zeros(K)
    regret = np.zeros(T); total = 0.0
    for t in range(T):
        x = rng.normal(size=d); x /= np.linalg.norm(x)
        exp_r = theta @ x
        if policy == "LinUCB":
            scores = []
            for a in range(K):
                Ainv = np.linalg.inv(A[a])
                mu = Ainv @ b[a]
                scores.append(mu @ x + alpha * np.sqrt(x @ Ainv @ x))
            a = int(np.argmax(scores))
        elif policy == "CTS":  # linear Thompson Sampling (Agrawal & Goyal 2013)
            scores = []
            for a in range(K):
                Ainv = np.linalg.inv(A[a])
                mu = Ainv @ b[a]
                th = rng.multivariate_normal(mu, v ** 2 * Ainv)
                scores.append(th @ x)
            a = int(np.argmax(scores))
        elif policy == "UCB1 (no context)":
            a = t if t < K else int(np.argmax(sums / counts + np.sqrt(2 * np.log(t + 1) / counts)))
        else:
            raise ValueError(policy)
        r = exp_r[a] + noise * rng.normal()
        A[a] += np.outer(x, x); b[a] += r * x
        counts[a] += 1; sums[a] += r
        total += exp_r.max() - exp_r[a]
        regret[t] = total
    return regret


def module_cmab(args):
    print("\n=== Module 2: Contextual MAB (K=5 arms, d=5 context) ===")
    print("Story: arms = treatments / news articles, context = patient / user features.")
    T, runs, K, d = 2000, 10, 5, 5
    rng = np.random.default_rng(args.seed)
    theta = rng.normal(size=(K, d))
    results = {}
    for pol in ["LinUCB", "CTS", "UCB1 (no context)"]:
        results[pol] = np.mean([run_cmab(pol, theta, T, rng) for _ in range(runs)], axis=0)
    plt.figure(figsize=(7, 4.5))
    for pol, reg in results.items():
        plt.plot(reg, label=pol)
    plt.xlabel("Round t"); plt.ylabel("Cumulative regret")
    plt.title(f"Module 2: Contextual bandit regret (mean of {runs} runs)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    path = os.path.join(args.out, "2_cmab_regret.png"); plt.savefig(path, dpi=150); plt.close()
    print(f"{'Policy':<20}{'Final regret':>14}")
    for pol, reg in results.items():
        print(f"{pol:<20}{reg[-1]:>14.1f}")
    print("Saved:", path)


# ----------------------------------------------------------------------------
# Module 3: non-stationary MAB (abrupt change, e.g. adversarial anomaly drift)
# ----------------------------------------------------------------------------
def run_nonstat(policy, segments, T, rng, gamma=0.998, window=1000, c=0.5):
    """All three policies use the same exploration constant c so the comparison
    isolates the *forgetting* mechanism (none / exponential discount / sliding window)."""
    K = len(segments[0])
    seg_len = T // len(segments)
    counts = np.zeros(K); sums = np.zeros(K)
    hist = deque()
    regret = np.zeros(T); total = 0.0
    for t in range(T):
        p = segments[min(t // seg_len, len(segments) - 1)]
        if counts.min() < 1e-9 or (policy != "Discounted-UCB" and counts.min() < 1):
            a = int(np.argmin(counts))
        else:
            n = counts.sum()
            a = int(np.argmax(sums / counts + np.sqrt(c * np.log(max(n, 2)) / counts)))
        r = float(rng.random() < p[a])
        if policy == "Discounted-UCB":
            counts *= gamma; sums *= gamma
            counts[a] += 1; sums[a] += r
        elif policy == "Sliding-Window-UCB":
            counts[a] += 1; sums[a] += r
            hist.append((a, r))
            if len(hist) > window:
                oa, orr = hist.popleft()
                counts[oa] -= 1; sums[oa] -= orr
        else:  # plain UCB (no forgetting)
            counts[a] += 1; sums[a] += r
        total += p.max() - p[a]
        regret[t] = total
    return regret


def module_nonstat(args):
    print("\n=== Module 3: Non-stationary MAB (best arm changes 3 times) ===")
    print("Story: Sec 2.10 says anomaly detection / telecom need drift handling.")
    T, runs = 20000, 8
    rng = np.random.default_rng(args.seed)
    base = np.array([0.9, 0.5, 0.4, 0.3, 0.1])
    segments = [np.roll(base, k) for k in range(4)]   # best arm moves: 0 -> 1 -> 2 -> 3
    names = ["UCB (no forgetting)", "Discounted-UCB", "Sliding-Window-UCB"]
    results = {}
    for pol in names:
        results[pol] = np.mean([run_nonstat(pol, segments, T, rng) for _ in range(runs)], axis=0)
    plt.figure(figsize=(7, 4.5))
    for pol, reg in results.items():
        plt.plot(reg, label=pol)
    for k in range(1, 4):
        plt.axvline(k * T // 4, color="k", ls="--", lw=1, label="environment change" if k == 1 else None)
    plt.xlabel("Round t"); plt.ylabel("Cumulative dynamic regret")
    plt.title(f"Module 3: Non-stationary MAB (mean of {runs} runs)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    path = os.path.join(args.out, "3_nonstationary_regret.png"); plt.savefig(path, dpi=150); plt.close()
    print(f"{'Policy':<22}{'Final regret':>14}")
    for pol, reg in results.items():
        print(f"{pol:<22}{reg[-1]:>14.1f}")
    print("Saved:", path)


# ----------------------------------------------------------------------------
# Module 4: bandits for ML - hyperparameter optimisation (Hyperband core idea)
# ----------------------------------------------------------------------------
def make_configs(n, rng):
    final = rng.uniform(0.1, 0.9, n)      # true final loss of each configuration
    slope = rng.uniform(0.05, 0.4, n)     # how slowly it converges
    return final, slope


def observed_loss(final, slope, r, rng):
    return final + slope / np.sqrt(r) + 0.02 * rng.normal(size=np.shape(final))


def successive_halving(final, slope, R, eta, rng):
    """Returns (index of chosen config, total budget spent)."""
    n = len(final)
    s = int(round(np.log(n) / np.log(eta)))
    alive = np.arange(n); spent = 0
    for i in range(s + 1):
        r_i = R * eta ** (i - s)
        spent += len(alive) * r_i
        obs = observed_loss(final[alive], slope[alive], max(r_i, 1e-9), rng)
        keep = max(int(len(alive) / eta), 1)
        alive = alive[np.argsort(obs)[:keep]] if i < s else alive[[np.argmin(obs)]]
    return int(alive[0]), spent


def module_hpo(args):
    print("\n=== Module 4: Hyperparameter optimisation as a bandit (Sec 3.2) ===")
    n, R, eta, trials = 81, 81, 3, 300
    rng = np.random.default_rng(args.seed)
    sh_loss, rs_loss, best_loss = [], [], []
    for _ in range(trials):
        final, slope = make_configs(n, rng)
        idx, spent = successive_halving(final, slope, R, eta, rng)
        sh_loss.append(final[idx])
        n_rs = max(int(spent // R), 1)            # random search: same total budget, full R each
        cand = rng.choice(n, n_rs, replace=False)
        obs = observed_loss(final[cand], slope[cand], R, rng)
        rs_loss.append(final[cand[np.argmin(obs)]])
        best_loss.append(final.min())
    print(f"Budget per trial ~ {spent:.0f} resource units (same for both methods)")
    print(f"Random search evaluates {n_rs} configs fully; Successive Halving screens {n}.")
    print(f"{'Method':<20}{'Mean true loss of chosen config':>34}")
    print(f"{'Oracle (best)':<20}{np.mean(best_loss):>34.4f}")
    print(f"{'Successive Halving':<20}{np.mean(sh_loss):>34.4f}")
    print(f"{'Random search':<20}{np.mean(rs_loss):>34.4f}")
    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot([best_loss, sh_loss, rs_loss])
    plt.xticks([1, 2, 3], ["Oracle", "Succ. Halving", "Random search"])
    plt.ylabel("True final loss of selected config (lower is better)")
    plt.title(f"Module 4: equal budget, {trials} trials")
    plt.grid(alpha=.3); plt.tight_layout()
    path = os.path.join(args.out, "4_hpo_successive_halving.png"); plt.savefig(path, dpi=150); plt.close()
    print("Saved:", path)


# ----------------------------------------------------------------------------
MODULES = {"mab": module_mab, "cmab": module_cmab, "nonstat": module_nonstat, "hpo": module_hpo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="all", choices=["all"] + list(MODULES))
    ap.add_argument("--seed", type=int, default=152)  # group number as default seed
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in MODULES.items():
        if args.module in ("all", name):
            fn(args)
    print("\nDone. Plots saved in:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
