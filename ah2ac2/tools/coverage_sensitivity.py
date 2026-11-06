"""coverage_sensitivity.py — sensitivity sweeps over modelling choices.

Reviewer concern: "the behaviour-stat density modelling and PCA-based diagnostics
depend on choices (mixture components, smoothing); sensitivity analyses are not
reported."

We sweep three knobs and report whether the headline conclusions survive:

* **K** — number of GMM components for the human density used in concentration
  diagnostics (Q1/Q2/Q3-stratification).
* **Covariance regularisation** — the per-cloud cov estimator used in the KL
  table: Ledoit-Wolf, MLE + jitter levels, fixed-λ shrinkage.
* **Laplace smoothing α** — the prior weight on each behaviour-stat
  distributional bin (default 0.5; tests 0.1, 0.5, 1.0). Rebuilds the
  behaviour-stat matrices with the chosen α and reruns ensemble KL coverage.

Outputs:
* ``sensitivity_summary.json`` — headline numbers under every setting
* ``sensitivity_plots.png`` — one panel per knob × representation
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numpy.linalg import cholesky, slogdet, solve
from safetensors.numpy import load_file

# ---------- Gaussian + utilities (with multiple cov regularisers) ------------


def _mle_cov(x: np.ndarray, jitter_frac: float):
    n, d = x.shape
    mu = x.mean(0)
    xc = x - mu
    S = (xc.T @ xc) / n
    S = S + jitter_frac * np.trace(S) / d * np.eye(d)
    return mu, S


def _lw_cov(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(0)
    xc = x - mu
    S = (xc.T @ xc) / n
    target = (np.trace(S) / d) * np.eye(d)
    norms_sq = np.sum(xc * xc, axis=1)
    xS = xc @ S
    xSx = np.sum(xc * xS, axis=1)
    b_bar = (np.sum(norms_sq**2) / n - 2 * np.sum(xSx) / n + np.sum(S * S)) / n
    diff = S - target
    sigma2 = np.sum(diff * diff)
    shrink = 0.0 if sigma2 < 1e-30 else float(np.clip(b_bar / sigma2, 0.0, 1.0))
    cov = shrink * target + (1.0 - shrink) * S + 1e-7 * np.trace(S) / d * np.eye(d)
    return mu, cov, float(shrink)


def _fixed_shrink_cov(x: np.ndarray, lam: float):
    n, d = x.shape
    mu = x.mean(0)
    xc = x - mu
    S = (xc.T @ xc) / n
    target = (np.trace(S) / d) * np.eye(d)
    cov = lam * target + (1.0 - lam) * S + 1e-7 * np.trace(S) / d * np.eye(d)
    return mu, cov


def fit_cov(x: np.ndarray, method: str):
    if method == "lw":
        mu, cov, _ = _lw_cov(x)
        return mu, cov
    elif method.startswith("mle_"):
        return _mle_cov(x, jitter_frac=float(method[4:]))
    elif method.startswith("fixed_"):
        return _fixed_shrink_cov(x, lam=float(method[6:]))
    else:
        raise ValueError(method)


def gauss_logpdf(x, mu, S):
    d = mu.shape[0]
    diff = x - mu
    L = cholesky(S)
    sol = solve(L, diff.T)
    _, logdet = slogdet(S)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + np.sum(sol * sol, axis=0))


def kl_gauss(mu0, S0, mu1, S1):
    d = mu0.shape[0]
    inv = solve(S1, S0)
    diff = (mu1 - mu0).reshape(-1, 1)
    quad = float((diff.T @ solve(S1, diff)).squeeze())
    s0, ld0 = slogdet(S0)
    s1, ld1 = slogdet(S1)
    if s0 <= 0 or s1 <= 0:
        return float("nan")
    return float(0.5 * (np.trace(inv) + quad - d + ld1 - ld0))


def _logsumexp(x, axis=0):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def gmm_logpdf(X, fit):
    K = len(fit["pis"])
    log_terms = np.stack(
        [
            np.log(fit["pis"][k] + 1e-300) + gauss_logpdf(X, fit["mus"][k], fit["covs"][k])
            for k in range(K)
        ],
        0,
    )
    return _logsumexp(log_terms, axis=0)


def fit_gmm(X, K, n_iter=150, seed=0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, size=K, replace=False)
    mus = [X[i].copy() for i in idx]
    base_cov = _lw_cov(X)[1]
    covs = [base_cov.copy() for _ in range(K)]
    pis = np.ones(K) / K
    prev_ll = -np.inf
    for it in range(n_iter):
        log_resp = np.stack(
            [np.log(pis[k] + 1e-300) + gauss_logpdf(X, mus[k], covs[k]) for k in range(K)], 0
        )
        log_norm = _logsumexp(log_resp, axis=0)
        log_resp = log_resp - log_norm
        resp = np.exp(log_resp)
        Nk = resp.sum(1) + 1e-12
        pis = Nk / N
        for k in range(K):
            mus[k] = (resp[k, :, None] * X).sum(0) / Nk[k]
            xc = X - mus[k]
            covs[k] = (resp[k, :, None] * xc).T @ xc / Nk[k]
            covs[k] += 1e-5 * np.trace(covs[k]) / D * np.eye(D)
        ll = float(log_norm.sum())
        if abs(ll - prev_ll) < 1e-3 * abs(ll + 1):
            break
        prev_ll = ll
    n_params = K * (D + D * (D + 1) / 2) + (K - 1)
    bic = -2 * ll + n_params * np.log(N)
    return {"pis": pis, "mus": mus, "covs": covs, "ll": ll, "bic": bic}


# ---------- ensemble mixture KL ---------------------------------------------


def optimise_alpha(
    component_means,
    component_covs,
    target_mu,
    target_S,
    rng,
    n_samples=10000,
    n_restarts=3,
    n_steps=300,
    lr=0.05,
):
    K = len(component_means)
    best_kl = np.inf
    best_a = np.ones(K) / K
    for restart in range(n_restarts):
        L = cholesky(target_S)
        x = target_mu + rng.standard_normal((n_samples, target_mu.shape[0])) @ L.T
        log_target = gauss_logpdf(x, target_mu, target_S)
        comp_lp = np.stack(
            [gauss_logpdf(x, component_means[i], component_covs[i]) for i in range(K)], 0
        )
        theta = rng.standard_normal(K) * 0.1
        m = np.zeros(K)
        v = np.zeros(K)
        b1, b2, eps = 0.9, 0.999, 1e-8
        prev_kl = np.inf
        for t in range(1, n_steps + 1):
            la = theta - _logsumexp(theta)
            a = np.exp(la)
            comp = la[:, None] + comp_lp
            log_mix = _logsumexp(comp, 0)
            kl = float(np.mean(log_target - log_mix))
            resp = np.exp(comp - log_mix[None, :])
            grad_la = -resp.mean(1)
            sm = np.exp(theta - _logsumexp(theta))
            grad = grad_la - sm * grad_la.sum()
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad**2
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            theta = theta - lr * mh / (np.sqrt(vh) + eps)
            if abs(prev_kl - kl) < 1e-6 and t > 50:
                break
            prev_kl = kl
        la = theta - _logsumexp(theta)
        a = np.exp(la)
        if kl < best_kl:
            best_kl, best_a = kl, a
    return float(best_kl), best_a


def ensemble_logpdf(X, names, alphas, base, cov_method):
    log_terms = []
    for nm, a in zip(names, alphas):
        B = np.load(f"{base}/{nm}_per_seat.npy").astype(np.float64)
        mu, S = fit_cov(B, cov_method)
        log_terms.append(np.log(a + 1e-300) + gauss_logpdf(X, mu, S))
    return _logsumexp(np.stack(log_terms, 0), axis=0)


# ---------- behaviour-stat rebuild with arbitrary α --------------------------

SLOT_BINS = list(range(1, 6))
RANK_BINS = list(range(1, 6))


def _norm_counter(c, bins, alpha):
    counts = np.array([c.get(b, 0) for b in bins], dtype=np.float64)
    counts = counts + alpha
    return counts / counts.sum()


def per_seat_vector(turn_states, seat, alpha):
    from ah2ac2.tools.hgroup_metrics import (
        compute_m1,
        compute_m2,
        compute_m3,
        compute_m5,
        compute_m6,
        compute_m8,
        compute_m9,
    )

    ts_seat = [ts for ts in turn_states if ts.seat == seat]
    m1 = _norm_counter(compute_m1(ts_seat), SLOT_BINS, alpha)
    m3 = _norm_counter(compute_m3(ts_seat), SLOT_BINS, alpha)
    m8 = _norm_counter(compute_m8(ts_seat), RANK_BINS, alpha)
    m5 = compute_m5(ts_seat)
    m2 = compute_m2(ts_seat)
    m9 = compute_m9(ts_seat)
    m6 = compute_m6(ts_seat)
    return np.concatenate([m1, m3, m8, [m5, m2, m9, m6]]).astype(np.float32)


def rebuild_behaviour_matrix(games_path, env, alpha, max_games=None):
    from ah2ac2.tools.hgroup_replay import replay_game

    raw = load_file(games_path)
    actions = np.asarray(raw["actions"])
    decks = np.asarray(raw["decks"])
    nactions = np.asarray(raw["num_actions"])
    N = actions.shape[0]
    if max_games is not None:
        N = min(N, max_games)
    rows = []
    for i in range(N):
        ts = replay_game(actions[i], decks[i], num_actions=int(nactions[i]), env=env)
        for seat in (0, 1):
            rows.append(per_seat_vector(ts, seat, alpha))
    return np.stack(rows, axis=0)


# ---------- main -------------------------------------------------------------

BC_NAMES = ["BC_proxy_1", "BC_proxy_2"]
LATENT_NAMES = ["Latent_proxy_1", "Latent_proxy_2", "Latent_proxy_3"]
ALL_BOTS = BC_NAMES + LATENT_NAMES


def sweep_K(H, base, bots, log_humans_baseline, Ks=(1, 2, 3, 4, 5)):
    out = []
    for K in Ks:
        f = fit_gmm(H, K, seed=0)
        log_h = gmm_logpdf(H, f)
        ratios = {}
        for bot in bots:
            B = np.load(f"{base}/{bot}_per_seat.npy").astype(np.float64)
            log_b = gmm_logpdf(B, f)
            ratios[bot] = float(np.exp(log_b.mean() - log_h.mean()))
        out.append(
            {
                "K": K,
                "BIC": float(f["bic"]),
                "weights": [round(w, 3) for w in sorted(f["pis"], reverse=True)],
                "concentration_ratio": ratios,
            }
        )
    return out


def sweep_cov(
    base,
    bots,
    n_human,
    kl_path,
    cov_methods=("lw", "mle_0.001", "mle_0.01", "mle_0.1", "fixed_0.1", "fixed_0.3", "fixed_0.5"),
):
    """For each cov method, recompute single-bot KLs and ensemble mixture KLs."""
    H = np.load(n_human).astype(np.float64)
    rng = np.random.default_rng(0)
    out = []
    kl = json.load(open(kl_path))
    bc_names = kl["ensemble_BC"]["components"]
    lat_names = kl["ensemble_Latent"]["components"]
    for method in cov_methods:
        mu_h, S_h = fit_cov(H, method)
        bot_fits = {
            bot: fit_cov(np.load(f"{base}/{bot}_per_seat.npy").astype(np.float64), method)
            for bot in bots
        }
        single = {bot: kl_gauss(bot_fits[bot][0], bot_fits[bot][1], mu_h, S_h) for bot in bots}
        single_rev = {bot: kl_gauss(mu_h, S_h, bot_fits[bot][0], bot_fits[bot][1]) for bot in bots}
        bc_kl, bc_a = optimise_alpha(
            [bot_fits[n][0] for n in bc_names],
            [bot_fits[n][1] for n in bc_names],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        lat_kl, lat_a = optimise_alpha(
            [bot_fits[n][0] for n in lat_names],
            [bot_fits[n][1] for n in lat_names],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        all_names = bc_names + lat_names
        all_kl, all_a = optimise_alpha(
            [bot_fits[n][0] for n in all_names],
            [bot_fits[n][1] for n in all_names],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        out.append(
            {
                "cov_method": method,
                "single_bot_fwd_KL": single,
                "single_bot_rev_KL": single_rev,
                "BC_ensemble_KL": bc_kl,
                "BC_alpha": [round(a, 3) for a in bc_a],
                "Latent_ensemble_KL": lat_kl,
                "Latent_alpha": [round(a, 3) for a in lat_a],
                "All_ensemble_KL": all_kl,
                "All_alpha": [round(a, 3) for a in all_a],
            }
        )
    return out


def sweep_alpha_for_behaviour(
    human_games_path, bot_rollouts_dir, alphas=(0.1, 0.5, 1.0), max_human=2500, max_bot=500, seed=0
):
    """Rebuild behaviour matrices with each Laplace α, recompute ensemble KLs."""
    from jaxmarl import make

    env = make("hanabi", num_agents=2)
    # Subsample humans deterministically
    raw = load_file(human_games_path)
    n_total = raw["actions"].shape[0]
    rng_np = np.random.RandomState(seed)
    if max_human is not None and max_human < n_total:
        idx = rng_np.choice(n_total, size=max_human, replace=False)
        idx.sort()
        sub_actions = raw["actions"][idx]
        sub_decks = raw["decks"][idx]
        sub_num_actions = raw["num_actions"][idx]
    else:
        sub_actions = raw["actions"]
        sub_decks = raw["decks"]
        sub_num_actions = raw["num_actions"]
    from safetensors.numpy import save_file

    tmp_dir = "outputs/coverage/_sensitivity_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    sub_path = f"{tmp_dir}/human_subsample_{max_human or 'full'}.safetensors"
    save_file(
        {
            "actions": sub_actions.astype(np.int32),
            "decks": sub_decks.astype(np.int32),
            "num_actions": sub_num_actions.astype(np.int32),
        },
        sub_path,
    )

    out = []
    for alpha in alphas:
        print(f"  building behaviour matrices for α={alpha}…")
        t0 = time.time()
        H = rebuild_behaviour_matrix(sub_path, env, alpha, max_games=max_human)
        h_mean = H.mean(0)
        h_std = H.std(0)
        h_std = np.where(h_std < 1e-6, 1.0, h_std)
        H_z = ((H - h_mean) / h_std).astype(np.float64)
        bot_zs = {}
        for bot in ALL_BOTS:
            gp = f"{bot_rollouts_dir}/{bot}/games.safetensors"
            B = rebuild_behaviour_matrix(gp, env, alpha, max_games=max_bot)
            bot_zs[bot] = ((B - h_mean) / h_std).astype(np.float64)
        elapsed = time.time() - t0
        print(f"    rebuild took {elapsed:.1f}s")

        # Fit covs (LW), compute ensemble KLs
        rng = np.random.default_rng(0)
        mu_h, S_h, _ = _lw_cov(H_z)
        bot_fits = {bot: _lw_cov(bot_zs[bot])[:2] for bot in ALL_BOTS}
        bc_kl, bc_a = optimise_alpha(
            [bot_fits[n][0] for n in BC_NAMES],
            [bot_fits[n][1] for n in BC_NAMES],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        lat_kl, lat_a = optimise_alpha(
            [bot_fits[n][0] for n in LATENT_NAMES],
            [bot_fits[n][1] for n in LATENT_NAMES],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        all_kl, all_a = optimise_alpha(
            [bot_fits[n][0] for n in ALL_BOTS],
            [bot_fits[n][1] for n in ALL_BOTS],
            mu_h,
            S_h,
            rng,
            n_samples=4000,
            n_restarts=2,
            n_steps=200,
        )
        # Stratified Q3 (rescue gradient)
        bc_alpha_canon = bc_a
        lat_alpha_canon = lat_a
        # build mixtures using fitted bot Gaussians
        log_bc = ensemble_logpdf_from_fits(H_z, BC_NAMES, bc_alpha_canon, bot_fits)
        log_lat = ensemble_logpdf_from_fits(H_z, LATENT_NAMES, lat_alpha_canon, bot_fits)
        delta = log_lat - log_bc
        edges = np.percentile(log_bc, [0, 25, 50, 75, 100])
        quartiles = np.digitize(log_bc, edges[1:-1], right=False)
        rescue = [float((delta[quartiles == q] > 0).mean()) for q in range(4)]
        bot10 = log_bc <= np.percentile(log_bc, 10)
        rescue_bottom10 = float((delta[bot10] > 0).mean())
        out.append(
            {
                "alpha": alpha,
                "n_humans_resampled": int(H.shape[0]),
                "BC_ensemble_KL": float(bc_kl),
                "BC_alpha": [round(a, 3) for a in bc_a],
                "Latent_ensemble_KL": float(lat_kl),
                "Latent_alpha": [round(a, 3) for a in lat_a],
                "All_ensemble_KL": float(all_kl),
                "All_alpha": [round(a, 3) for a in all_a],
                "rescue_quartiles_Q1_to_Q4": rescue,
                "rescue_bottom10_by_BC": rescue_bottom10,
            }
        )
    return out


def ensemble_logpdf_from_fits(X, names, alphas, bot_fits):
    log_terms = []
    for nm, a in zip(names, alphas):
        mu, S = bot_fits[nm]
        log_terms.append(np.log(a + 1e-300) + gauss_logpdf(X, mu, S))
    return _logsumexp(np.stack(log_terms, 0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/coverage/sensitivity.json")
    ap.add_argument("--skip_alpha", action="store_true", help="skip the slow Laplace-α sweep")
    ap.add_argument("--alpha_max_human", type=int, default=2500)
    ap.add_argument("--alpha_max_bot", type=int, default=500)
    args = ap.parse_args()

    results = {}

    print("====== Sensitivity 1: GMM K (mixture components) ======")
    print("\n-- VAE 32-d --")
    Hv = np.load("outputs/coverage/encodings/human_5000_per_seat.npy").astype(np.float64)
    sweep_v = sweep_K(Hv, "outputs/coverage/encodings", ALL_BOTS, None)
    for r in sweep_v:
        print(
            f"  K={r['K']}  BIC={r['BIC']:.0f}  conc-ratios={ {k: round(v, 2) for k, v in r['concentration_ratio'].items()} }"
        )
    results["K_sweep_VAE"] = sweep_v

    print("\n-- Behaviour 19-d --")
    Hb = np.load("outputs/coverage/behaviour_space_no_m10/Human_per_seat.npy").astype(np.float64)
    sweep_b = sweep_K(Hb, "outputs/coverage/behaviour_space_no_m10", ALL_BOTS, None)
    for r in sweep_b:
        print(
            f"  K={r['K']}  BIC={r['BIC']:.0f}  conc-ratios={ {k: round(v, 2) for k, v in r['concentration_ratio'].items()} }"
        )
    results["K_sweep_Behaviour"] = sweep_b

    print("\n====== Sensitivity 2: Covariance regularisation ======")
    print("\n-- VAE 32-d --")
    cov_v = sweep_cov(
        "outputs/coverage/encodings",
        ALL_BOTS,
        "outputs/coverage/encodings/human_5000_per_seat.npy",
        "outputs/coverage/vae_space/kl_table.json",
    )
    for r in cov_v:
        print(
            f"  {r['cov_method']:<12}  BC_ens={r['BC_ensemble_KL']:.2f}  Lat_ens={r['Latent_ensemble_KL']:.2f}  All_ens={r['All_ensemble_KL']:.2f}"
        )
    results["cov_sweep_VAE"] = cov_v

    print("\n-- Behaviour 19-d --")
    cov_b = sweep_cov(
        "outputs/coverage/behaviour_space_no_m10",
        ALL_BOTS,
        "outputs/coverage/behaviour_space_no_m10/Human_per_seat.npy",
        "outputs/coverage/behaviour_space_no_m10/kl_table.json",
    )
    for r in cov_b:
        print(
            f"  {r['cov_method']:<12}  BC_ens={r['BC_ensemble_KL']:.2f}  Lat_ens={r['Latent_ensemble_KL']:.2f}  All_ens={r['All_ensemble_KL']:.2f}"
        )
    results["cov_sweep_Behaviour"] = cov_b

    if not args.skip_alpha:
        print("\n====== Sensitivity 3: Laplace smoothing α (behaviour space) ======")
        print(
            f"(rebuilding behaviour matrices for α∈{{0.1, 0.5, 1.0}}; humans cap = {args.alpha_max_human}, bots cap = {args.alpha_max_bot})"
        )
        alpha_sweep = sweep_alpha_for_behaviour(
            "outputs/coverage/games/human_train.safetensors",
            "outputs/coverage/rollouts",
            alphas=(0.1, 0.5, 1.0),
            max_human=args.alpha_max_human,
            max_bot=args.alpha_max_bot,
        )
        for r in alpha_sweep:
            print(
                f"  α={r['alpha']}  BC_ens={r['BC_ensemble_KL']:.2f}  Lat_ens={r['Latent_ensemble_KL']:.2f}  All_ens={r['All_ensemble_KL']:.2f}  "
                f"rescue Q1..Q4={[round(x, 2) for x in r['rescue_quartiles_Q1_to_Q4']]}  bot10={r['rescue_bottom10_by_BC']:.2f}"
            )
        results["alpha_sweep_Behaviour"] = alpha_sweep

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
