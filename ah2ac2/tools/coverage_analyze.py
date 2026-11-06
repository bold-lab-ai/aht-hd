"""coverage_analyze.py — Stages 2, 3, 4 of the human-coverage analysis.

Loads per-seat encodings produced by ``coverage_encode.py`` (or any per-seat
matrix produced by ``coverage_behaviour.py``) and runs:

* Stage 2: PCA fit on the human cloud, projection of bot clouds.
* Stage 3: 2-D coverage figure (KDE backdrop + per-bot clouds + centroids).
* Stage 4: Ledoit-Wolf Gaussians per cloud, single-bot forward/reverse KL,
           ensemble mixture KL with simplex-α optimisation, bootstrap CIs,
           Mahalanobis + 2-D convex hull sanity checks.

The same script is reused for Stage 5 by feeding it behaviour-stat vectors
instead of VAE latents.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Cloud:
    name: str
    points: np.ndarray  # (N, D)


def fit_ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf shrinkage to a scaled-identity target. Returns (mean, cov)."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0)
    xc = x - mu
    S = (xc.T @ xc) / n  # MLE covariance
    trace_S = np.trace(S)
    target = (trace_S / d) * np.eye(d)
    # Numerator: average squared Frobenius dist between per-sample outer and S
    # Vectorised: b̄ = (1/n²) Σ_t ||x_t x_t.T - S||² = (1/n²) Σ_t (||x_t||⁴ - 2 x_t.T S x_t) + ||S||²
    # But simpler still: compute directly.
    norms_sq = np.sum(xc * xc, axis=1)  # (n,) ||x_t||²
    # ||x_t x_t.T||²_F = ||x_t||⁴
    # tr(x_t x_t.T S) = x_t.T S x_t
    xS = xc @ S
    xSx = np.sum(xc * xS, axis=1)  # (n,)
    S_frob_sq = np.sum(S * S)
    b_bar = (np.sum(norms_sq**2) / n - 2 * np.sum(xSx) / n + S_frob_sq) / n
    # Denominator: ||S - target||²_F
    diff = S - target
    sigma2 = np.sum(diff * diff)
    if sigma2 < 1e-30:
        shrink = 0.0
    else:
        shrink = float(np.clip(b_bar / sigma2, 0.0, 1.0))
    cov = shrink * target + (1.0 - shrink) * S
    # Tiny diagonal jitter for numerical PSD safety (cholesky-friendly)
    cov = cov + 1e-8 * np.trace(cov) / d * np.eye(d)
    return mu, cov


def kl_gaussian(mu0: np.ndarray, S0: np.ndarray, mu1: np.ndarray, S1: np.ndarray) -> float:
    """KL( N(mu0, S0) || N(mu1, S1) ) — closed form."""
    d = mu0.shape[0]
    _L1 = np.linalg.cholesky(S1)
    inv_S1_S0 = np.linalg.solve(S1, S0)
    diff = (mu1 - mu0).reshape(-1, 1)
    quad = float((diff.T @ np.linalg.solve(S1, diff)).squeeze())
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    if sign0 <= 0 or sign1 <= 0:
        return float("nan")
    return float(0.5 * (np.trace(inv_S1_S0) + quad - d + logdet1 - logdet0))


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """log N(x; mu, S) for x: (N, D)."""
    d = mu.shape[0]
    diff = x - mu
    L = np.linalg.cholesky(S)
    sol = np.linalg.solve(L, diff.T)  # (D, N)
    quad = np.sum(sol * sol, axis=0)  # (N,)
    sign, logdet = np.linalg.slogdet(S)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def sample_gaussian(rng: np.random.Generator, mu: np.ndarray, S: np.ndarray, n: int) -> np.ndarray:
    L = np.linalg.cholesky(S)
    eps = rng.standard_normal((n, mu.shape[0]))
    return mu + eps @ L.T


def estimate_kl_to_mixture(
    sampler_means: list[np.ndarray],
    sampler_covs: list[np.ndarray],
    target_mu: np.ndarray,
    target_S: np.ndarray,
    alphas: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10_000,
) -> float:
    """Monte-Carlo estimate of D(target ‖ Σ_i α_i N(μ_i, S_i)).

    Uses samples from the target distribution, evaluates log of the mixture
    analytically per sample.
    """
    x = sample_gaussian(rng, target_mu, target_S, n_samples)
    log_target = gaussian_logpdf(x, target_mu, target_S)
    # log of mixture density via logsumexp
    component_logp = np.stack(
        [
            np.log(alphas[i] + 1e-300) + gaussian_logpdf(x, sampler_means[i], sampler_covs[i])
            for i in range(len(sampler_means))
        ],
        axis=0,
    )
    log_mix = _logsumexp(component_logp, axis=0)
    return float(np.mean(log_target - log_mix))


def _logsumexp(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def optimise_mixture_alpha(
    component_means: list[np.ndarray],
    component_covs: list[np.ndarray],
    target_mu: np.ndarray,
    target_S: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10_000,
    n_restarts: int = 5,
    n_steps: int = 400,
    lr: float = 0.05,
) -> tuple[float, np.ndarray]:
    """Adam over softmax-parameterised α. Returns (KL, α*)."""
    K = len(component_means)
    # MC samples drawn ONCE for the inner optimisation (variance reduction);
    # each restart redraws a fresh batch.
    best_kl = np.inf
    best_alpha = np.ones(K) / K

    for restart in range(n_restarts):
        x = sample_gaussian(rng, target_mu, target_S, n_samples)
        log_target = gaussian_logpdf(x, target_mu, target_S)
        comp_logp_xy = np.stack(
            [gaussian_logpdf(x, component_means[i], component_covs[i]) for i in range(K)],
            axis=0,
        )  # (K, N)

        # init theta
        theta = rng.standard_normal(K) * 0.1
        m = np.zeros(K)
        v = np.zeros(K)
        b1, b2, eps = 0.9, 0.999, 1e-8

        prev_kl = np.inf
        for t in range(1, n_steps + 1):
            log_alpha = theta - _logsumexp(theta, axis=0)
            alpha = np.exp(log_alpha)
            comp = log_alpha[:, None] + comp_logp_xy  # (K, N)
            log_mix = _logsumexp(comp, axis=0)
            kl = float(np.mean(log_target - log_mix))

            # gradient of -log_mix wrt log_alpha[i]:
            # d/dlog_alpha[i] log_mix = softmax_i(log_alpha + comp_logp_xy)
            # → mean over N gives a posterior responsibility under the mixture.
            # dKL/dlog_alpha[i] = -E_target[softmax_i(...)] (with the - log_mix sign)
            resp = np.exp(comp - log_mix[None, :])  # (K, N) — softmax
            grad_log_alpha = -resp.mean(axis=1)  # (K,)
            # log_alpha = theta - logsumexp(theta), so
            # dlog_alpha[i]/dtheta[j] = δ_ij - softmax(theta)[j]
            # dKL/dtheta[j] = Σ_i grad_log_alpha[i] (δ_ij - softmax(theta)[j])
            #              = grad_log_alpha[j] - softmax(theta)[j] * Σ_i grad_log_alpha[i]
            sm_theta = np.exp(theta - _logsumexp(theta, axis=0))
            grad_theta = grad_log_alpha - sm_theta * grad_log_alpha.sum()

            m = b1 * m + (1 - b1) * grad_theta
            v = b2 * v + (1 - b2) * grad_theta**2
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            theta = theta - lr * mh / (np.sqrt(vh) + eps)

            if abs(prev_kl - kl) < 1e-7 and t > 50:
                break
            prev_kl = kl

        log_alpha = theta - _logsumexp(theta, axis=0)
        alpha = np.exp(log_alpha)
        if kl < best_kl:
            best_kl = kl
            best_alpha = alpha
    return best_kl, best_alpha


def mahalanobis_in_human(human: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu = human.mean(axis=0)
    S = np.cov(human, rowvar=False)
    S = S + 1e-6 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    diff = x - mu
    sol = np.linalg.solve(S, diff.T)
    return np.sqrt(np.sum(diff.T * sol, axis=0))


def fit_2d_pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean, components_2xD, singular_values_2,)."""
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    return mu.squeeze(0), Vt[:2], S[:2]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


BOT_DEFAULT_ORDER = [
    "BC_proxy_1",
    "BC_proxy_2",
    "Latent_proxy_1",
    "Latent_proxy_2",
    "Latent_proxy_3",
]
BC_NAMES = ["BC_proxy_1", "BC_proxy_2"]
LATENT_NAMES = ["Latent_proxy_1", "Latent_proxy_2", "Latent_proxy_3"]


def load_clouds(args: argparse.Namespace) -> tuple[Cloud, list[Cloud]]:
    human = np.load(args.human_npy).astype(np.float32)
    bots: list[Cloud] = []
    for name in BOT_DEFAULT_ORDER:
        path = os.path.join(args.bot_dir, f"{name}_per_seat.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        bots.append(Cloud(name=name, points=np.load(path).astype(np.float32)))
    return Cloud(name="Human", points=human), bots


def _ellipse_from_points(pts: np.ndarray, n_std: float = 1.0):
    """Return matplotlib Ellipse params (xy, width, height, angle_deg) for a
    2-D point cloud's empirical covariance, scaled to ``n_std`` standard
    deviations along principal axes.
    """
    mu = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    # eigen-decomp ascending; sort descending for major axis first
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width = 2.0 * n_std * np.sqrt(max(vals[0], 0.0))
    height = 2.0 * n_std * np.sqrt(max(vals[1], 0.0))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return tuple(mu), width, height, angle


def stage_2_3_figure(
    human: Cloud,
    bots: list[Cloud],
    out_png: str,
    out_pca_npz: str,
    out_points_dir: str,
    title: str,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from scipy.stats import gaussian_kde

    mu, comps, S = fit_2d_pca(human.points)
    np.savez(out_pca_npz, mean=mu, components=comps, singular_values=S)

    def project(x: np.ndarray) -> np.ndarray:
        return (x - mu) @ comps.T

    os.makedirs(out_points_dir, exist_ok=True)
    h_pts = project(human.points)
    np.save(os.path.join(out_points_dir, "Human_points.npy"), h_pts)
    bot_pts = []
    for b in bots:
        pts = project(b.points)
        bot_pts.append(pts)
        np.save(os.path.join(out_points_dir, f"{b.name}_points.npy"), pts)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # --- Human KDE backdrop (lines only — lighter visual weight) ---
    kde = gaussian_kde(h_pts.T)
    pad = 0.8
    xmin = min(h_pts[:, 0].min(), *(p[:, 0].min() for p in bot_pts)) - pad
    xmax = max(h_pts[:, 0].max(), *(p[:, 0].max() for p in bot_pts)) + pad
    ymin = min(h_pts[:, 1].min(), *(p[:, 1].min() for p in bot_pts)) - pad
    ymax = max(h_pts[:, 1].max(), *(p[:, 1].max() for p in bot_pts)) + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 220), np.linspace(ymin, ymax, 220))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(grid).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=8, cmap="Greys", alpha=0.35)
    # light human scatter (faint dots so the cloud has texture)
    ax.scatter(
        h_pts[:, 0],
        h_pts[:, 1],
        s=3,
        alpha=0.05,
        color="0.25",
        zorder=1,
    )

    bot_colors = {
        "BC_proxy_1": "#1f77b4",  # blue
        "BC_proxy_2": "#ff7f0e",  # orange
        "Latent_proxy_1": "#2ca02c",  # green
        "Latent_proxy_2": "#d62728",  # red
        "Latent_proxy_3": "#9467bd",  # purple
    }
    # scatter (very faint — ellipses are the primary visual)
    for b, pts in zip(bots, bot_pts):
        c = bot_colors.get(b.name, "gray")
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.06, color=c, zorder=2)

    # --- 1-σ ellipses ---
    # Human (1σ) — heavy dashed outline, no fill
    h_xy, h_w, h_h, h_ang = _ellipse_from_points(h_pts, n_std=1.0)
    ax.add_patch(
        Ellipse(
            xy=h_xy,
            width=h_w,
            height=h_h,
            angle=h_ang,
            facecolor="none",
            edgecolor="black",
            linewidth=3.0,
            linestyle=(0, (6, 3)),
            zorder=5,
            label="Human 1σ",
        )
    )

    # Bot ellipses: filled with brighter alpha + crisp coloured edge + thin
    # black halo for contrast against the grey backdrop.
    for b, pts in zip(bots, bot_pts):
        c = bot_colors.get(b.name, "gray")
        xy, w, h, ang = _ellipse_from_points(pts, n_std=1.0)
        # outer halo (slightly larger to give a dark rim)
        ax.add_patch(
            Ellipse(
                xy=xy,
                width=w * 1.02,
                height=h * 1.02,
                angle=ang,
                facecolor="none",
                edgecolor="black",
                linewidth=4.5,
                alpha=0.55,
                zorder=5.5,
            )
        )
        ax.add_patch(
            Ellipse(
                xy=xy,
                width=w,
                height=h,
                angle=ang,
                facecolor=c,
                alpha=0.32,
                edgecolor=c,
                linewidth=3.0,
                zorder=6,
            )
        )

    # centroids
    for b, pts in zip(bots, bot_pts):
        c = bot_colors.get(b.name, "gray")
        cx, cy = pts.mean(axis=0)
        ax.scatter(
            [cx],
            [cy],
            marker="X",
            s=240,
            color=c,
            edgecolors="black",
            linewidths=2.0,
            zorder=7,
            label=b.name,
        )
    hcx, hcy = h_pts.mean(axis=0)
    ax.scatter(
        [hcx],
        [hcy],
        marker="*",
        s=420,
        color="white",
        edgecolors="black",
        linewidths=2.5,
        zorder=8,
        label="Human centroid",
    )

    ax.set_title(title)
    ax.set_xlabel(f"PCA-1  (sing. val. {S[0]:.1f})")
    ax.set_ylabel(f"PCA-2  (sing. val. {S[1]:.1f})")
    # square aspect for ellipses to read truthfully
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25, linestyle=":")
    # legend in dedicated panel space outside the axes
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        borderpad=0.7,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "pca_singular_values": [float(s) for s in S],
        "human_centroid_2d": [float(hcx), float(hcy)],
        "bot_centroids_2d": {
            b.name: [float(p.mean(axis=0)[0]), float(p.mean(axis=0)[1])]
            for b, p in zip(bots, bot_pts)
        },
    }


def stage_4_kl_analysis(
    human: Cloud,
    bots: list[Cloud],
    out_dir: str,
    n_samples: int = 10_000,
    n_bootstrap: int = 1000,
    seed: int = 0,
    skip_bootstrap: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)

    print("[stage 4] fitting Ledoit-Wolf Gaussians…")
    mu_h, S_h = fit_ledoit_wolf(human.points)
    bot_fits = {}
    for b in bots:
        mu_b, S_b = fit_ledoit_wolf(b.points)
        bot_fits[b.name] = (mu_b, S_b)

    # Single-bot KLs
    print("[stage 4] single-bot KLs…")
    single = {}
    for b in bots:
        mu_b, S_b = bot_fits[b.name]
        single[b.name] = {
            "fwd_KL_bot_to_human": kl_gaussian(mu_b, S_b, mu_h, S_h),  # D(P_bot ‖ P_h)
            "rev_KL_human_to_bot": kl_gaussian(mu_h, S_h, mu_b, S_b),  # D(P_h ‖ P_bot)
        }

    # Ensemble mixture coverage
    print("[stage 4] BC ensemble mixture KL…")
    bc_means = [bot_fits[n][0] for n in BC_NAMES]
    bc_covs = [bot_fits[n][1] for n in BC_NAMES]
    kl_bc, alpha_bc = optimise_mixture_alpha(
        bc_means, bc_covs, mu_h, S_h, rng, n_samples=n_samples
    )
    print(f"  BC ensemble: KL={kl_bc:.4f}  α={alpha_bc.round(3)}")

    print("[stage 4] Latent ensemble mixture KL…")
    lat_means = [bot_fits[n][0] for n in LATENT_NAMES]
    lat_covs = [bot_fits[n][1] for n in LATENT_NAMES]
    kl_lat, alpha_lat = optimise_mixture_alpha(
        lat_means, lat_covs, mu_h, S_h, rng, n_samples=n_samples
    )
    print(f"  Latent ensemble: KL={kl_lat:.4f}  α={alpha_lat.round(3)}")

    # Combined ensemble (all 5)
    print("[stage 4] All-5 ensemble mixture KL…")
    all_names = BC_NAMES + LATENT_NAMES
    all_means = [bot_fits[n][0] for n in all_names]
    all_covs = [bot_fits[n][1] for n in all_names]
    kl_all, alpha_all = optimise_mixture_alpha(
        all_means, all_covs, mu_h, S_h, rng, n_samples=n_samples
    )
    print(f"  All ensemble: KL={kl_all:.4f}  α={alpha_all.round(3)}")

    # Sanity: Mahalanobis median per bot
    print("[stage 4] Mahalanobis sanity…")
    d = human.points.shape[1]
    from scipy.stats import chi2

    mahal_thresh = float(np.sqrt(chi2.ppf(0.99, df=d)))
    mahal_stats = {}
    for b in bots:
        m = mahalanobis_in_human(human.points, b.points)
        mahal_stats[b.name] = {
            "median": float(np.median(m)),
            "p90": float(np.percentile(m, 90)),
            "frac_outside_99pct_chi2": float(np.mean(m > mahal_thresh)),
        }
    mahal_stats["_threshold_99pct"] = mahal_thresh

    # Bootstrap CIs
    boot = {}
    if not skip_bootstrap and n_bootstrap > 0:
        print(f"[stage 4] bootstrap {n_bootstrap} reps…")
        N = human.points.shape[0]
        kl_single_boot = {b.name: {"fwd": [], "rev": []} for b in bots}
        kl_ens_boot = {"BC": [], "Latent": [], "All": []}
        alpha_ens_boot = {"BC": [], "Latent": [], "All": []}
        t0 = time.time()
        for r in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            mu_h_b, S_h_b = fit_ledoit_wolf(human.points[idx])
            for b in bots:
                mu_bot, S_bot = bot_fits[b.name]
                kl_single_boot[b.name]["fwd"].append(kl_gaussian(mu_bot, S_bot, mu_h_b, S_h_b))
                kl_single_boot[b.name]["rev"].append(kl_gaussian(mu_h_b, S_h_b, mu_bot, S_bot))
            kl_b, alpha_b = optimise_mixture_alpha(
                bc_means, bc_covs, mu_h_b, S_h_b, rng, n_samples=2000, n_restarts=2, n_steps=200
            )
            kl_ens_boot["BC"].append(kl_b)
            alpha_ens_boot["BC"].append(alpha_b.tolist())
            kl_l, alpha_l = optimise_mixture_alpha(
                lat_means, lat_covs, mu_h_b, S_h_b, rng, n_samples=2000, n_restarts=2, n_steps=200
            )
            kl_ens_boot["Latent"].append(kl_l)
            alpha_ens_boot["Latent"].append(alpha_l.tolist())
            kl_a, alpha_a = optimise_mixture_alpha(
                all_means, all_covs, mu_h_b, S_h_b, rng, n_samples=2000, n_restarts=2, n_steps=200
            )
            kl_ens_boot["All"].append(kl_a)
            alpha_ens_boot["All"].append(alpha_a.tolist())
            if (r + 1) % max(1, n_bootstrap // 10) == 0:
                print(f"  bootstrap {r + 1}/{n_bootstrap}  elapsed={time.time() - t0:.1f}s")

        # Summarise CIs
        def ci(arr):
            arr = np.asarray(arr)
            return {
                "mean": float(arr.mean()),
                "ci_2.5": float(np.percentile(arr, 2.5)),
                "ci_97.5": float(np.percentile(arr, 97.5)),
            }

        for b in bots:
            single[b.name]["fwd_CI"] = ci(kl_single_boot[b.name]["fwd"])
            single[b.name]["rev_CI"] = ci(kl_single_boot[b.name]["rev"])
        boot["ensemble_BC_CI"] = ci(kl_ens_boot["BC"])
        boot["ensemble_Latent_CI"] = ci(kl_ens_boot["Latent"])
        boot["ensemble_All_CI"] = ci(kl_ens_boot["All"])

        # alpha CIs (per-component)
        def alpha_ci(rows):
            arr = np.asarray(rows)
            return {
                "mean": arr.mean(axis=0).tolist(),
                "ci_2.5": np.percentile(arr, 2.5, axis=0).tolist(),
                "ci_97.5": np.percentile(arr, 97.5, axis=0).tolist(),
            }

        boot["alpha_BC_CI"] = alpha_ci(alpha_ens_boot["BC"])
        boot["alpha_Latent_CI"] = alpha_ci(alpha_ens_boot["Latent"])
        boot["alpha_All_CI"] = alpha_ci(alpha_ens_boot["All"])

    out = {
        "single_bot": single,
        "ensemble_BC": {"KL": float(kl_bc), "alpha": alpha_bc.tolist(), "components": BC_NAMES},
        "ensemble_Latent": {
            "KL": float(kl_lat),
            "alpha": alpha_lat.tolist(),
            "components": LATENT_NAMES,
        },
        "ensemble_All": {
            "KL": float(kl_all),
            "alpha": alpha_all.tolist(),
            "components": all_names,
        },
        "mahalanobis_sanity": mahal_stats,
        "bootstrap_CI": boot,
        "n_human": int(human.points.shape[0]),
        "n_bot": {b.name: int(b.points.shape[0]) for b in bots},
        "dim": int(human.points.shape[1]),
        "n_mc_samples": n_samples,
        "n_bootstrap": n_bootstrap,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kl_table.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--human_npy", required=True, help="(N, D) per-seat human encodings")
    p.add_argument(
        "--bot_dir",
        required=True,
        help="directory containing {bot}_per_seat.npy for each of the 5 bots",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="Policy clouds in human PCA basis (VAE space)")
    p.add_argument("--n_mc_samples", type=int, default=10_000)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_figure", action="store_true")
    p.add_argument("--skip_bootstrap", action="store_true")
    p.add_argument(
        "--figure_only",
        action="store_true",
        help="only redraw the figure; leave kl_table.json untouched",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    human, bots = load_clouds(args)
    print(f"loaded human: {human.points.shape}  bots: {[(b.name, b.points.shape) for b in bots]}")

    if not args.skip_figure:
        fig_info = stage_2_3_figure(
            human=human,
            bots=bots,
            out_png=os.path.join(args.out_dir, "coverage_figure.png"),
            out_pca_npz=os.path.join(args.out_dir, "human_pca.npz"),
            out_points_dir=os.path.join(args.out_dir, "points"),
            title=args.title,
        )
        with open(os.path.join(args.out_dir, "figure_info.json"), "w") as f:
            json.dump(fig_info, f, indent=2)
        print(f"  figure -> {os.path.join(args.out_dir, 'coverage_figure.png')}")

    if args.figure_only:
        print("[figure_only] skipping Stage 4 (kl_table.json untouched)")
        return

    res = stage_4_kl_analysis(
        human=human,
        bots=bots,
        out_dir=args.out_dir,
        n_samples=args.n_mc_samples,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        skip_bootstrap=args.skip_bootstrap,
    )
    print(
        json.dumps(
            {
                "single_bot": res["single_bot"],
                "ensemble_BC": res["ensemble_BC"],
                "ensemble_Latent": res["ensemble_Latent"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
