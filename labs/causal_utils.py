"""causal_utils.py - reusable helpers for the Experimentation & Causal Inference labs.

Import with:  from causal_utils import *
Every function is small on purpose so participants can read it in one screen.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chisquare

# ----------------------------------------------------------------------------
# Assignment and experiment integrity
# ----------------------------------------------------------------------------
def assign_variant(unit_id: str, salt: str, split=(0.5, 0.5), arms=("control", "treatment")) -> str:
    """Deterministic hash bucketing: same id + salt -> same arm, no stored state."""
    assert abs(sum(split) - 1) < 1e-9, "split must sum to 1"
    u = int(hashlib.sha256(f"{salt}:{unit_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cum = 0.0
    for arm, frac in zip(arms, split):
        cum += frac
        if u < cum:
            return arm
    return arms[-1]


def srm_check(counts: dict, expected_split: dict, alpha: float = 0.001) -> dict:
    """Sample-ratio-mismatch chi-square test. A failure means: do not analyse, debug the pipeline."""
    arms = list(counts)
    obs = np.array([counts[a] for a in arms], dtype=float)
    exp = np.array([expected_split[a] for a in arms]) * obs.sum()
    stat, p = chisquare(obs, exp)
    return {"chi2": round(float(stat), 3), "p_value": float(p), "srm_detected": bool(p < alpha),
            "verdict": "INVALID - investigate assignment/logging" if p < alpha else "ratios OK"}


def smd(df: pd.DataFrame, covariate: str, treat_col: str, treated_value=1, weights: str | None = None) -> float:
    """Standardised mean difference (treated minus control) / pooled SD. |SMD| < 0.1 is the usual balance rule."""
    t = df[df[treat_col] == treated_value]
    c = df[df[treat_col] != treated_value]
    if weights is None:
        mt, mc = t[covariate].mean(), c[covariate].mean()
        vt, vc = t[covariate].var(ddof=1), c[covariate].var(ddof=1)
    else:
        mt = np.average(t[covariate], weights=t[weights]); mc = np.average(c[covariate], weights=c[weights])
        vt = np.average((t[covariate] - mt) ** 2, weights=t[weights])
        vc = np.average((c[covariate] - mc) ** 2, weights=c[weights])
    pooled = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / pooled) if pooled > 0 else 0.0


def balance_table(df: pd.DataFrame, covariates: list[str], treat_col: str, treated_value=1,
                  weights: str | None = None) -> pd.DataFrame:
    rows = []
    for cov in covariates:
        s = smd(df, cov, treat_col, treated_value, weights)
        rows.append({"covariate": cov, "smd": round(s, 4), "balanced": abs(s) < 0.1})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# Power and sizing
# ----------------------------------------------------------------------------
def sample_size_proportions(p1: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm n to detect an absolute lift `mde` over baseline p1 (two-sided)."""
    p2 = p1 + mde
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num / (p2 - p1) ** 2))


def sample_size_means(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(np.ceil(2 * sigma ** 2 * (z_a + z_b) ** 2 / delta ** 2))


def mde_proportions(p1: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute lift detectable at fixed n per arm (normal approximation around p1)."""
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * p1 * (1 - p1) / n_per_arm))


def run_length_days(n_per_arm: int, n_arms: int, daily_eligible: int, trigger_rate: float, min_days: int = 14) -> int:
    total = n_per_arm * n_arms
    days = int(np.ceil(total / (daily_eligible * trigger_rate)))
    return max(days, min_days)

# ----------------------------------------------------------------------------
# Estimation for experiments
# ----------------------------------------------------------------------------
def diff_in_means(y: np.ndarray, t: np.ndarray) -> dict:
    """Neyman estimator with the conservative design-based SE: S1^2/n1 + S0^2/n0."""
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - 1.96 * se),
            "ci_high": float(est + 1.96 * se), "p_value": float(2 * (1 - norm.cdf(abs(est / se))))}


def randomization_test(y: np.ndarray, t: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """Fisher randomization inference for the sharp null H0: Y_i(1) = Y_i(0) for all i."""
    rng = np.random.default_rng(seed)
    obs = y[t == 1].mean() - y[t == 0].mean()
    perms = np.empty(n_perm)
    for k in range(n_perm):
        tp = rng.permutation(t)
        perms[k] = y[tp == 1].mean() - y[tp == 0].mean()
    p = float((np.abs(perms) >= abs(obs)).mean())
    return {"observed": float(obs), "p_value": p, "null_draws": perms}


def cuped_adjust(y: pd.Series, x_pre: pd.Series) -> pd.Series:
    """CUPED: subtract theta * (x_pre - mean); theta = cov(y, x)/var(x). x_pre must be pre-treatment."""
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / x_pre.var(ddof=1)
    return y - theta * (x_pre - x_pre.mean())


def ratio_delta(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Ratio-of-means estimate and delta-method SE (for metrics like uploads per session)."""
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cov = np.cov(num, den, ddof=1)[0, 1]
    var = (vx - 2 * (mx / my) * cov + (mx / my) ** 2 * vy) / (my ** 2 * n)
    return float(mx / my), float(np.sqrt(var))


def msprt_reject(diff: float, var_diff: float, tau2: float = 0.01 ** 2 * 4, alpha: float = 0.05) -> bool:
    """Mixture sequential probability ratio test for a difference in means with (estimated) variance var_diff.

    Always-valid: reject the null when the mixture likelihood ratio >= 1/alpha. Can be checked at every
    look without inflating Type I error. tau2 is the mixing variance (prior scale of the effect)."""
    lam = np.sqrt(var_diff / (var_diff + tau2)) * np.exp(tau2 * diff ** 2 / (2 * var_diff * (var_diff + tau2)))
    return bool(lam >= 1 / alpha)

# ----------------------------------------------------------------------------
# Observational estimators
# ----------------------------------------------------------------------------
def ipw_ate(y: np.ndarray, t: np.ndarray, e: np.ndarray, stabilised: bool = True, clip=(0.01, 0.99)) -> dict:
    e = np.clip(e, *clip)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    if stabilised:
        w = np.where(t == 1, t.mean() / e, (1 - t.mean()) / (1 - e))
    mu1 = np.sum(w * t * y) / np.sum(w * t)
    mu0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    return {"ate": float(mu1 - mu0), "weights": w}


def aipw_ate(y: np.ndarray, t: np.ndarray, e: np.ndarray, m1: np.ndarray, m0: np.ndarray) -> dict:
    """Augmented IPW (doubly robust). e: propensity; m1, m0: outcome-model predictions under T=1 / T=0.
    Returns the estimate and an influence-function-based SE."""
    e = np.clip(e, 0.01, 0.99)
    psi = (m1 - m0) + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    est = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(len(psi))
    return {"ate": float(est), "se": float(se), "ci_low": float(est - 1.96 * se), "ci_high": float(est + 1.96 * se)}


def did_2x2(df: pd.DataFrame, y: str, group: str, post: str) -> float:
    m = df.groupby([group, post])[y].mean()
    return float((m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0]))


def e_value(rr: float) -> float:
    """VanderWeele & Ding E-value for a risk ratio (use 1/rr if rr < 1)."""
    rr = max(rr, 1 / rr)
    return float(rr + np.sqrt(rr * (rr - 1)))
