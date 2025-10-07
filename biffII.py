
import numpy as np
import matplotlib.pyplot as plt
from bif import *

def F_vec(y, Iext, par):
    V, n, m, h, p = y
    Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax = par
    n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V, VT, smax)
    Iion = ionic_currents(V, m, h, n, p, gNa, gK, gM, gL, ENa, EK, EL)
    f0 = (Iext - Iion) / Cm
    f1 = (n_inf - n) / max(tn, 1e-9)
    f2 = (m_inf - m) / max(tm, 1e-9)
    f3 = (h_inf - h) / max(th, 1e-9)
    f4 = (p_inf - p) / max(tp, 1e-9)
    return np.array([f0, f1, f2, f3, f4], dtype=np.float64)

def jacobian_fd(y, Iext, par, abs_min=1e-8, abs_max=1e-2):
    f0 = F_vec(y, Iext, par)
    J = np.empty((5, 5), dtype=np.float64)
    for i in range(5):
        # scale-aware step
        hi = np.sqrt(np.finfo(float).eps) * (abs(y[i]) + 1.0)
        hi = float(min(max(hi, abs_min), abs_max))
        ei = np.zeros(5); ei[i] = 1.0
        fp = F_vec(y + hi*ei, Iext, par)
        fm = F_vec(y - hi*ei, Iext, par)
        J[:, i] = (fp - fm) / (2.0*hi)
    return J

def newton_equilibrium(y0, Iext, par, maxit=1000, tol=1e-13):
    y = y0.astype(np.float64).copy()
    for _ in range(maxit):
        f = F_vec(y, Iext, par)
        if np.linalg.norm(f, ord=np.inf) < tol:
            return y, True
        J = jacobian_fd(y, Iext, par)
        # Levenberg regularization fallback if needed
        try:
            delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J + 1e-6*np.eye(5), -f, rcond=None)[0]
        # backtracking line search
        alpha = 1.0
        f_norm = np.linalg.norm(f)
        improved = False
        while alpha >= 1e-4:
            y_try = y + alpha*delta
            f_try = F_vec(y_try, Iext, par)
            if np.linalg.norm(f_try) < 0.8*f_norm:
                y = y_try
                improved = True
                break
            alpha *= 0.5
        if not improved:
            # tiny nudged step
            y = y + 1e-2*delta
    return y, False

def continuation_equilibria(Ivals, par, V_seed=None):
    eq_states = np.empty((len(Ivals), 5), dtype=np.float64)
    success = np.zeros(len(Ivals), dtype=bool)

    # initial guess from steady-state at a reasonable V
    if V_seed is None:
        V_seed = resting_potential(Ivals[0], par)
        if not np.isfinite(V_seed):
            V_seed = -65.0
    n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V_seed, par[8], par[9])
    y0 = np.array([V_seed, n_inf, m_inf, h_inf, p_inf], dtype=np.float64)

    for i, I in enumerate(Ivals):
        y_star, ok = newton_equilibrium(y0, I, par)
        eq_states[i] = y_star
        success[i] = ok
        # use this equilibrium as the predictor for the next I
        y0 = y_star.copy()
    return eq_states, success

def eigs_at_equilibrium(y_eq, Iext, par):
    J = jacobian_fd(y_eq, Iext, par)
    w = np.linalg.eigvals(J)
    return w, J

def identify_leading_pair(eigs):
    # indices of the two eigenvalues with smallest |Re|
    idx = np.argsort(np.abs(np.real(eigs)))[:2]
    return idx

# ========================= Runner & Plots =========================

def run_equilibrium_sweep_and_plots(Imin=-190, Imax=12.0, nI=1201, par=None,
                                    out_prefix="eq_sweep", show=False):
    """
    Continuation across Iext, compute equilibria, eigenvalues, and make:
      1) V* vs Iext
      2) Re(lambda_lead) vs Iext
      3) Eigenvalues on complex plane for selected Iext
    """
    if par is None:
        par = rsa_par_tuple()

    Ivals = np.linspace(Imin, Imax, nI)
    eq_states, ok = continuation_equilibria(Ivals, par)
    Vstars = eq_states[:, 0]

    # Compute spectra
    eigs_all = np.empty((nI, 5), dtype=np.complex128)
    for i in range(nI):
        w, _ = eigs_at_equilibrium(eq_states[i], Ivals[i], par)
        eigs_all[i] = w

    # Leading pair real parts
    lead_re = np.full(nI, np.nan, dtype=float)
    lead_im = np.full(nI, np.nan, dtype=float)
    lead_idx_save = np.zeros((nI, 2), dtype=int)

    # Track by nearest-neighbor matching to avoid pair swapping noise
    # start: choose by smallest |Re|
    idx0 = identify_leading_pair(eigs_all[0])
    lead_idx_save[0] = idx0
    lead_re[0] = np.mean(np.real(eigs_all[0, idx0]))
    lead_im[0] = np.mean(np.abs(np.imag(eigs_all[0, idx0])))

    for i in range(1, nI):
        prev = eigs_all[i-1, lead_idx_save[i-1]]
        # pair current eigenvalues to previous pair by minimal distance
        distances = np.abs(eigs_all[i][:, None] - prev[None, :]).sum(axis=1)
        idx_sorted = np.argsort(distances)
        lead_idx_save[i] = idx_sorted[:2]
        curpair = eigs_all[i, lead_idx_save[i]]
        lead_re[i] = np.mean(np.real(curpair))
        lead_im[i] = np.mean(np.abs(np.imag(curpair)))

    # ----------------- Plot 1: V* vs Iext -----------------
    fig1, ax1 = plt.subplots(1, 1, figsize=(8.5, 4.0), constrained_layout=True)
    ax1.plot(Ivals, Vstars, lw=1.6)
    ax1.set_xlabel("Iext (µA/cm²)")
    ax1.set_ylabel("Equilibrium V* (mV)")
    ax1.set_title("V* vs Iext")
    if out_prefix:
        fig1.savefig(f"{out_prefix}_Vstar_vs_I.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)

    # ----------------- Plot 2: Re(leading λ) vs Iext -----------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(8.5, 4.0), constrained_layout=True)
    ax2.plot(Ivals, lead_re, lw=1.6)
    ax2.axhline(0.0, linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Iext (µA/cm²)")
    ax2.set_ylabel("Re(leading eigenpair) (1/ms)")
    ax2.set_title("")
    if out_prefix:
        fig2.savefig(f"{out_prefix}_Relead_vs_I.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig2)

    # ----------------- Plot 3: Eigenvalues in complex plane -----------------
    # choose a handful of Iext values across the range
    sel_idx = np.linspace(0, nI-1, 6, dtype=int)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6.2, 6.0), constrained_layout=True)
    for j, i in enumerate(sel_idx):
        w = eigs_all[i]
        ax3.scatter(np.real(w), np.imag(w), s=16, label=f"I={Ivals[i]:.2f}")
        # highlight the tracked leading pair
        lp = w[lead_idx_save[i]]
        ax3.plot(np.real(lp), np.imag(lp), marker='o', linestyle='-')
    ax3.axvline(0.0, linestyle="--", linewidth=1.0)
    ax3.set_xlabel("Re")
    ax3.set_ylabel("Im")
    ax3.set_title("Eigenvalues of Jacobian at equilibria")
    ax3.legend(loc="best", fontsize=8)
    if out_prefix:
        fig3.savefig(f"{out_prefix}_eigs_complex.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig3)

    # Simple textual hint about Hopf
    # Detect sign change in lead_re to suggest crossing interval
    sign = np.sign(lead_re)
    crosses = np.where((sign[:-1] * sign[1:] < 0) & (lead_im[:-1] > 1e-6) & (lead_im[1:] > 1e-6))[0]
    if crosses.size > 0:
        i0 = crosses[0]
        I_left, I_right = Ivals[i0], Ivals[i0+1]
        slope = (lead_re[i0+1] - lead_re[i0]) / (I_right - I_left)
        print(f"[Hopf?] Sign change near Iext in [{I_left:.4f}, {I_right:.4f}], "
              f"approx slope d(Re λ)/dI ≈ {slope:.4g} 1/(ms·µA/cm²)")
    else:
        print("[Hopf?] No clean sign change of the leading pair’s real part on this grid.")

    return {
        "Ivals": Ivals,
        "eq_states": eq_states,
        "ok": ok,
        "eigs": eigs_all,
        "lead_re": lead_re,
        "lead_im": lead_im
    }

# ========================= Example CLI hook =========================

def run_eq_cli(Imin=0.0, Imax=12.0, nI=121, Vseed=None, show=False):
    """
    Convenience entry point you can call from __main__ or your CLI.
    """
    par = rsa_par_tuple()
    results = run_equilibrium_sweep_and_plots(Imin, Imax, nI, par, out_prefix="eq_sweep", show=show)
    return results

run_eq_cli(show=True)
