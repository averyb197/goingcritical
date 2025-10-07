from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numba import njit

# ========================= Logging =========================

def setup_logging(level: str = "INFO", logfile: str | None = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w"))
    logging.basicConfig(
        level=lvl,
        handlers=handlers,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

# ========================= Parameters =========================

@dataclass(frozen=True)
class RSAParams:
    Cm: float   = 1.0
    gNa: float  = 56.0
    gK: float   = 6.0
    gM: float   = 0.075
    gL: float   = 0.0205
    ENa: float  = 56.0
    EK: float   = -90.0
    EL: float   = -70.3
    VT: float   = -56.2
    smax: float = 608.0

def rsa_par_tuple() -> tuple:
    p = RSAParams()
    return (p.Cm, p.gNa, p.gK, p.gM, p.gL, p.ENa, p.EK, p.EL, p.VT, p.smax)

# ========================= Numba-safe helpers =========================

@njit(cache=True, fastmath=False)
def _safe_exp(x: float) -> float:
    # avoid overflow/underflow storms in exp
    if x > 60.0:
        x = 60.0
    elif x < -60.0:
        x = -60.0
    return np.exp(x)

@njit(cache=True, fastmath=False)
def _phi(x: float, k: float) -> float:
    # x / (1 - e^{-x/k}) with benign limit k at x→0
    z = -x / k
    den = 1.0 - _safe_exp(z)
    if np.abs(den) < 1e-12:
        return k
    return x / den

@njit(cache=True, fastmath=False)
def _psi(x: float, k: float) -> float:
    # x / (e^{x/k} - 1) with benign limit k at x→0
    z = x / k
    den = _safe_exp(z) - 1.0
    if np.abs(den) < 1e-12:
        return k
    return x / den

@njit(cache=True, fastmath=False)
def rates_nmh(V: float, VT: float):
    x = V - VT
    a_n = 0.032 * _phi(x - 15.0, 5.0)
    b_n = 0.5   * _safe_exp(-(x - 10.0) / 40.0)
    a_m = 0.32  * _phi(x - 13.0, 4.0)
    b_m = 0.28  * _psi(x - 40.0, 5.0)
    a_h = 0.128 * _safe_exp(-(x - 17.0) / 18.0)
    b_h = 4.0   / (1.0 + _safe_exp(-(x - 40.0) / 5.0))
    return a_n, b_n, a_m, b_m, a_h, b_h

@njit(cache=True, fastmath=False)
def steady_state_and_tau(alpha: float, beta: float):
    denom = alpha + beta + 1e-12
    return alpha / denom, 1.0 / denom

@njit(cache=True, fastmath=False)
def im_kinetics(V: float, smax: float):
    p_inf = 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))
    tau_p = smax / (3.3 * _safe_exp((V + 35.0) / 20.0) + _safe_exp(-(V + 35.0) / 20.0))
    return p_inf, tau_p

@njit(cache=True, fastmath=False)
def gating_inf_tau(V: float, VT: float, smax: float):
    a_n, b_n, a_m, b_m, a_h, b_h = rates_nmh(V, VT)
    n_inf, tn = steady_state_and_tau(a_n, b_n)
    m_inf, tm = steady_state_and_tau(a_m, b_m)
    h_inf, th = steady_state_and_tau(a_h, b_h)
    p_inf, tp = im_kinetics(V, smax)
    return n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp

@njit(cache=True, fastmath=False)
def ionic_currents(V, m, h, n, p, gNa, gK, gM, gL, ENa, EK, EL):
    INa = gNa * (m**3) * h * (V - ENa)
    IK  = gK  * (n**4) * (V - EK)
    IM  = gM  * p      * (V - EK)
    IL  = gL  * (V - EL)
    return INa + IK + IM + IL  # outward positive

# ========================= ODE RHS and RK4 step =========================

@njit(cache=True, fastmath=False)
def rhs(V, n, m, h, p, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax):
    n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V, VT, smax)
    Iion = ionic_currents(V, m, h, n, p, gNa, gK, gM, gL, ENa, EK, EL)
    # Standard HH sign: Iext>0 is inward depolarizing
    dVdt = (Iext - Iion) / Cm
    dndt = (n_inf - n) / max(tn, 1e-6)
    dmdt = (m_inf - m) / max(tm, 1e-6)
    dhdt = (h_inf - h) / max(th, 1e-6)
    dpdt = (p_inf - p) / max(tp, 1e-6)
    # Also return min gating tau for stability hints
    min_tau = min(tn, tm, th, tp)
    return dVdt, dndt, dmdt, dhdt, dpdt, min_tau

@njit(cache=True, fastmath=False)
def rk4_step(V, n, m, h, p, dt, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax):
    k1 = rhs(V, n, m, h, p, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax)

    V2 = V + 0.5*dt*k1[0]; n2 = n + 0.5*dt*k1[1]; m2 = m + 0.5*dt*k1[2]; h2 = h + 0.5*dt*k1[3]; p2 = p + 0.5*dt*k1[4]
    k2 = rhs(V2, n2, m2, h2, p2, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax)

    V3 = V + 0.5*dt*k2[0]; n3 = n + 0.5*dt*k2[1]; m3 = m + 0.5*dt*k2[2]; h3 = h + 0.5*dt*k2[3]; p3 = p + 0.5*dt*k2[4]
    k3 = rhs(V3, n3, m3, h3, p3, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax)

    V4 = V + dt*k3[0]; n4 = n + dt*k3[1]; m4 = m + dt*k3[2]; h4 = h + dt*k3[3]; p4 = p + dt*k3[4]
    k4 = rhs(V4, n4, m4, h4, p4, Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax)

    Vn = V + (dt/6.0)*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    nn = n + (dt/6.0)*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    mn = m + (dt/6.0)*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
    hn = h + (dt/6.0)*(k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])
    pn = p + (dt/6.0)*(k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4])

    # clamp gates to [0,1]
    if nn < 0.0: nn = 0.0
    elif nn > 1.0: nn = 1.0
    if mn < 0.0: mn = 0.0
    elif mn > 1.0: mn = 1.0
    if hn < 0.0: hn = 0.0
    elif hn > 1.0: hn = 1.0
    if pn < 0.0: pn = 0.0
    elif pn > 1.0: pn = 1.0

    # surface min_tau using k2 (midpoint) as proxy
    min_tau_mid = min(k2[5], 1e9)
    return Vn, nn, mn, hn, pn, min_tau_mid

# ========================= Simulation & frequency =========================

def simulate_stable(Iext: float, T: float, dt: float, V0: float, par: tuple, burn: float,
                    safety: float = 0.2, min_dt: float = 1e-5, max_dt: float = 1.0,
                    log_prefix: str = "Sim"):

    Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax = par
    steps = int(T / dt + 0.5)
    # init gates at steady-state for V0
    n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V0, VT, smax)
    n = float(n_inf); m = float(m_inf); h = float(h_inf); p = float(p_inf)
    V = float(V0)

    V_tr = np.empty(steps, dtype=np.float64)
    n_tr = np.empty(steps, dtype=np.float64)
    m_tr = np.empty(steps, dtype=np.float64)

    # diagnostic accumulators
    global_min_tau = 1e9
    global_max_absV = 0.0
    total_substeps = 0
    nan_hit = False

    for k in range(steps):
        V_tr[k] = V; n_tr[k] = n; m_tr[k] = m

        # compute a recommended substep from current V
        _, tn0, _, tm0, _, th0, _, tp0 = gating_inf_tau(V, VT, smax)
        min_tau = max(min(tn0, tm0, th0, tp0), 1e-6)
        global_min_tau = min(global_min_tau, min_tau)
        h_sub = min(dt, safety * min_tau)
        if h_sub < min_dt:
            h_sub = min_dt
        n_sub = int(np.ceil(dt / h_sub))
        if n_sub < 1: n_sub = 1
        h_sub = dt / n_sub  # equal partition

        for _ in range(n_sub):
            V, n, m, h, p, mt = rk4_step(V, n, m, h, p, h_sub,
                                         Iext, Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax)
            total_substeps += 1
            global_max_absV = max(global_max_absV, abs(V))
            # NaN/Inf guard
            if not np.isfinite(V) or not np.isfinite(n) or not np.isfinite(m) or not np.isfinite(h) or not np.isfinite(p):
                logging.error(f"{log_prefix}: Numerical blow-up at step {k}, substep; V={V}, n={n}, m={m}, h={h}, p={p}")
                nan_hit = True
                # truncate traces to what we have
                V_tr = V_tr[:k+1]; n_tr = n_tr[:k+1]; m_tr = m_tr[:k+1]
                steps = k + 1
                break
        if nan_hit:
            break

    burn_idx = int(burn / dt + 0.5)
    burn_idx = 0 if burn_idx < 0 else burn_idx
    burn_idx = steps if burn_idx > steps else burn_idx

    logging.info(f"{log_prefix}: steps={steps}, dt={dt:.6f} ms, Iext={Iext:.4f} µA/cm², "
                 f"min_tau≈{global_min_tau:.6f} ms, max|V|≈{global_max_absV:.3f} mV, "
                 f"substeps≈{total_substeps}, truncated={nan_hit}")

    return V_tr[burn_idx:], n_tr[burn_idx:], m_tr[burn_idx:], dt, nan_hit

@njit(cache=True)
def estimate_frequency_numba(V: np.ndarray, dt: float, thr: float, refractory_ms: float) -> float:
    n = V.size
    if n < 2:
        return 0.0
    ref_steps = int(refractory_ms / dt + 0.5)
    last_idx = -10**9
    s_times = np.empty(n, dtype=np.float64)
    s_count = 0

    for k in range(n - 1):
        V0 = V[k]
        V1 = V[k + 1]
        if V0 < thr and V1 >= thr:
            if k - last_idx >= ref_steps:
                dV = V1 - V0
                frac = 0.0 if np.abs(dV) < 1e-12 else (thr - V0) / dV
                t_cross = (k + frac) * dt
                s_times[s_count] = t_cross
                s_count += 1
                last_idx = k

    if s_count < 2:
        return 0.0
    acc = 0.0
    for i in range(1, s_count):
        acc += (s_times[i] - s_times[i - 1])
    mean_isi = acc / (s_count - 1)
    if mean_isi <= 0.0:
        return 0.0
    return 1000.0 / mean_isi  # Hz

def sweep_freq_vs_I(Imin, Imax, nI, T, dt, V0, burn, par):
    Ivals = np.linspace(Imin, Imax, nI, dtype=np.float64)
    freqs = np.zeros(nI, dtype=np.float64)
    for i, I in enumerate(Ivals):
        V, n, m, dt_, blew = simulate_stable(I, T, dt, V0, par, burn, log_prefix=f"Sweep[{i+1}/{nI}]")
        if not blew and V.size > 1:
            freqs[i] = estimate_frequency_numba(V, dt_, thr=-20.0, refractory_ms=2.0)
        else:
            freqs[i] = np.nan
    return Ivals, freqs

# ========================= Resting potential (steady-state V-nullcline) =========================

def f_total_current(V: float, Iext: float, par: tuple) -> float:
    Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax = par
    n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V, VT, smax)
    Iion = ionic_currents(V, m_inf, h_inf, n_inf, p_inf, gNa, gK, gM, gL, ENa, EK, EL)
    return Iion - Iext  # root when Iion == Iext

def resting_potential(Iext, par, Vmin=-100.0, Vmax=50.0, ngrid=600, tol=1e-6, maxit=80):
    Vgrid = np.linspace(Vmin, Vmax, ngrid)
    F = np.empty_like(Vgrid)
    for i, V in enumerate(Vgrid):
        F[i] = f_total_current(V, Iext, par)
    for i in range(len(Vgrid) - 1):
        f0 = F[i]; f1 = F[i+1]
        if f0 == 0.0:
            return Vgrid[i]
        if f1 == 0.0:
            return Vgrid[i+1]
        if f0 * f1 < 0.0:
            a = Vgrid[i]; b = Vgrid[i+1]
            fa = f0; fb = f1
            for _ in range(maxit):
                c = 0.5*(a+b)
                fc = f_total_current(c, Iext, par)
                if np.abs(fc) < tol or 0.5*(b-a) < tol:
                    return c
                if fa * fc < 0.0:
                    b = c; fb = fc
                else:
                    a = c; fa = fc
            return 0.5*(a+b)
    return np.nan


def nullclines_Vn(Iext, par, V_grid):
    Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax = par
    n_inf_vals = np.empty_like(V_grid)
    n_Vnull = np.full_like(V_grid, np.nan)
    for i, V in enumerate(V_grid):
        n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V, VT, smax)
        n_inf_vals[i] = n_inf
        denom = gK * (V - EK)
        rhs = -(gNa * (m_inf**3) * h_inf * (V - ENa) + gM * p_inf * (V - EK) + gL * (V - EL) - Iext)  # using Iion - Iext = 0
        if np.abs(denom) > 1e-12:
            n4 = rhs / denom
            if n4 >= 0.0:
                n_candidate = n4**0.25
                if 0.0 <= n_candidate <= 1.0:
                    n_Vnull[i] = n_candidate
    return n_inf_vals, n_Vnull

def nullclines_Vm(Iext, par, V_grid):
    Cm, gNa, gK, gM, gL, ENa, EK, EL, VT, smax = par
    m_inf_vals = np.empty_like(V_grid)
    m_Vnull = np.full_like(V_grid, np.nan)
    for i, V in enumerate(V_grid):
        n_inf, tn, m_inf, tm, h_inf, th, p_inf, tp = gating_inf_tau(V, VT, smax)
        m_inf_vals[i] = m_inf
        # Iion - Iext = 0 ⇒ gNa m^3 h (V-ENa) = Iext - [gK n^4 (V-EK) + gM p (V-EK) + gL (V-EL)]
        rest = gK*(n_inf**4)*(V - EK) + gM*p_inf*(V - EK) + gL*(V - EL)
        denom = gNa * h_inf * (V - ENa)
        if np.abs(denom) > 1e-12:
            m3 = (Iext - rest) / denom
            if m3 >= 0.0:
                m_candidate = m3**(1.0/3.0)
                if 0.0 <= m_candidate <= 1.0:
                    m_Vnull[i] = m_candidate
    return m_inf_vals, m_Vnull

# ========================= Plots =========================

def make_phaseplanes_and_trace(I0, T, dt, V0, burn, par, out_prefix=None, show=False):
    V, n, m, dt_, blew = simulate_stable(I0, T, dt, V0, par, burn, log_prefix="Main")
    t = np.arange(V.size, dtype=np.float64) * dt_

    # V vs time
    fig_t, ax_t = plt.subplots(1, 1, figsize=(9.5, 3.6), constrained_layout=True)
    ax_t.plot(t, V, lw=1.2)
    ax_t.set_xlabel("Time (ms)"); ax_t.set_ylabel("V (mV)")
    ax_t.set_title("Membrane potential trace")
    if out_prefix:
        fig_t.savefig(out_prefix + "_V_time.png", dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig_t)

    # Phase planes
    stride = max(1, int(0.2 / dt_))
    Vd = V[::stride]; nd = n[::stride]; md = m[::stride]

    Vg = np.linspace(-100.0, 50.0, 600)
    n_inf_curve, n_vnull = nullclines_Vn(I0, par, Vg)
    m_inf_curve, m_vnull = nullclines_Vm(I0, par, Vg)

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.plot(Vd, nd, lw=0.9, label='trajectory')
    ax1.plot(Vg, n_inf_curve, '--', lw=1.1, label='n = n_inf(V)')
    ax1.plot(Vg, n_vnull, '-', lw=1.1, label='V-nullcline')
    ax1.set_xlabel("V (mV)"); ax1.set_ylabel("n"); ax1.set_title("V–n phase plane")
    ax1.set_xlim(Vg[0], Vg[-1]); ax1.set_ylim(-0.05, 1.05); ax1.legend(loc='best')

    ax2.plot(Vd, md, lw=0.9, label='trajectory')
    ax2.plot(Vg, m_inf_curve, '--', lw=1.1, label='m = m_inf(V)')
    ax2.plot(Vg, m_vnull, '-', lw=1.1, label='V-nullcline')
    ax2.set_xlabel("V (mV)"); ax2.set_ylabel("m"); ax2.set_title("V–m phase plane")
    ax2.set_xlim(Vg[0], Vg[-1]); ax2.set_ylim(-0.05, 1.05); ax2.legend(loc='best')

    ax3.plot(Vd, nd, md, lw=0.7)
    ax3.set_xlabel("V (mV)"); ax3.set_ylabel("n"); ax3.set_zlabel("m"); ax3.set_title("V–n–m trajectory")

    if out_prefix:
        fig.savefig(out_prefix + "_phaseplanes.png", dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    # Frequency and resting potentials
    freq = estimate_frequency_numba(V, dt_, thr=-20.0, refractory_ms=2.0)
    Vrest_ss = resting_potential(I0, par)
    # robust time-average tail
    tail_ms = min(200.0, max(0.0, T - burn))
    if tail_ms > 0.0 and V.size > 0:
        tail_idx = max(0, V.size - int(tail_ms / dt_ + 0.5))
        tail = V[tail_idx:]
        Vrest_timeavg = float(np.nanmean(tail)) if tail.size > 0 else np.nan
    else:
        Vrest_timeavg = float(np.nanmean(V)) if V.size > 0 else np.nan

    print(f"[Info] Iext={I0:.4f} µA/cm² | freq≈{freq:.3f} Hz | "
          f"V_rest (steady-state root)≈{Vrest_ss:.3f} mV | "
          f"V_rest (time-avg tail)≈{Vrest_timeavg:.3f} mV | blew_up={blew}")

def make_freq_plot(Ivals, freqs, out=None, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.2), constrained_layout=True)
    ax.plot(Ivals, freqs, lw=1.6)
    ax.set_xlabel("Iext (µA/cm²)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Frequency vs Iext")
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# ========================= CLI =========================

def main():
    ap = argparse.ArgumentParser(description="Single-neuron RSA-HH (Numba, RK4+substeps) with nullclines and logging")
    ap.add_argument("--Imin", type=float, default=-5.0)
    ap.add_argument("--Imax", type=float, default=100.0)
    ap.add_argument("--nI", type=int, default=100)
    ap.add_argument("--I0", type=float, default=6.0)
    ap.add_argument("--T", type=float, default=3000.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--burn", type=float, default=800.0)
    ap.add_argument("--V0", type=float, default=-65.0)
    ap.add_argument("--phase_only", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--loglevel", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument("--logfile", type=str, default=None)
    args = ap.parse_args()

    setup_logging(args.loglevel, args.logfile)

    par = rsa_par_tuple()
    logging.info(f"Params: Cm={par[0]}, gNa={par[1]}, gK={par[2]}, gM={par[3]}, gL={par[4]}, "
                 f"ENa={par[5]}, EK={par[6]}, EL={par[7]}, VT={par[8]}, smax={par[9]}")
    logging.info(f"Run: I0={args.I0} µA/cm², T={args.T} ms, dt={args.dt} ms, burn={args.burn} ms, V0={args.V0} mV")

    if not args.phase_only:
        Ivals, freqs = sweep_freq_vs_I(args.Imin, args.Imax, args.nI, args.T, args.dt, args.V0, args.burn, par)
        make_freq_plot(Ivals, freqs, out=None if args.show else "freq_vs_I.png", show=args.show)

    make_phaseplanes_and_trace(args.I0, args.T, args.dt, args.V0, args.burn, par,
                               out_prefix=None if args.show else "rsa_single",
                               show=args.show)

if __name__ == "__main__":
    main()
