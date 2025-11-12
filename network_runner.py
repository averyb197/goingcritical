from __future__ import annotations
import argparse
import time
from typing import Optional, Union, Tuple, List, Sequence, Dict

import numpy as np
import torch
# from matplotlib import pyplot as plt
# import networkx as nx

# ------------------------- device & RNG helpers -------------------------

def rsa_default_params():
    """Return params tuned for RSA neurons (Table values in the paper)."""
    return dict(
        Cm=1.0,
        gNa=56.0,
        gK=6.0,
        gM=0.075,
        gL=0.0205,
        ENa=56.0,
        EK=-90.0,
        EL=-70.3,
        Ee=0.0,
        Ei=-75.0,
        smax=608.0,  # ms
    )

def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _gen(seed: Optional[int], device):
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g

# ------------------------- safe phi/psi helpers -------------------------

def _phi(x, k):
    """x / (1 - exp(-x/k)) with safe limit → k as x→0 (avoids 0/0)."""
    z = -x / k
    den = 1.0 - torch.exp(z)
    out = x / den
    return torch.where(torch.abs(den) < 1e-12, torch.full_like(out, k), out)

def _psi(x, k):
    """x / (exp(x/k) - 1) with safe limit → k as x→0."""
    z = x / k
    den = torch.exp(z) - 1.0
    out = x / den
    return torch.where(torch.abs(den) < 1e-12, torch.full_like(out, k), out)

# ------------------------- gating kinetics -------------------------

def steady_state_and_tau(alpha, beta, eps=1e-12):
    denom = alpha + beta + eps
    return alpha / denom, 1.0 / denom

def rates_nmh(V, VT):
    """HH core for n, m, h using VT shift (Destexhe/Pospischil/Giannari forms)."""
    x = V - VT
    a_n = 0.032 * _phi(x - 15.0, 5.0)
    b_n = 0.5   * torch.exp(-(x - 10.0) / 40.0)
    a_m = 0.32  * _phi(x - 13.0, 4.0)
    b_m = 0.28  * _psi(x - 40.0, 5.0)
    a_h = 0.128 * torch.exp(-(x - 17.0) / 18.0)
    b_h = 4.0   / (1.0 + torch.exp(-(x - 40.0) / 5.0))
    return a_n, b_n, a_m, b_m, a_h, b_h

def im_kinetics(V, smax):
    """Slow K (M‑current) gate p: returns p_inf(V), tau_p(V) in ms."""
    p_inf = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    tau_p = smax / (3.3 * torch.exp((V + 35.0) / 20.0) + torch.exp(-(V + 35.0) / 20.0))
    return p_inf, tau_p

# ------------------------- synapse kinetics -------------------------

def syn_r_ab(V_pre, sr, sd, Vth=-20.0, krel=2.0):
    """Receptor binding variable r for chemical synapses.
    a = S/sr, b = S/sr + 1/sd where S is logistic in V_pre."""
    S = 1.0 / (1.0 + torch.exp(-(V_pre - Vth) / krel))
    a = S / sr
    b = S / sr + 1.0 / sd
    r_inf = torch.where(b > 1e-12, a / b, torch.zeros_like(b))
    tau_r = torch.where(b > 1e-12, 1.0 / b, torch.full_like(b, float("inf")))
    return a, b, r_inf, tau_r

# ------------------------- connectivity -------------------------

def _to_coo(M: torch.Tensor) -> Optional[torch.Tensor]:
    idx = M.nonzero(as_tuple=False).T
    if idx.numel() == 0:
        return None
    vals = M[idx[0], idx[1]]
    return torch.sparse_coo_tensor(idx, vals, size=M.shape, device=M.device).coalesce()

@torch.no_grad()
def random_signed_adjacency_3to1(
    N: int,
    p_conn: float = 0.2,
    w_exc: Tuple[float, float] = (0.02, 0.40),
    w_inh: Tuple[float, float] = (0.02, 0.40),
    allow_self: bool = False,
    symmetric: bool = False,
    seed: Optional[int] = 69,
    device: Optional[Union[str, torch.device]] = None,
    return_sparse: bool = True,
):
    """Dale‑compliant random signed dense A with ~25% inhibitory columns.

    Returns (A_dense, A_pos_coo, A_neg_coo, is_inh_col)
    """
    if device is None:
        device = _dev()
    g = _gen(seed, device)

    # exact 3:1 split by columns
    n_inh = int(round(0.25 * N))
    is_inh_col = torch.zeros(N, dtype=torch.bool, device=device)
    if n_inh > 0:
        inh_idx = torch.randperm(N, generator=g, device=device)[:n_inh]
        is_inh_col[inh_idx] = True

    if symmetric:
        tril = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=-1 if not allow_self else 0)
        mask_upper = (torch.rand((N, N), generator=g, device=device) < p_conn) & (~tril)
        mask = mask_upper | mask_upper.t()
    else:
        mask = (torch.rand((N, N), generator=g, device=device) < p_conn)
        if not allow_self:
            mask.fill_diagonal_(False)

    U = torch.rand((N, N), generator=g, device=device)
    mag_exc = w_exc[0] + (w_exc[1] - w_exc[0]) * U
    mag_inh = w_inh[0] + (w_inh[1] - w_inh[0]) * U

    col_E = (~is_inh_col).to(torch.float32)[None, :]
    col_I = (is_inh_col).to(torch.float32)[None, :]
    mag = mag_exc * col_E + mag_inh * col_I
    sign = (1.0 * col_E) + (-1.0 * col_I)

    A = torch.zeros((N, N), device=device, dtype=torch.float32)
    A[mask] = (mag * sign)[mask]

    if symmetric and not allow_self:
        A.fill_diagonal_(0.0)

    A_pos = torch.clamp(A, min=0.0)
    A_neg = torch.clamp(-A, min=0.0)

    A_pos_coo = _to_coo(A_pos) if return_sparse else None
    A_neg_coo = _to_coo(A_neg) if return_sparse else None
    return A, A_pos_coo, A_neg_coo, is_inh_col

# ------------------------- background OU point‑conductances -------------------------

@torch.no_grad()
def precompute_point_conductances(
    T_ms: float,
    dt_ms: float,
    N: int,
    ge0: Union[float, torch.Tensor],
    gi0: Union[float, torch.Tensor],
    tau_e_ms: Union[float, torch.Tensor],
    tau_i_ms: Union[float, torch.Tensor],
    De: Union[float, torch.Tensor],
    Di: Union[float, torch.Tensor],
    seed: Optional[int] = 7,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (ge_t, gi_t) with shape (T_steps, N), independent of V."""
    device = _dev() if device is None else device
    g = _gen(seed, device)
    T_steps = int(round(T_ms / dt_ms))

    def as_vec(x):
        t = torch.as_tensor(x, device=device, dtype=dtype)
        return t.expand(N) if t.ndim == 0 else t.reshape(N).to(device=device, dtype=dtype)

    ge0 = as_vec(ge0); gi0 = as_vec(gi0)
    tau_e = as_vec(tau_e_ms); tau_i = as_vec(tau_i_ms)
    De   = as_vec(De);        Di   = as_vec(Di)

    he = torch.exp(-dt_ms / tau_e)
    hi = torch.exp(-dt_ms / tau_i)
    sig_e = torch.sqrt(De * tau_e * (1.0 - torch.exp(-2.0 * dt_ms / tau_e)))
    sig_i = torch.sqrt(Di * tau_i * (1.0 - torch.exp(-2.0 * dt_ms / tau_i)))

    ge = ge0 + torch.sqrt(De * tau_e) * torch.randn((N,), device=device, dtype=dtype, generator=g)
    gi = gi0 + torch.sqrt(Di * tau_i) * torch.randn((N,), device=device, dtype=dtype, generator=g)

    ge_t = torch.empty((T_steps, N), device=device, dtype=dtype)
    gi_t = torch.empty((T_steps, N), device=device, dtype=dtype)

    for t in range(T_steps):
        ge_t[t] = ge
        gi_t[t] = gi
        ge = ge0 + (ge - ge0) * he + sig_e * torch.randn_like(ge)
        gi = gi0 + (gi - gi0) * hi + sig_i * torch.randn_like(gi)

    return ge_t, gi_t

# ------------------------- HH step & simulation -------------------------

@torch.no_grad()
def rl_update(x, x_inf, tau, dt):
    return x_inf + (x - x_inf) * torch.exp(-dt / torch.clamp(tau, min=1e-6))

@torch.no_grad()
def ionic_currents(V, m, h, n, p, params):
    """Outward‑positive ionic currents (µA/cm^2). Includes RSA M‑current.
       I_Na = gNa m^3 h (V-ENa); I_K = gK n^4 (V-EK); I_M = gM p (V-EK); I_L = gL (V-EL)
    """
    ENa, EK, EL = params["ENa"], params["EK"], params["EL"]
    gNa, gK, gL = params["gNa"], params["gK"], params["gL"]
    gM = params.get("gM", torch.as_tensor(0.0, device=V.device, dtype=V.dtype))
    INa = gNa * (m**3) * h * (V - ENa)
    IK  = gK  * (n**4) * (V - EK)
    IM  = gM  * p      * (V - EK)
    IL  = gL  * (V - EL)
    return INa + IK + IM + IL

@torch.no_grad()
def syn_current_point(V, ge, gi, Ee=0.0, Ei=-75.0):
    return ge * (V - Ee) + gi * (V - Ei)

@torch.no_grad()
def hh_step_net(
    V, n, m, h, p, r,
    dt_ms: float,
    params: dict,
    VT: torch.Tensor,
    ge_bg: Optional[torch.Tensor] = None,
    gi_bg: Optional[torch.Tensor] = None,
    Iext_t: Optional[torch.Tensor] = None,
    Aexc_coo: Optional[torch.Tensor] = None,
    Ainh_coo: Optional[torch.Tensor] = None,
    Vsyn_e: float = 20.0,
    Vsyn_i: float = -80.0,
    sr_ms: float = 0.5,
    sd_ms: float = 8.0,
    Vth_rel: float = -20.0,
    krel: float = 2.0,
):
    # gates (Rush–Larsen)
    a_n, b_n, a_m, b_m, a_h, b_h = rates_nmh(V, VT)
    n_inf, tn = steady_state_and_tau(a_n, b_n)
    m_inf, tm = steady_state_and_tau(a_m, b_m)
    h_inf, th = steady_state_and_tau(a_h, b_h)
    p_inf, tp = im_kinetics(V, params["smax"])  # RSA gate

    n = rl_update(n, n_inf, tn, dt_ms)
    m = rl_update(m, m_inf, tm, dt_ms)
    h = rl_update(h, h_inf, th, dt_ms)
    p = rl_update(p, p_inf, tp, dt_ms)

    # intrinsic currents
    Iion = ionic_currents(V, m, h, n, p, params)

    # OU background
    Isyn_bg = torch.zeros_like(V)
    if ge_bg is not None and gi_bg is not None:
        Isyn_bg = syn_current_point(V, ge_bg, gi_bg, params.get("Ee", 0.0), params.get("Ei", -75.0))

    # chemical synapse
    if Aexc_coo is not None or Ainh_coo is not None:
        a_r, b_r, r_inf, tau_r = syn_r_ab(V, sr=sr_ms, sd=sd_ms, Vth=Vth_rel, krel=krel)
        r = rl_update(r, r_inf, tau_r, dt_ms)
        s_e = torch.zeros_like(V); s_i = torch.zeros_like(V)
        if Aexc_coo is not None:
            s_e = torch.sparse.mm(Aexc_coo, r.unsqueeze(1)).squeeze(1)
        if Ainh_coo is not None:
            s_i = torch.sparse.mm(Ainh_coo, r.unsqueeze(1)).squeeze(1)
        Ichem = s_e * (Vsyn_e - V) + s_i * (Vsyn_i - V)
    else:
        Ichem = torch.zeros_like(V)

    Iext = Iext_t if Iext_t is not None else 0.0
    Cm = params["Cm"]
    dV = (-Iion - Isyn_bg + Ichem - Iext) * (dt_ms / Cm)
    V = V + dV
    return V, n, m, h, p, r

@torch.no_grad()
def run_network(
    T_ms: float,
    dt_ms: float,
    params: dict,
    N: int,
    VT_val: float = -56.2,
    V0: float = -65.0,
    jitter: float = 3.0,
    ge_t: Optional[torch.Tensor] = None,
    gi_t: Optional[torch.Tensor] = None,
    Iext_t: Optional[torch.Tensor] = None,
    A_signed: Optional[torch.Tensor] = None,
    Aexc_coo: Optional[torch.Tensor] = None,
    Ainh_coo: Optional[torch.Tensor] = None,
    Vsyn_e: float = 20.0,
    Vsyn_i: float = -80.0,
    sr_ms: float = 0.5,
    sd_ms: float = 8.0,
    Vth_rel: float = -20.0,
    krel: float = 2.0,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
):
    device = _dev() if device is None else device
    g = _gen(seed, device)
    T_steps = int(round(T_ms / dt_ms))

    V = (torch.as_tensor(V0, device=device, dtype=dtype) + jitter * torch.randn(N, device=device, dtype=dtype, generator=g))
    VT = torch.as_tensor(VT_val, device=device, dtype=dtype).expand(N)

    a_n, b_n, a_m, b_m, a_h, b_h = rates_nmh(V, VT)
    n, _ = steady_state_and_tau(a_n, b_n)
    m, _ = steady_state_and_tau(a_m, b_m)
    h, _ = steady_state_and_tau(a_h, b_h)
    p_inf0, _ = im_kinetics(V, params["smax"])  # start RSA gate near steady state
    p = p_inf0.clone()
    r = torch.zeros(N, device=device, dtype=dtype)

    params = {k: (torch.as_tensor(v, device=device, dtype=dtype) if isinstance(v, (float, int)) else v.to(device=device, dtype=dtype)) for k, v in params.items()}

    if (Aexc_coo is None and Ainh_coo is None) and (A_signed is not None):
        A_pos = torch.clamp(A_signed, min=0.0)
        A_neg = torch.clamp(-A_signed, min=0.0)
        Aexc_coo = _to_coo(A_pos)
        Ainh_coo = _to_coo(A_neg)
    if Aexc_coo is not None: Aexc_coo = Aexc_coo.coalesce()
    if Ainh_coo is not None: Ainh_coo = Ainh_coo.coalesce()

    V_trace = torch.empty((T_steps, N), device=device, dtype=dtype)

    t0 = time.perf_counter()
    for t in range(T_steps):
        ge_bg = ge_t[t] if ge_t is not None else None
        gi_bg = gi_t[t] if gi_t is not None else None
        Iext_row = Iext_t[t] if Iext_t is not None else None
        V_trace[t] = V
        V, n, m, h, p, r = hh_step_net(
            V, n, m, h, p, r,
            dt_ms, params, VT,
            ge_bg=ge_bg, gi_bg=gi_bg, Iext_t=Iext_row,
            Aexc_coo=Aexc_coo, Ainh_coo=Ainh_coo,
            Vsyn_e=Vsyn_e, Vsyn_i=Vsyn_i,
            sr_ms=sr_ms, sd_ms=sd_ms, Vth_rel=Vth_rel, krel=krel,
        )
    elapsed = time.perf_counter() - t0
    return V_trace, elapsed


from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import torch

# ---------- your existing helpers (unchanged imports assumed) ----------
# _dev, _gen, precompute_point_conductances, random_signed_adjacency_3to1,
# rsa_default_params, run_network
# (Paste those from your file above without modification.)

# ---------- JSON sanitizer ----------
def _jsonify(d: Dict) -> str:
    def conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return x
    return json.dumps({k: conv(v) for k, v in d.items()}, indent=2, sort_keys=True)

def run_and_save(
    N: int,
    T_ms: float,
    dt_ms: float,
    p_conn: float,
    seed: int,
    outdir: Union[str, Path],
    dtype: str = "float32",
    compress: bool = True,
    save_connectivity: bool = True,
) -> Path:
    device = _dev()
    dtype_t = getattr(torch, dtype)

    # Background conductances (fixed defaults; expose if you want more knobs)
    ge_t, gi_t = precompute_point_conductances(
        T_ms=T_ms, dt_ms=dt_ms, N=N,
        ge0=0.012, gi0=0.057,
        tau_e_ms=2.7, tau_i_ms=10.5,
        De=(0.003**2)/2.7, Di=(0.066**2)/10.5,
        seed=seed, device=device, dtype=dtype_t
    )

    # Connectivity (Dale-compliant signed)
    A_signed, Apos, Aneg, is_inh = random_signed_adjacency_3to1(
        N, p_conn=p_conn, w_exc=(0.001, 0.05), w_inh=(0.01, 0.05),
        seed=1337, device=device, return_sparse=True
    )

    params = rsa_default_params()

    # Simulate
    t0 = time.perf_counter()
    V_trace, elapsed = run_network(
        T_ms=T_ms, dt_ms=dt_ms, params=params, N=N,
        VT_val=-56.2, V0=-65.0,
        ge_t=ge_t, gi_t=gi_t,
        A_signed=A_signed,
        Vsyn_e=20.0, Vsyn_i=-80.0,
        sr_ms=0.5, sd_ms=8.0, Vth_rel=-20.0, krel=2.0,
        seed=seed, device=device, dtype=dtype_t,
    )
    t1 = time.perf_counter()

    # Prepare output
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"N{N}_T{int(T_ms)}ms_dt{dt_ms}ms_seed{seed}"
    out_path = outdir / f"run_{tag}.npz"

    # Move tensors to CPU numpy
    V_np = V_trace.detach().cpu().numpy()  # [T_steps, N]
    # time vector (ms) for convenience
    T_steps = V_np.shape[0]
    tvec = (np.arange(T_steps, dtype=np.float64) * dt_ms)

    meta = {
        "N": N,
        "T_ms": T_ms,
        "dt_ms": dt_ms,
        "p_conn": p_conn,
        "seed": seed,
        "dtype": dtype,
        "device": str(device),
        "elapsed_s_inner": elapsed,
        "elapsed_s_total": t1 - t0,
        "V_shape": list(V_np.shape),
        "code_version": "v1",
    }
    params_json = _jsonify(params)
    meta_json   = _jsonify(meta)

    # Connectivity payload (optional; keep small)
    # Save signs and inhibitory labels in compact form
    if save_connectivity:
        A_signed_cpu = A_signed.detach().cpu().numpy().astype(np.float32)
        is_inh_cpu   = is_inh.detach().cpu().numpy().astype(np.bool_)
    else:
        A_signed_cpu = np.array([], dtype=np.float32)
        is_inh_cpu   = np.array([], dtype=np.bool_)

    # Write .npz (optionally compressed)
    if compress:
        np.savez_compressed(
            out_path,
            V=V_np,
            t_ms=tvec,
            A_signed=A_signed_cpu,
            is_inh=is_inh_cpu,
            meta_json=meta_json,
            hh_params_json=params_json,
        )
    else:
        np.savez(
            out_path,
            V=V_np,
            t_ms=tvec,
            A_signed=A_signed_cpu,
            is_inh=is_inh_cpu,
            meta_json=meta_json,
            hh_params_json=params_json,
        )

    return out_path

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run RSA HH network and save full voltage array to .npz")
    ap.add_argument("--N", type=int, default=100, help="number of neurons")
    ap.add_argument("--T", type=float, default=5000.0, help="simulation length (ms)")
    ap.add_argument("--dt", type=float, default=0.05, help="time step (ms)")
    ap.add_argument("--p_conn", type=float, default=0.10, help="connection probability")
    ap.add_argument("--seed", type=int, default=2025, help="random seed")
    ap.add_argument("--out_dir", type=str, default="runs_out", help="output directory")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="compute dtype")
    ap.add_argument("--no-compress", action="store_true", help="use np.savez (uncompressed)")
    ap.add_argument("--no-connectivity", action="store_true", help="do not include A_signed/is_inh in npz")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    out_path = run_and_save(
        N=args.N,
        T_ms=args.T,
        dt_ms=args.dt,
        p_conn=args.p_conn,
        seed=args.seed,
        outdir=args.out_dir,
        dtype=args.dtype,
        compress=not args.no_compress,
        save_connectivity=not args.no_connectivity,
    )
    print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    main()
