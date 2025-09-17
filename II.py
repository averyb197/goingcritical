#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regular‑Spiking‑with‑Adaptation (RSA) HH Network Simulator
=========================================================

What this script does (tidy version of your sandbox):
  • Simulates a network of Hodgkin–Huxley neurons using Rush–Larsen gate updates
    and a point‑conductance OU background (Destexhe‑style) for ge/gi.
  • Adds the missing slow K (M‑current) gate p for RSA neurons with the standard
    p_inf(V) and tau_p(V) kinetics.
  • Chemical synapses use a first‑order receptor fraction r with sigmoid release.
  • Dale‑compliant random connectivity with an enforced ~3:1 E:I column ratio.

Plots produced (and only these):
  1) Network topology (prettier: sized by degree, colored by E/I, weighted edges)
  2) Raster (all neurons)
  3) Population firing rate (Hz per neuron)
  4) Beggs‑style avalanche size and duration distributions (log‑log)

Notes:
  • Units are in ms, mV, mS/cm^2, µA/cm^2 throughout.
  • Defaults target the RSA parameter set (Giannari & Astolfi 2022 table values),
    but FS behavior emerges if gM=0; we keep code generic.
"""

from __future__ import annotations
import argparse
import time
from typing import Optional, Union, Tuple, List, Sequence, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
import networkx as nx

# ------------------------- device & RNG helpers -------------------------

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

# ------------------------- spike detection & raster -------------------------

@torch.no_grad()
def detect_spikes(
    V_trace: torch.Tensor,     # [T, N]
    dt_ms: float,
    thr_mV: float = -20.0,
    refractory_ms: float = 2.0,
    slope_min_mV_per_ms: float = 5.0,
    interpolate: bool = True,
):
    T, N = V_trace.shape
    device = V_trace.device
    thr = torch.as_tensor(thr_mV, device=device, dtype=V_trace.dtype)
    dV = (V_trace[1:] - V_trace[:-1]) / dt_ms
    below = V_trace[:-1] < thr
    above = V_trace[1:] >= thr
    slope_ok = dV >= slope_min_mV_per_ms
    crossings = below & above & slope_ok
    ref_steps = max(1, int(round(refractory_ms / dt_ms)))

    spike_ix_per_neuron: List[torch.Tensor] = []
    spike_t_per_neuron: List[torch.Tensor] = []

    for j in range(N):
        idx = torch.nonzero(crossings[:, j], as_tuple=False).flatten()
        if idx.numel() == 0:
            spike_ix_per_neuron.append(idx.to(torch.long))
            spike_t_per_neuron.append(idx.to(V_trace.dtype))
            continue
        keep = []
        last = -10**9
        for k in idx.tolist():
            if k - last >= ref_steps:
                keep.append(k)
                last = k
        keep = torch.tensor(keep, device=device, dtype=torch.long)
        if interpolate:
            Vk   = V_trace[keep, j]
            Vk1  = V_trace[keep + 1, j]
            frac = (thr - Vk) / torch.clamp(Vk1 - Vk, min=1e-6)
            t_ms = (keep.to(V_trace.dtype) + frac) * dt_ms
            spike_t_per_neuron.append(t_ms)
        else:
            spike_t_per_neuron.append((keep.to(V_trace.dtype) + 1.0) * dt_ms)
        spike_ix_per_neuron.append(keep)

    return spike_ix_per_neuron, spike_t_per_neuron

# ------------------------- Beggs‑style avalanches -------------------------

@torch.no_grad()
def avalanches_beggs_fixed(
    spike_t_per_neuron: List[torch.Tensor],
    T_ms: float,
    bin_ms: float = 4.0,
) -> Dict[str, np.ndarray]:
    times, nids = [], []
    for j, tj in enumerate(spike_t_per_neuron):
        if len(tj):
            arr = tj.detach().cpu().numpy().astype(np.float64)
            times.append(arr)
            nids.append(np.full(arr.shape, j, dtype=np.int64))
    if not times:
        return dict(size_spikes=np.array([]), size_unique_neurons=np.array([]), duration_bins=np.array([]),
                    duration_ms=np.array([]), starts_ms=np.array([]), ends_ms=np.array([]), bin_ms=float(bin_ms))
    times = np.concatenate(times); nids = np.concatenate(nids)

    M = int(np.ceil(T_ms / bin_ms))
    edges = np.arange(M + 1, dtype=np.float64) * bin_ms
    edges[-1] = max(edges[-1], T_ms)
    bin_idx = np.clip((times / bin_ms).astype(np.int64), 0, M - 1)

    counts = np.bincount(bin_idx, minlength=M)
    nz = np.nonzero(counts > 0)[0]
    if nz.size == 0:
        return dict(size_spikes=np.array([]), size_unique_neurons=np.array([]), duration_bins=np.array([]),
                    duration_ms=np.array([]), starts_ms=np.array([]), ends_ms=np.array([]), bin_ms=float(bin_ms))

    split_pts = np.where(np.diff(nz) > 1)[0] + 1
    runs = np.split(nz, split_pts)

    size_spikes, size_unique = [], []
    duration_bins, duration_ms = [], []
    starts_ms, ends_ms = [], []

    for r in runs:
        b0, b1 = int(r[0]), int(r[-1])
        dur_b = b1 - b0 + 1
        duration_bins.append(dur_b)
        duration_ms.append(dur_b * bin_ms)
        starts_ms.append(b0 * bin_ms)
        ends_ms.append(min((b1 + 1) * bin_ms, T_ms))
        size_spikes.append(int(counts[b0:b1 + 1].sum()))
        mask = (bin_idx >= b0) & (bin_idx <= b1)
        size_unique.append(int(np.unique(nids[mask]).size))

    return dict(
        size_spikes=np.asarray(size_spikes, dtype=int),
        size_unique_neurons=np.asarray(size_unique, dtype=int),
        duration_bins=np.asarray(duration_bins, dtype=int),
        duration_ms=np.asarray(duration_ms, dtype=float),
        starts_ms=np.asarray(starts_ms, dtype=float),
        ends_ms=np.asarray(ends_ms, dtype=float),
        bin_ms=float(bin_ms),
    )

# ------------------------- plotting -------------------------

def plot_topology_from_A(A_signed: torch.Tensor, is_inh_col: torch.Tensor,
                         max_edges=4000, seed=0, out: Optional[str] = None):
    """Prettier directed graph: node size by degree, color by type (E/I), edge width by |w|."""
    if hasattr(A_signed, "detach"):
        A_np = A_signed.detach().cpu().numpy()
    else:
        A_np = np.asarray(A_signed)
    N = A_np.shape[0]

    G = nx.from_numpy_array(A_np, create_using=nx.DiGraph)
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0.0) == 0.0])

    if G.number_of_edges() > max_edges:
        edges_sorted = sorted(G.edges(data=True), key=lambda e: abs(e[2]["weight"]), reverse=True)[:max_edges]
        G = nx.DiGraph(); G.add_nodes_from(range(N)); G.add_edges_from([(u, v, d) for u, v, d in edges_sorted])

    pos = nx.spring_layout(G, seed=seed, k=1.0 / np.sqrt(max(N, 1)))

    deg = np.array([G.degree(n) for n in G.nodes()], dtype=float)
    node_sizes = 50 + 150 * (deg / (deg.max() + 1e-6))
    node_colors = ["tab:red" if bool(is_inh_col[n].item()) else "tab:blue" for n in G.nodes()]

    weights = np.array([abs(d["weight"]) for _, _, d in G.edges(data=True)])
    wmax = weights.max() if len(weights) else 1.0
    widths = 0.5 + 2.5 * (weights / wmax)
    colors = ["#aa0000" if G[u][v]["weight"] < 0 else "#0066cc" for u, v in G.edges()]

    plt.figure(figsize=(9, 9), dpi=150)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.0)
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color=colors, width=widths, alpha=0.5)
    plt.title(f"Network topology (N={N}, E={G.number_of_edges()})")
    plt.axis("off")
    if out:
        plt.savefig(out, bbox_inches="tight")


def plot_raster_all(spike_t_per_neuron: List[torch.Tensor], T_ms: float, title: str = "Raster (all neurons)"):
    seq = []
    for tj in spike_t_per_neuron:
        tj = tj.detach().cpu().numpy() if isinstance(tj, torch.Tensor) else np.asarray(tj, dtype=float)
        seq.append(tj)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax.eventplot(seq, orientation="horizontal", colors="k",
                 lineoffsets=np.arange(len(seq)), linelengths=0.8, linewidths=0.6)
    ax.set_xlim(0, T_ms)
    ax.set_ylim(-0.5, len(seq) - 0.5)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Neuron index")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(title)


def plot_population_rate(spike_t_per_neuron: List[torch.Tensor], N: int, T_ms: float, bin_ms: float = 5.0):
    all_spikes = [t.cpu().numpy() for t in spike_t_per_neuron if len(t)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
    if len(all_spikes):
        all_spikes = np.concatenate(all_spikes)
        edges = np.arange(0.0, T_ms + bin_ms, bin_ms, dtype=np.float64)
        counts, _ = np.histogram(all_spikes, bins=edges)
        bin_s = bin_ms / 1000.0
        rate_hz_per_neuron = counts / (N * bin_s)
        ctrs = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(ctrs, rate_hz_per_neuron)
    else:
        ax.text(0.5, 0.5, "No spikes detected", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(0, T_ms)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Population rate (Hz/neuron)")
    ax.set_title(f"Population firing rate (bin = {bin_ms} ms)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_avalanche_histograms(summary: Dict[str, np.ndarray]):
    sizes = summary["size_spikes"]
    durs  = summary["duration_bins"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if sizes.size:
        u_s, c_s = np.unique(sizes, return_counts=True)
        ax1.loglog(u_s, c_s, marker="o", linestyle="-")
    else:
        ax1.text(0.5, 0.5, "No avalanches", transform=ax1.transAxes, ha="center", va="center")
    if durs.size:
        u_d, c_d = np.unique(durs,  return_counts=True)
        ax2.loglog(u_d, c_d, marker="o", linestyle="-")
    else:
        ax2.text(0.5, 0.5, "No avalanches", transform=ax2.transAxes, ha="center", va="center")
    ax1.set_xlabel("Avalanche size (spikes)"); ax1.set_ylabel("Count"); ax1.set_title("Size distribution")
    ax2.set_xlabel("Avalanche duration (bins)"); ax2.set_ylabel("Count"); ax2.set_title("Duration distribution")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ------------------------- RSA defaults (Giannari/Astolfi 2022) -------------------------

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

# ------------------------- CLI harness -------------------------

def main():
    ap = argparse.ArgumentParser(description="RSA HH network with OU background and chemical synapses")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--T", type=float, default=5000.0, help="simulation length (ms)")
    ap.add_argument("--dt", type=float, default=0.05, help="time step (ms)")
    ap.add_argument("--p_conn", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--plot", action="store_true", help="show figures")
    args = ap.parse_args()

    device = _dev()

    # Background OU (Destexhe point‑conductance)
    ge_t, gi_t = precompute_point_conductances(
        T_ms=args.T,
        dt_ms=args.dt,
        N=args.N,
        ge0=0.012, gi0=0.057,
        tau_e_ms=2.7, tau_i_ms=10.5,
        De=(0.003**2)/2.7,  # 0.003 mS baseline sigma_e (matches your earlier numbers, scaled)
        Di=(0.066**2)/10.5,
        seed=args.seed,
        device=device,
    )

    # Connectivity
    A_signed, Apos, Aneg, is_inh = random_signed_adjacency_3to1(
        args.N, p_conn=args.p_conn, w_exc=(0.001, 0.05), w_inh=(0.01, 0.05), seed=1337,
        device=device, return_sparse=True
    )

    params = rsa_default_params()

    V_trace, elapsed = run_network(
        T_ms=args.T, dt_ms=args.dt, params=params, N=args.N,
        VT_val=-56.2, V0=-65.0,
        ge_t=ge_t, gi_t=gi_t,
        A_signed=A_signed,
        Vsyn_e=20.0, Vsyn_i=-80.0,
        sr_ms=0.5, sd_ms=8.0, Vth_rel=-20.0, krel=2.0,
        seed=args.seed, device=device
    )

    # Spikes & summaries
    spike_ix, spike_t = detect_spikes(V_trace, dt_ms=args.dt, thr_mV=-20.0, refractory_ms=2.0,
                                      slope_min_mV_per_ms=5.0, interpolate=True)

    # ---- PLOTS (exactly the four requested) ----
    plot_topology_from_A(A_signed, is_inh, max_edges=4000, seed=0)
    plot_raster_all(spike_t, T_ms=args.T)
    plot_population_rate(spike_t, N=args.N, T_ms=args.T, bin_ms=5.0)
    summary = avalanches_beggs_fixed(spike_t, T_ms=args.T, bin_ms=4.0)
    print(f"bin = {summary['bin_ms']} ms, #avalanches = {summary['size_spikes'].size}")
    plot_avalanche_histograms(summary)

    print(f"V_trace shape: {tuple(V_trace.shape)}, elapsed: {elapsed:.3f}s, Vmin={V_trace.min().item():.2f}, Vmax={V_trace.max().item():.2f}")

    if args.plot:
        plt.show()
    else:
        # Save figures if not showing
        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(f"figure_{i+1}.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
