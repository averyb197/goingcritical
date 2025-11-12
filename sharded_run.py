from __future__ import annotations
import argparse, json, os, time, math, sys, logging
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
import torch

# =========================================
# Logging & timing
# =========================================

class Timer:
    def __init__(self, logger: logging.Logger, msg: str):
        self.log = logger
        self.msg = msg
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log.info(f"[start] {self.msg}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.log.info(f"[done ] {self.msg} in {dt:.4f}s")

def setup_logger(log_path: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("hhnet")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def log_cuda_mem(logger: logging.Logger, prefix: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        logger.info(f"{prefix}CUDA mem: allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

# =========================================
# Device & RNG
# =========================================

def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _gen(seed: Optional[int], device):
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g

# =========================================
# Safe helpers
# =========================================

def _phi(x, k):
    z = -x / k
    den = 1.0 - torch.exp(z)
    out = x / den
    return torch.where(torch.abs(den) < 1e-12, torch.full_like(out, k), out)

def _psi(x, k):
    z = x / k
    den = torch.exp(z) - 1.0
    out = x / den
    return torch.where(torch.abs(den) < 1e-12, torch.full_like(out, k), out)

# =========================================
# Kinetics
# =========================================

def steady_state_and_tau(alpha, beta, eps=1e-12):
    denom = alpha + beta + eps
    return alpha / denom, 1.0 / denom

def rates_nmh(V, VT):
    x = V - VT
    a_n = 0.032 * _phi(x - 15.0, 5.0)
    b_n = 0.5   * torch.exp(-(x - 10.0) / 40.0)
    a_m = 0.32  * _phi(x - 13.0, 4.0)
    b_m = 0.28  * _psi(x - 40.0, 5.0)
    a_h = 0.128 * torch.exp(-(x - 17.0) / 18.0)
    b_h = 4.0   / (1.0 + torch.exp(-(x - 40.0) / 5.0))
    return a_n, b_n, a_m, b_m, a_h, b_h

def im_kinetics(V, smax):
    p_inf = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    tau_p = smax / (3.3 * torch.exp((V + 35.0) / 20.0) + torch.exp(-(V + 35.0) / 20.0))
    return p_inf, tau_p

def syn_r_ab(V_pre, sr, sd, Vth=-20.0, krel=2.0):
    S = 1.0 / (1.0 + torch.exp(-(V_pre - Vth) / krel))
    a = S / sr
    b = S / sr + 1.0 / sd
    r_inf = torch.where(b > 1e-12, a / b, torch.zeros_like(b))
    tau_r = torch.where(b > 1e-12, 1.0 / b, torch.full_like(b, float("inf")))
    return a, b, r_inf, tau_r

# =========================================
# Connectivity (COO only, per your edict)
# =========================================

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
    device = _dev() if device is None else device
    g = _gen(seed, device)

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

class OUSource:
    def __init__(self, N, ge0, gi0, tau_e_ms, tau_i_ms, De, Di, dt_ms,
                 seed=None, device=None, dtype=torch.float32, logger: Optional[logging.Logger]=None):
        self.device = _dev() if device is None else device
        self.dtype = dtype
        self.logger = logger
        # keep RNG on the instance
        self.rng = _gen(seed, self.device)

        def vec(x):
            t = torch.as_tensor(x, device=self.device, dtype=dtype)
            return t.expand(N) if t.ndim == 0 else t.reshape(N).to(self.device, dtype)

        self.ge0, self.gi0 = vec(ge0), vec(gi0)
        self.tau_e, self.tau_i = vec(tau_e_ms), vec(tau_i_ms)
        self.De, self.Di = vec(De), vec(Di)
        self.dt = torch.as_tensor(dt_ms, device=self.device, dtype=dtype)

        self.he = torch.exp(-self.dt / self.tau_e)
        self.hi = torch.exp(-self.dt / self.tau_i)
        self.sig_e = torch.sqrt(self.De * self.tau_e * (1.0 - torch.exp(-2.0 * self.dt / self.tau_e)))
        self.sig_i = torch.sqrt(self.Di * self.tau_i * (1.0 - torch.exp(-2.0 * self.dt / self.tau_i)))

        # ↓↓↓ replace randn_like(..., generator=...) with randn(..., generator=self.rng)
        self.ge = self.ge0 + torch.sqrt(self.De * self.tau_e) * torch.randn(
            self.ge0.shape, device=self.device, dtype=self.dtype, generator=self.rng
        )
        self.gi = self.gi0 + torch.sqrt(self.Di * self.tau_i) * torch.randn(
            self.gi0.shape, device=self.device, dtype=self.dtype, generator=self.rng
        )

        if self.logger:
            nb = (self.ge.numel() + self.gi.numel()) * self.ge.element_size()
            self.logger.info(f"OU initialized, state bytes on device: {nb/1024/1024:.2f} MB")

    @torch.no_grad()
    def step_pair(self):
        ge_now, gi_now = self.ge, self.gi
        # ↓↓↓ also avoid randn_like here for reproducibility with the same RNG
        ge_next = self.ge0 + (self.ge - self.ge0) * self.he + self.sig_e * torch.randn(
            self.ge.shape, device=self.device, dtype=self.dtype, generator=self.rng
        )
        gi_next = self.gi0 + (self.gi - self.gi0) * self.hi + self.sig_i * torch.randn(
            self.gi.shape, device=self.device, dtype=self.dtype, generator=self.rng
        )
        self.ge, self.gi = ge_next, gi_next
        return ge_now, gi_now, ge_next, gi_next


# =========================================
# HH dynamics (2nd order, midpoint in state and time)
# =========================================


#for tweaking the background drive
def compute_bg_from_ratio(
    rho: float, Gbg: float, tau_e: float, tau_i: float,
    CVe: float = 0.3, CVi: float = 0.3
):
    ge0 = Gbg * (rho / (1.0 + rho))
    gi0 = Gbg * (1.0 / (1.0 + rho))
    De  = (CVe**2) * (ge0**2) / tau_e
    Di  = (CVi**2) * (gi0**2) / tau_i
    return ge0, gi0, De, Di

def neutral_rho(Vstar: float, Ee: float = 0.0, Ei: float = -75.0) -> float:
    return (Ei - Vstar) / (Vstar - Ee)


@torch.no_grad()
def rl_update(x, x_inf, tau, dt):
    return x_inf + (x - x_inf) * torch.exp(-dt / torch.clamp(tau, min=1e-6))

@torch.no_grad()
def ionic_currents(V, m, h, n, p, params):
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
def hh_step_second_order(
    V, n, m, h, p, r,
    dt_ms: float,
    params: dict,
    VT: torch.Tensor,
    # time t inputs
    ge_bg: Optional[torch.Tensor] = None,
    gi_bg: Optional[torch.Tensor] = None,
    Iext_t: Optional[torch.Tensor] = None,
    # time t+dt inputs (for midpoint in time)
    ge_bg_next: Optional[torch.Tensor] = None,
    gi_bg_next: Optional[torch.Tensor] = None,
    Iext_next: Optional[torch.Tensor] = None,
    Aexc_coo: Optional[torch.Tensor] = None,
    Ainh_coo: Optional[torch.Tensor] = None,
    Vsyn_e: float = 20.0,
    Vsyn_i: float = -80.0,
    sr_ms: float = 0.5,
    sd_ms: float = 8.0,
    Vth_rel: float = -20.0,
    krel: float = 2.0,
):
    # ===== Stage 1: derivative at current state (Euler half-step predictor) =====
    a_n0, b_n0, a_m0, b_m0, a_h0, b_h0 = rates_nmh(V, VT)
    n_inf0, tn0 = steady_state_and_tau(a_n0, b_n0)
    m_inf0, tm0 = steady_state_and_tau(a_m0, b_m0)
    h_inf0, th0 = steady_state_and_tau(a_h0, b_h0)
    p_inf0, tp0 = im_kinetics(V, params["smax"])

    f_n0 = (n_inf0 - n) / tn0
    f_m0 = (m_inf0 - m) / tm0
    f_h0 = (h_inf0 - h) / th0
    f_p0 = (p_inf0 - p) / tp0

    _, _, r_inf0, tau_r0 = syn_r_ab(V, sr=sr_ms, sd=sd_ms, Vth=Vth_rel, krel=krel)
    f_r0 = (r_inf0 - r) / torch.clamp(tau_r0, min=1e-6)

    if ge_bg is not None and gi_bg is not None:
        Isyn_bg0 = syn_current_point(V, ge_bg, gi_bg, params.get("Ee", 0.0), params.get("Ei", -75.0))
    else:
        Isyn_bg0 = torch.zeros_like(V)

    s_e0 = torch.zeros_like(V)
    s_i0 = torch.zeros_like(V)
    if Aexc_coo is not None or Ainh_coo is not None:
        if Aexc_coo is not None:
            s_e0 = torch.sparse.mm(Aexc_coo, r.unsqueeze(1)).squeeze(1)
        if Ainh_coo is not None:
            s_i0 = torch.sparse.mm(Ainh_coo, r.unsqueeze(1)).squeeze(1)
        Ichem0 = s_e0 * (Vsyn_e - V) + s_i0 * (Vsyn_i - V)
    else:
        Ichem0 = torch.zeros_like(V)

    Iion0 = ionic_currents(V, m, h, n, p, params)
    Iext_val = Iext_t if Iext_t is not None else 0.0
    f_V0 = (-Iion0 - Isyn_bg0 + Ichem0 - Iext_val) / params["Cm"]

    half_dt = 0.5 * dt_ms
    V_t = V + half_dt * f_V0
    n_t = n + half_dt * f_n0
    m_t = m + half_dt * f_m0
    h_t = h + half_dt * f_h0
    p_t = p + half_dt * f_p0
    r_t = r + half_dt * f_r0

    # ===== Stage 2: coefficients at (state midpoint, time midpoint) =====
    a_n_t, b_n_t, a_m_t, b_m_t, a_h_t, b_h_t = rates_nmh(V_t, VT)
    n_inf_t, tn_t = steady_state_and_tau(a_n_t, b_n_t)
    m_inf_t, tm_t = steady_state_and_tau(a_m_t, b_m_t)
    h_inf_t, th_t = steady_state_and_tau(a_h_t, b_h_t)
    p_inf_t, tp_t = im_kinetics(V_t, params["smax"])

    n_new = rl_update(n, n_inf_t, tn_t, dt_ms)
    m_new = rl_update(m, m_inf_t, tm_t, dt_ms)
    h_new = rl_update(h, h_inf_t, th_t, dt_ms)
    p_new = rl_update(p, p_inf_t, tp_t, dt_ms)

    _, _, r_inf_t, tau_r_t = syn_r_ab(V_t, sr=sr_ms, sd=sd_ms, Vth=Vth_rel, krel=krel)
    r_new = rl_update(r, r_inf_t, tau_r_t, dt_ms)

    s_e_t = torch.zeros_like(V)
    s_i_t = torch.zeros_like(V)
    if Aexc_coo is not None or Ainh_coo is not None:
        if Aexc_coo is not None:
            s_e_t = torch.sparse.mm(Aexc_coo, r_t.unsqueeze(1)).squeeze(1)
        if Ainh_coo is not None:
            s_i_t = torch.sparse.mm(Ainh_coo, r_t.unsqueeze(1)).squeeze(1)

    # time-midpoint for exogenous inputs
    def mid(a, b):
        if (a is not None) and (b is not None):
            return 0.5 * (a + b)
        return a if a is not None else 0.0

    ge_mid = mid(ge_bg, ge_bg_next)
    gi_mid = mid(gi_bg, gi_bg_next)
    Iext_mid = mid(Iext_t, Iext_next)

    gNa_eff = params["gNa"] * (m_t**3) * h_t
    gK_eff  = params["gK"]  * (n_t**4)
    gM_eff  = params.get("gM", torch.as_tensor(0.0, device=V.device, dtype=V.dtype)) * p_t
    gL_eff  = params["gL"]
    g_syn   = s_e_t + s_i_t

    g_tot = gNa_eff + gK_eff + gM_eff + gL_eff + ge_mid + gi_mid + g_syn
    numA  = (
        gNa_eff * params["ENa"] +
        gK_eff  * params["EK"]  +
        gM_eff  * params["EK"]  +
        gL_eff  * params["EL"]  +
        ge_mid  * params.get("Ee", 0.0) +
        gi_mid  * params.get("Ei", -75.0) +
        s_e_t   * Vsyn_e +
        s_i_t   * Vsyn_i -
        Iext_mid
    )
    B_V = g_tot / params["Cm"]
    A_V = numA  / params["Cm"]

    # Exponential Euler with phi1
    z = -B_V * dt_ms
    ez = torch.exp(z)
    phi1 = torch.where(torch.abs(z) < 1e-8, torch.ones_like(z), (torch.exp(z) - 1.0) / z)
    V_new = ez * V + phi1 * (A_V * dt_ms)

    return V_new, n_new, m_new, h_new, p_new, r_new

# try torch.compile for speed
try:
    hh_step_second_order = torch.compile(hh_step_second_order, dynamic=True, mode="max-autotune")
except Exception as _e:
    # Fine, run uncompiled. Log at runtime.
    pass

# =========================================
# Defaults
# =========================================

def rsa_default_params():
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
        smax=608.0,
    )



class NPZShardWriter:
    """Writes samples to per-chunk .npz files with integer step metadata.
    This makes shards stitchable for full-resolution spike detection.
    """
    def __init__(self, outdir: Path, base: str, dt_ms: float, decimate: int, logger: logging.Logger, compress: bool = False):
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.base = base
        self.dt_ms = float(dt_ms)
        self.dec = int(max(1, decimate))
        self.shard = 0
        self.log = logger
        self.compress = compress

    def write_chunk(self, step0: int, V_chunk: torch.Tensor):
        """V_chunk: [rows, N] on CPU. Rows correspond to steps:
           step = step0 + i*keep_every, i=0..rows-1
        """
        assert V_chunk.device.type == "cpu", "write_chunk expects CPU tensor"
        rows, N = V_chunk.shape
        step1 = step0 + rows * self.dec  # exclusive end step
        fn = self.outdir / f"{self.base}_shard{self.shard:05d}.npz"
        with Timer(self.log, f"write shard {self.shard} -> {fn.name}"):
            if self.compress:
                np.savez_compressed(
                    fn,
                    V=V_chunk.numpy(),
                    # integer continuity metadata (authoritative)
                    step0=np.int64(step0),
                    step1=np.int64(step1),          # exclusive
                    keep_every=np.int64(self.dec),  # step stride
                    # convenience
                    dt_ms=np.float64(self.dt_ms),
                    N=np.int64(N),
                    dtype=np.bytes_(str(V_chunk.numpy().dtype)),
                )
            else:
                np.savez(
                    fn,
                    V=V_chunk.numpy(),
                    # integer continuity metadata (authoritative)
                    step0=np.int64(step0),
                    step1=np.int64(step1),  # exclusive
                    keep_every=np.int64(self.dec),  # step stride
                    # convenience
                    dt_ms=np.float64(self.dt_ms),
                    N=np.int64(N),
                    dtype=np.bytes_(str(V_chunk.numpy().dtype)),
                )

        self.shard += 1

@torch.no_grad()
def run_network(
    *,
    T_ms: float,
    dt_ms: float,
    params: dict,
    N: int,
    VT_val: float = -56.2,
    V0: float = -65.0,
    jitter: float = 3.0,
    ou: Optional[OUSource] = None,
    Iext_callable = None,
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
    # chunking / recording
    chunk_steps: int = 2000,
    decimate: int = 1,
    writer: Optional[NPZShardWriter] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    device = _dev() if device is None else device
    g = _gen(seed, device)
    T_steps = int(round(T_ms / dt_ms))

    if logger is None:
        logger = setup_logger()

    with Timer(logger, "state allocation"):
        V = (torch.as_tensor(V0, device=device, dtype=dtype) +
             jitter * torch.randn(N, device=device, dtype=dtype, generator=g))
        VT = torch.as_tensor(VT_val, device=device, dtype=dtype).expand(N)

        a_n, b_n, a_m, b_m, a_h, b_h = rates_nmh(V, VT)
        n, _ = steady_state_and_tau(a_n, b_n)
        m, _ = steady_state_and_tau(a_m, b_m)
        h, _ = steady_state_and_tau(a_h, b_h)
        p_inf0, _ = im_kinetics(V, params["smax"])
        p = p_inf0.clone()
        r = torch.zeros(N, device=device, dtype=dtype)

        params = {k: (torch.as_tensor(v, device=device, dtype=dtype)
                      if isinstance(v, (float, int)) else v.to(device=device, dtype=dtype))
                  for k, v in params.items()}

        if (Aexc_coo is None and Ainh_coo is None) and (A_signed is not None):
            A_pos = torch.clamp(A_signed, min=0.0)
            A_neg = torch.clamp(-A_signed, min=0.0)
            Aexc_coo = _to_coo(A_pos)
            Ainh_coo = _to_coo(A_neg)
        if Aexc_coo is not None: Aexc_coo = Aexc_coo.coalesce()
        if Ainh_coo is not None: Ainh_coo = Ainh_coo.coalesce()

    log_cuda_mem(logger, "[after alloc] ")

    keep_every = int(max(1, decimate))
    step = 0
    shard_start_step = 0
    t_sim = time.perf_counter()

    while step < T_steps:
        # chunk bounds
        s_end = min(step + chunk_steps, T_steps)
        logger.info(f"[chunk] steps {step}..{s_end-1} (len={s_end-step})")

        # compute how many kept rows this chunk will produce, exactly
        steps_in_chunk = s_end - step
        # first kept index within chunk is the smallest i >= 0 such that (step+i) % keep_every == 0
        offset = (-step) % keep_every
        first_kept = offset if offset < steps_in_chunk else steps_in_chunk
        n_keep = 0 if first_kept == steps_in_chunk else 1 + (steps_in_chunk - 1 - first_kept) // keep_every

        # allocate CPU buffer for this chunk's kept rows (full res when keep_every==1)
        V_buf_cpu = torch.empty((n_keep, N), device="cpu", dtype=dtype)
        kept_idx = 0

        # integrate this chunk
        while step < s_end:
            if ou is not None:
                ge_now, gi_now, ge_next, gi_next = ou.step_pair()
            else:
                ge_now = gi_now = ge_next = gi_next = None

            if Iext_callable is not None:
                I_now, I_next = Iext_callable(step)
            else:
                I_now = I_next = None

            V, n, m, h, p, r = hh_step_second_order(
                V, n, m, h, p, r,
                dt_ms, params, VT,
                ge_bg=ge_now, gi_bg=gi_now, Iext_t=I_now,
                ge_bg_next=ge_next, gi_bg_next=gi_next, Iext_next=I_next,
                Aexc_coo=Aexc_coo, Ainh_coo=Ainh_coo,
                Vsyn_e=Vsyn_e, Vsyn_i=Vsyn_i,
                sr_ms=sr_ms, sd_ms=sd_ms, Vth_rel=Vth_rel, krel=krel,
            )

            if (step % keep_every) == 0 and writer is not None:
                V_buf_cpu[kept_idx] = V.detach().to("cpu", non_blocking=True)
                kept_idx += 1

            step += 1

        # end of chunk: write shard exactly with global step index for its first kept row
        if writer is not None and n_keep > 0:
            # the first kept global step in this chunk is chunk_start + offset
            first_global_step_kept = shard_start_step + ((-shard_start_step) % keep_every)
            # BUT shard_start_step is the first step of the chunk; make it equal to the chunk's 'step' at entry
            # We recorded 'step' above; reconstruct here:
            first_global_step_kept = (s_end - steps_in_chunk) + ((-(s_end - steps_in_chunk)) % keep_every)
            writer.write_chunk(first_global_step_kept, V_buf_cpu)

        shard_start_step = s_end
        log_cuda_mem(logger, "[chunk end] ")

    sim_time = time.perf_counter() - t_sim
    logger.info(f"[total] simulated {T_steps} steps in {sim_time:.3f}s")
    return {"sim_seconds": sim_time}


# =========================================
# CLI
# =========================================

def build_argparser():
    ap = argparse.ArgumentParser(description="Chunked HH network with OU background, 2nd order scheme, sharded NPZ writer")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--T", type=float, default=5000.0, help="simulation length (ms)")
    ap.add_argument("--dt", type=float, default=0.05, help="time step (ms)")
    ap.add_argument("--p_conn", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_dir", type=str, default="runs_out")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--decimate", type=int, default=1, help="keep every k-th sample in output shards")
    ap.add_argument("--chunk_steps", type=int, default=48000, help="steps per output shard")
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile")
    return ap

def main():
    args = build_argparser().parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(outdir / f"{args.tag}.log")

    device = _dev()
    logger.info(f"device: {device} | torch {torch.__version__}")

    # torch.compile toggle
    if args.no_compile and hasattr(hh_step_second_order, "compiler_fn"):
        logger.info("Disabling compiled kernel per flag")
        # Can't un-compile easily; just warn. Next run use --no_compile at import time if you care.

    # Connectivity
    with Timer(logger, "build connectivity"):
        A_signed, Apos, Aneg, is_inh = random_signed_adjacency_3to1(
            args.N, p_conn=args.p_conn, w_exc=(0.001, 0.05), w_inh=(0.01, 0.05),
            seed=1337, device=device, return_sparse=True
        )

    # OU source (streamed)
    with Timer(logger, "init OU source"):
        # ou = OUSource(
        #     N=args.N,
        #     ge0=0.012, gi0=0.057,
        #     tau_e_ms=2.7, tau_i_ms=10.5,
        #     De=(0.003**2)/2.7,
        #     Di=(0.066**2)/10.5,
        #     dt_ms=args.dt,
        #     seed=args.seed,
        #     device=device,
        #     dtype=torch.float32,
        # )
        Gbg = 0.069
        rho = neutral_rho(-65.0)  # find a neautral ratio...
        ge0, gi0, De, Di = compute_bg_from_ratio(rho, Gbg, tau_e=2.7, tau_i=10.5, CVe=0.3, CVi=0.3)

        ou = OUSource(
            N=args.N,
            ge0=ge0, gi0=gi0,
            tau_e_ms=2.7, tau_i_ms=10.5,
            De=De, Di=Di,
            dt_ms=args.dt,
            seed=args.seed,
            device=device,
            dtype=torch.float32,
        )

    # Params
    params = rsa_default_params()

    # Writer
    writer = NPZShardWriter(outdir, base=args.tag, dt_ms=args.dt, decimate=args.decimate, logger=logger)

    # Run
    stats = run_network(
        T_ms=args.T,
        dt_ms=args.dt,
        params=params,
        N=args.N,
        VT_val=-56.2,
        V0=-65.0,
        jitter=3.0,
        ou=ou,
        Iext_callable=None,  # or provide a function(step)->(I_now, I_next)
        A_signed=A_signed,
        Vsyn_e=20.0, Vsyn_i=-80.0,
        sr_ms=0.5, sd_ms=8.0, Vth_rel=-20.0, krel=2.0,
        seed=args.seed,
        device=device,
        dtype=torch.float32,
        chunk_steps=args.chunk_steps,
        decimate=args.decimate,
        writer=writer,
        logger=logger,
    )

    logger.info(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
