from __future__ import annotations
import argparse, json, math, os, time, multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import importlib
import numpy as np
import torch

# ------------------------- helpers -------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def build_rho_grid(args) -> List[float]:
    if args.rhos:
        return parse_float_list(args.rhos)
    if args.rho_logspace:
        a, b, k = args.rho_logspace
        return list(np.geomspace(a, b, num=k))
    a, b, k = args.rho_linspace
    return list(np.linspace(a, b, num=k))

def make_run_tag(rho: float, rep: int) -> str:
    return f"rho{rho:.6f}_rep{rep:02d}"

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------- worker -------------------------

def run_one(job):
    """
    job = dict with all args already resolved, to make it easy for multiprocessing.
    """
    sim = importlib.import_module(job["sim_module"])

    # Unpack job config
    out_root = Path(job["out_root"])
    rho = float(job["rho"])
    rep = int(job["rep"])
    tag = make_run_tag(rho, rep)
    run_dir = out_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logger per run
    logger = sim.setup_logger(run_dir / f"{tag}.log")

    # Reuse provided connectivity
    A_signed = job["A_signed"]
    device = job["device"]
    dtype = getattr(torch, job["dtype"])
    N = job["N"]

    # Background drive from ratio
    Gbg = job["Gbg"]
    tau_e, tau_i = job["tau_e_ms"], job["tau_i_ms"]
    CVe, CVi = job["CVe"], job["CVi"]
    ge0, gi0, De, Di = sim.compute_bg_from_ratio(rho, Gbg, tau_e, tau_i, CVe=CVe, CVi=CVi)

    # OU instance
    ou = sim.OUSource(
        N=N,
        ge0=ge0, gi0=gi0,
        tau_e_ms=tau_e, tau_i_ms=tau_i,
        De=De, Di=Di,
        dt_ms=job["dt_ms"],
        seed=job["seed_run"],
        device=device,
        dtype=dtype,
    )

    # Writer
    writer = sim.NPZShardWriter(
        outdir=run_dir,
        base=tag,
        dt_ms=job["dt_ms"],
        decimate=job["decimate"],
        logger=logger,
        compress=False,   # keep it fast
    )

    # Run
    t0 = time.perf_counter()
    stats = sim.run_network(
        T_ms=job["T_ms"],
        dt_ms=job["dt_ms"],
        params=sim.rsa_default_params(),
        N=N,
        VT_val=job["VT_val"],
        V0=job["V0"],
        jitter=job["jitter"],
        ou=ou,
        Iext_callable=None,
        A_signed=A_signed,     # fixed graph for all runs
        Vsyn_e=job["Vsyn_e"], Vsyn_i=job["Vsyn_i"],
        sr_ms=job["sr_ms"], sd_ms=job["sd_ms"], Vth_rel=job["Vth_rel"], krel=job["krel"],
        seed=job["seed_run"],
        device=device,
        dtype=dtype,
        chunk_steps=job["chunk_steps"],
        decimate=job["decimate"],
        writer=writer,
        logger=logger,
    )
    t1 = time.perf_counter()

    # Meta
    meta = dict(
        tag=tag,
        rho=rho,
        Gbg=Gbg,
        CVe=CVe, CVi=CVi,
        tau_e_ms=tau_e, tau_i_ms=tau_i,
        dt_ms=job["dt_ms"],
        T_ms=job["T_ms"],
        N=N,
        chunk_steps=job["chunk_steps"],
        decimate=job["decimate"],
        VT_val=job["VT_val"],
        V0=job["V0"],
        jitter=job["jitter"],
        seeds=dict(connectivity=job["seed_net"], run=job["seed_run"]),
        timings=dict(sim_seconds=stats.get("sim_seconds", t1 - t0), wall_seconds=(t1 - t0)),
        writer=dict(compress=False),
        syn=dict(Vsyn_e=job["Vsyn_e"], Vsyn_i=job["Vsyn_i"], sr_ms=job["sr_ms"], sd_ms=job["sd_ms"],
                 Vth_rel=job["Vth_rel"], krel=job["krel"]),
        device=str(device),
        dtype=job["dtype"],
    )
    save_json(run_dir / "meta.json", meta)
    return tag, meta["timings"]["wall_seconds"]

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Sweep rho for HH network with OU background drive.")
    ap.add_argument("--sim-module", type=str, default="hhnet",
                    help="Python module name for your sim code (file name without .py).")
    # core sim
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--T", type=float, default=5000.0, help="simulation length (ms)")
    ap.add_argument("--dt", type=float, default=0.05, help="time step (ms)")
    ap.add_argument("--p_conn", type=float, default=0.10)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    # background OU
    ap.add_argument("--Gbg", type=float, default=0.069)
    ap.add_argument("--CVe", type=float, default=0.3)
    ap.add_argument("--CVi", type=float, default=0.3)
    ap.add_argument("--tau-e", dest="tau_e_ms", type=float, default=2.7)
    ap.add_argument("--tau-i", dest="tau_i_ms", type=float, default=10.5)
    # rho grid
    ap.add_argument("--rhos", type=str, default=None, help="Comma list, e.g. '0.6,0.8,1.0,1.2'")
    ap.add_argument("--rho-logspace", type=float, nargs=3, metavar=("START","STOP","NUM"),
                    default=None, help="Logspace in [START, STOP], NUM points")
    ap.add_argument("--rho-linspace", type=float, nargs=3, metavar=("START","STOP","NUM"),
                    default=(0.6, 1.6, 9), help="Linspace in [START, STOP], NUM points")
    # replicates & seeds
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=2025, help="base seed; graph uses this; runs add +rep")
    # IO
    ap.add_argument("--out-dir", type=str, default="sweeps/rho_sweep")
    ap.add_argument("--tag-prefix", type=str, default="run")
    ap.add_argument("--chunk-steps", type=int, default=48000)
    ap.add_argument("--decimate", type=int, default=1)
    # synapse params
    ap.add_argument("--Vsyn-e", dest="Vsyn_e", type=float, default=20.0)
    ap.add_argument("--Vsyn-i", dest="Vsyn_i", type=float, default=-80.0)
    ap.add_argument("--sr", dest="sr_ms", type=float, default=0.5)
    ap.add_argument("--sd", dest="sd_ms", type=float, default=8.0)
    ap.add_argument("--Vth-rel", dest="Vth_rel", type=float, default=-20.0)
    ap.add_argument("--krel", type=float, default=2.0)
    # ICs
    ap.add_argument("--VT", dest="VT_val", type=float, default=-56.2)
    ap.add_argument("--V0", type=float, default=-65.0)
    ap.add_argument("--jitter", type=float, default=3.0)
    # parallelism
    ap.add_argument("--procs", type=int, default=1, help="CPU processes (avoid >1 on CUDA)")
    args = ap.parse_args()

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and args.procs != 1:
        print("[warn] CUDA detected; forcing --procs=1 to avoid GPU contention.")
        args.procs = 1

    # Import sim module and prebuild connectivity once
    sim = importlib.import_module(args.sim_module)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Build and save graph once for fairness
    torch_device = device
    A_signed, Apos, Aneg, is_inh = sim.random_signed_adjacency_3to1(
        args.N, p_conn=args.p_conn, w_exc=(0.001, 0.05), w_inh=(0.01, 0.05),
        seed=args.seed0, device=torch_device, return_sparse=False
    )
    # Save connectivity for provenance (npy is fine)
    np.save(out_root / "A_signed.npy", A_signed.cpu().numpy())

    # Build rho list
    rhos = build_rho_grid(args)
    manifest = dict(
        sim_module=args.sim_module,
        N=args.N, dt_ms=args.dt, T_ms=args.T,
        p_conn=args.p_conn, dtype=args.dtype, device=device,
        Gbg=args.Gbg, CVe=args.CVe, CVi=args.CVi, tau_e_ms=args.tau_e_ms, tau_i_ms=args.tau_i_ms,
        rhos=rhos, replicates=args.replicates,
        seeds=dict(connectivity=args.seed0),
        chunk_steps=args.chunk_steps, decimate=args.decimate,
        syn=dict(Vsyn_e=args.Vsyn_e, Vsyn_i=args.Vsyn_i, sr_ms=args.sr_ms, sd_ms=args.sd_ms,
                 Vth_rel=args.Vth_rel, krel=args.krel),
        ICs=dict(VT_val=args.VT_val, V0=args.V0, jitter=args.jitter),
        out_dir=str(out_root),
    )
    save_json(out_root / "manifest.json", manifest)

    # Prepare jobs
    jobs = []
    for i_rho, rho in enumerate(rhos):
        for rep in range(args.replicates):
            seed_run = args.seed0 + 1000*i_rho + rep  # stable, distinct
            jobs.append(dict(
                sim_module=args.sim_module,
                out_root=str(out_root),
                rho=rho,
                rep=rep,
                Gbg=args.Gbg,
                CVe=args.CVe, CVi=args.CVi,
                tau_e_ms=args.tau_e_ms, tau_i_ms=args.tau_i_ms,
                dt_ms=args.dt, T_ms=args.T,
                N=args.N, chunk_steps=args.chunk_steps, decimate=args.decimate,
                VT_val=args.VT_val, V0=args.V0, jitter=args.jitter,
                Vsyn_e=args.Vsyn_e, Vsyn_i=args.Vsyn_i,
                sr_ms=args.sr_ms, sd_ms=args.sd_ms, Vth_rel=args.Vth_rel, krel=args.krel,
                device=device, dtype=args.dtype,
                seed_net=args.seed0, seed_run=seed_run,
                A_signed=A_signed,   # torch tensor gets pickled; fine for moderate N
            ))

    # Execute
    if args.procs == 1:
        results = [run_one(j) for j in jobs]
    else:
        # Spawn processes only for CPU; avoid CUDA
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.procs) as pool:
            results = pool.map(run_one, jobs)

    # Write quick summary
    summary = [{"tag": t, "wall_seconds": s} for (t, s) in results]
    save_json(out_root / "summary.json", summary)
    print(f"[ok] completed {len(summary)} runs into {out_root}")

if __name__ == "__main__":
    main()
