import os, glob, time, json, random, csv, multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except ImportError:
    HAVE_GUROBI = False

#Configuration

DATA_DIR       = "/Users/hengsichen/Downloads/ResourceAllocation-main/Data"
N_TIMES        = 15
N_COUNTIES     = 254
N_SCENARIOS    = 50
RANDOM_SEED    = 42
TOTAL_DOSES    = 1_000_000

PANDEMIC_CODES = {
    "1918": "c1", "1928": "c2", "1957": "c3",
    "1968": "c4", "2009": "c5", "2020": "c6",
}
BETA_G       = np.array([0.18, -0.03, 0.05, 0.03, 0.01])
GAMMA_WW     = -float(BETA_G.mean())
ALPHA_VALUES = [0.0, 0.10, 0.15, 0.20]

# PH penalty
BETA_RHO     = 0.05
RHO_GROWTH   = 1.05
RHO_MAX_MULT = 20.0
LAM_MAX_MULT = 1e4

# Dynamic MIP gap schedule
GAP_INIT  = 0.20
GAP_DECAY = 0.80
GAP_MIN   = 0.005

# Variable slamming thresholds
STD_TOL    = 0.15
MEAN_LOW   = 0.15
MEAN_HIGH  = 0.85
FIX_TARGET = 0.85

# Plateau early exit
PLATEAU_ITERS = 5
OBJ_TOL       = 1e-3

# Convergence
EPSILON_REL = 1e-4
MAX_ITER    = 300

N_WORKERS = None

#Avaulability Regimes
def make_At(regime: str) -> np.ndarray:
    At = np.zeros(N_TIMES)
    if regime == "I":
        At[0] = TOTAL_DOSES
    elif regime == "II":
        At[2] = TOTAL_DOSES
    elif regime == "III":
        At[0] = TOTAL_DOSES // 2
        At[2] = TOTAL_DOSES // 2
    else:
        raise ValueError(f"Unknown regime '{regime}'")
    return At

#Data Loading
def list_scenario_files(pandemic):
    code = PANDEMIC_CODES[pandemic]
    pat  = os.path.join(DATA_DIR, f"infectious-*_{code}_ic*_population_monthly.csv")
    pops = sorted(glob.glob(pat))
    if not pops:
        raise FileNotFoundError(f"No files for {pandemic}. Pattern: {pat}")
    out = []
    for pf in pops:
        bf = pf.replace("_population_monthly.csv", "_benefit_monthly.csv")
        if os.path.exists(bf):
            out.append((pf, bf))
    return out

def load_single(pop_path, ben_path):
    S_raw = pd.read_csv(pop_path, index_col=0).values.astype(float)
    B_raw = pd.read_csv(ben_path, index_col=0).values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        B_til = np.where(S_raw > 0, B_raw / S_raw, 0.0)
    return S_raw.T.copy(), B_til.T.copy()

def apply_alpha(U, Bt, alpha):
    if alpha == 0.0:
        return U.copy(), Bt.copy()
    return (1.0 + alpha) * U, (Bt + alpha * GAMMA_WW) / (1.0 + alpha)

def load_scenarios(pandemic, n=N_SCENARIOS, seed=RANDOM_SEED, alpha=0.0):
    triples = list_scenario_files(pandemic)
    rng     = random.Random(seed)
    sampled = rng.sample(triples, min(n, len(triples)))
    Ss, Bs  = [], []
    for pp, bp in sampled:
        U, Bt  = load_single(pp, bp)
        Sa, Ba = apply_alpha(U, Bt, alpha)
        Ss.append(Sa)
        Bs.append(Ba)
    return np.stack(Ss), np.stack(Bs)

#IRP Objective
def compute_obj(r, S, B, P):
    obj = 0.0
    for omega in range(S.shape[0]):
        q = np.zeros(N_COUNTIES)
        for t in range(N_TIMES):
            avail      = q + r[:, t]
            s_t, b_t   = S[omega, :, t], B[omega, :, t]
            with np.errstate(invalid="ignore", divide="ignore"):
                f_t = np.where(s_t > 0, np.minimum(1.0, avail / s_t), 1.0)
            obj += P[omega] * np.dot(b_t * s_t, f_t)
            q    = np.maximum(0.0, avail - f_t * s_t)
    return float(obj)

def compute_irp(S, B, At, P):
    r = np.zeros((N_COUNTIES, N_TIMES))
    for t in range(N_TIMES):
        if At[t] <= 0:
            continue
        exp_d  = (P[:, None] * S[:, :, t]).sum(0)
        tot    = exp_d.sum()
        r[:, t] = At[t] * exp_d / tot if tot > 1e-9 else At[t] / N_COUNTIES
    return r, compute_obj(r, S, B, P)

def project(r_bar, At):
    r = r_bar.copy()
    cum = 0.0
    for t in range(N_TIMES):
        allowed = max(0.0, At[:t + 1].sum() - cum)
        tot_t   = r[:, t].sum()
        if tot_t > allowed + 1e-9 and tot_t > 1e-12:
            r[:, t] *= allowed / tot_t
        r[:, t] = np.maximum(0.0, r[:, t])
        cum    += r[:, t].sum()
    return r

#Rho
def calibrate_rho(S, B, At, beta=BETA_RHO):
    n_omega  = S.shape[0]
    P_mean   = 1.0 / n_omega
    r_scale  = At.sum() / max(N_COUNTIES, 1)
    if r_scale < 1e-9:
        return 1e-15, 1e-14
    BS_active = (B * S)[S > 0]
    B_mean_S  = float(BS_active.mean()) if len(BS_active) > 0 else 1.0
    rho       = beta * 2.0 * P_mean * B_mean_S / (r_scale ** 2)
    rho_max   = RHO_MAX_MULT * rho
    return float(np.clip(rho, 1e-15, rho_max)), rho_max

#MIQP Subproblem
def _solve_one_scenario(args):
    (omega, S_omega, B_omega, At, lam_omega, r_bar, rho, P_omega,
     mip_gap, fixed_pairs, r_hint, x_hint) = args

    C, T = N_COUNTIES, N_TIMES
    M_t  = np.array([At[:t + 1].sum() for t in range(T)])

    lin_f       = P_omega * B_omega * S_omega
    coeff_r_lin = rho * r_bar - lam_omega

    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",   0)
        env.setParam("LogToConsole", 0)
        env.start()

        with gp.Model(env=env) as m:
            m.setParam("OutputFlag",   0)
            m.setParam("LogToConsole", 0)
            m.setParam("MIPGap",       mip_gap)

            r = m.addMVar((C, T), lb=0.0,           name="r")
            f = m.addMVar((C, T), lb=0.0, ub=1.0,  name="f")
            q = m.addMVar((C, T), lb=0.0,           name="q")
            x = m.addMVar((C, T), vtype=GRB.BINARY, name="x")

            if r_hint is not None:
                r.Start = r_hint
            if x_hint is not None:
                x.Start = x_hint

            for (c_fix, t_fix), val in fixed_pairs.items():
                m.addConstr(x[c_fix, t_fix] == float(val))

            for t in range(T):
                s_t = S_omega[:, t]
                if t == 0:
                    m.addConstr(q[:, t] == r[:, t] - f[:, t] * s_t)
                else:
                    m.addConstr(q[:, t] == q[:, t - 1] + r[:, t] - f[:, t] * s_t)

            for tau in range(T):
                m.addConstr(r[:, :tau + 1].sum() <= At[:tau + 1].sum())

            m.addConstr(x <= f)
            for t in range(T):
                if M_t[t] > 0:
                    m.addConstr(q[:, t] <= M_t[t] * x[:, t])
                else:
                    m.addConstr(q[:, t] == 0.0)

            m.setObjective(
                (lin_f * f).sum()
                + (coeff_r_lin * r).sum()
                - (rho / 2.0) * (r * r).sum(),
                GRB.MAXIMIZE
            )
            m.optimize()

            if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                result = (omega, r.X.copy(), x.X.copy())
            else:
                result = (omega,
                          _greedy(S_omega, B_omega, At, lam_omega, r_bar, rho, P_omega),
                          np.zeros((C, T)))

        env.dispose()
        return result

    except Exception as e:
        print(f"    [worker ω={omega} error: {e}]")
        return (omega,
                _greedy(S_omega, B_omega, At, lam_omega, r_bar, rho, P_omega),
                np.zeros((C, T)))


def _greedy(S_omega, B_omega, At, lam_omega, r_bar, rho, P_omega):
    C, T   = N_COUNTIES, N_TIMES
    r_out  = np.zeros((C, T))
    q_prev = np.zeros(C)
    cum    = 0.0
    for t in range(T):
        avail_t = min(At[t], At[:t + 1].sum() - cum)
        if avail_t < 1e-9:
            continue
        marg  = P_omega * B_omega[:, t] + rho * r_bar[:, t] - lam_omega[:, t]
        has_d = S_omega[:, t] > 0
        eff   = np.where(has_d, marg, -np.inf)
        rem   = float(avail_t)
        for c in np.argsort(-eff):
            if rem < 1e-9 or eff[c] <= 0:
                break
            alloc       = min(rem, max(0.0, S_omega[c, t] - q_prev[c]))
            r_out[c, t] += alloc
            rem         -= alloc
        if rem > 1e-9:
            pos = (eff > 0) & has_d
            if pos.any():
                w = eff[pos] / eff[pos].sum()
                r_out[pos, t] += rem * w
        cum  += r_out[:, t].sum()
        s_t   = S_omega[:, t]
        avail = q_prev + r_out[:, t]
        with np.errstate(invalid="ignore", divide="ignore"):
            f_t = np.where(s_t > 0, np.minimum(1.0, avail / s_t), 1.0)
        q_prev = np.maximum(0.0, avail - f_t * s_t)
    return r_out

#Parellel
def solve_all_scenarios_parallel(n_omega, S, B, At, lam, r_bar, rho, P,
                                  mip_gap, fixed_pairs,
                                  r_prev=None, x_prev=None,
                                  n_workers=N_WORKERS):
    C, T    = N_COUNTIES, N_TIMES
    r_omega = np.zeros((n_omega, C, T))
    x_omega = np.zeros((n_omega, C, T))

    jobs = []
    for omega in range(n_omega):
        r_hint = r_prev[omega] if r_prev is not None else None
        x_hint = x_prev[omega] if x_prev is not None else None
        jobs.append((
            omega,
            S[omega].copy(), B[omega].copy(), At.copy(),
            lam[omega].copy(), r_bar.copy(),
            rho, float(P[omega]),
            mip_gap, dict(fixed_pairs),
            r_hint, x_hint,
        ))

    actual_workers = n_workers if n_workers is not None else os.cpu_count()
    actual_workers = max(1, min(actual_workers, n_omega))

    if actual_workers == 1 or not HAVE_GUROBI:
        for job in jobs:
            omega, r_sol, x_sol = _solve_one_scenario(job)
            r_omega[omega] = r_sol
            x_omega[omega] = x_sol
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_solve_one_scenario, job): job[0]
                       for job in jobs}
            for fut in as_completed(futures):
                try:
                    omega, r_sol, x_sol = fut.result()
                    r_omega[omega] = r_sol
                    x_omega[omega] = x_sol
                except Exception as e:
                    omega = futures[fut]
                    print(f"    [future ω={omega} failed: {e}; using greedy]")
                    r_omega[omega] = _greedy(
                        S[omega], B[omega], At, lam[omega], r_bar, rho, P[omega])

    return r_omega, x_omega

#Slamming
def compute_slam(x_all, P, existing_fixed=None,
                 std_tol=STD_TOL, mean_low=MEAN_LOW, mean_high=MEAN_HIGH):
    if existing_fixed is None:
        existing_fixed = {}

    new_fixed = dict(existing_fixed)
    n_active = n_new = n_slam0 = n_slam1 = 0

    P_col = P[:, None, None]
    x_bar = (P_col * x_all).sum(axis=0)
    x_var = (P_col * (x_all - x_bar[None]) ** 2).sum(axis=0)
    x_std = np.sqrt(np.maximum(x_var, 0.0))

    for c in range(N_COUNTIES):
        for t in range(N_TIMES):
            if x_all[:, c, t].max() < 1e-6:
                continue
            n_active += 1
            if (c, t) in existing_fixed:
                continue
            mu, sig = float(x_bar[c, t]), float(x_std[c, t])
            if sig < std_tol:
                if mu < mean_low:
                    new_fixed[(c, t)] = 0; n_new += 1; n_slam0 += 1
                elif mu > mean_high:
                    new_fixed[(c, t)] = 1; n_new += 1; n_slam1 += 1

    fix_pct = len(new_fixed) / max(n_active, 1)
    diag    = dict(n_active=n_active, n_new=n_new, n_slam0=n_slam0, n_slam1=n_slam1)
    return new_fixed, fix_pct, n_active, n_new, diag

# Plateau Detector
class PlateauDetector:
    def __init__(self, patience=PLATEAU_ITERS, obj_tol=OBJ_TOL):
        self.patience  = patience
        self.obj_tol   = obj_tol
        self._counter  = 0
        self._prev_obj = None

    def step(self, obj_cur: float, n_new_fixed: int) -> bool:
        if self._prev_obj is None:
            self._prev_obj = obj_cur
            return False
        rel_change = abs(obj_cur - self._prev_obj) / max(abs(self._prev_obj), 1.0)
        obj_flat   = rel_change < self.obj_tol
        slam_flat  = n_new_fixed == 0
        if obj_flat and slam_flat:
            self._counter += 1
        else:
            self._counter = 0
        self._prev_obj = obj_cur
        return self._counter >= self.patience

# PH Main Algorithm
def ph_v7(pandemic, regime, S, B, At, P,
          beta_rho=BETA_RHO, rho_growth=RHO_GROWTH,
          rho_max_mult=RHO_MAX_MULT, lam_max_mult=LAM_MAX_MULT,
          gap_init=GAP_INIT, gap_decay=GAP_DECAY, gap_min=GAP_MIN,
          std_tol=STD_TOL, mean_low=MEAN_LOW, mean_high=MEAN_HIGH,
          fix_target=FIX_TARGET,
          epsilon_rel=EPSILON_REL, max_iter=MAX_ITER,
          plateau_iters=PLATEAU_ITERS, obj_tol=OBJ_TOL,
          n_workers=N_WORKERS,
          use_gurobi=True, verbose=True):

    t0 = time.perf_counter()
    n_omega, C, T = S.shape

    r_irp, obj_irp = compute_irp(S, B, At, P)
    rho, rho_max   = calibrate_rho(S, B, At, beta=beta_rho)
    lam_max        = lam_max_mult * rho
    rho_init       = rho

    lam         = np.zeros((n_omega, C, T))
    r_bar       = r_irp.copy()
    fixed_pairs = {}
    mip_gap     = gap_init
    plateau     = PlateauDetector(patience=plateau_iters, obj_tol=obj_tol)

    r_omega, x_omega = solve_all_scenarios_parallel(
        n_omega, S, B, At, lam, r_bar, rho, P,
        mip_gap, fixed_pairs,
        r_prev=None, x_prev=None,
        n_workers=n_workers,
    )
    r_bar = np.einsum("o,oct->ct", P, r_omega)

    conv_log    = []
    rel_res     = 1.0
    n_iter      = 0
    fix_pct     = 0.0
    stop_reason = f"max_iter ({max_iter})"

    if verbose:
        print(f"\n{'─'*84}")
        print(f"  PH — {pandemic} / Regime {regime}")
        print(f"  {'k':>5}  {'ρ':>9}  {'gap':>6}  {'Rel Res':>11}  "
              f"{'Obj':>13}  {'fixed%':>7}  {'new_fix':>7}  {'t':>6}")

    for k in range(1, max_iter + 1):
        r_bar_prev = r_bar.copy()

        for omega in range(n_omega):
            lam[omega] += rho * (r_omega[omega] - r_bar_prev)
            lam[omega]  = np.clip(lam[omega], -lam_max, lam_max)

        rho     = min(rho * rho_growth, rho_max)
        mip_gap = max(gap_init * (gap_decay ** k), gap_min)

        n_new = 0
        if use_gurobi and HAVE_GUROBI and x_omega.max() > 1e-6:
            fixed_pairs, fix_pct, _, n_new, _ = compute_slam(
                x_omega, P, fixed_pairs, std_tol, mean_low, mean_high)

        r_omega_new, x_omega_new = solve_all_scenarios_parallel(
            n_omega, S, B, At, lam, r_bar_prev, rho, P,
            mip_gap, fixed_pairs,
            r_prev=r_omega,
            x_prev=x_omega,
            n_workers=n_workers,
        )
        r_omega = r_omega_new
        x_omega = x_omega_new

        r_bar = np.einsum("o,oct->ct", P, r_omega)

        resids  = np.array([np.linalg.norm(r_omega[o] - r_bar)
                            for o in range(n_omega)])
        abs_res = float(resids.max())
        rel_res = abs_res / (np.linalg.norm(r_bar) + 1e-12)

        r_proj  = project(r_bar.copy(), At)
        obj_cur = compute_obj(r_proj, S, B, P)
        elapsed = time.perf_counter() - t0

        conv_log.append({
            "k": k, "rho": rho, "mip_gap": mip_gap,
            "abs_res": abs_res, "rel_res": rel_res,
            "obj": obj_cur, "fix_pct": fix_pct, "n_new_fixed": n_new,
        })
        n_iter = k

        if verbose:
            print(f"  {k:>5}  {rho:>9.3e}  {mip_gap:>5.1%}  "
                  f"{rel_res:>11.7f}  {obj_cur:>13.2f}  "
                  f"{fix_pct:>6.1%}  {n_new:>7d}  {elapsed:>5.1f}s")

        if rel_res < epsilon_rel:
            stop_reason = f"PH converged (rel_res={rel_res:.2e})"
            break
        if fix_pct >= fix_target:
            stop_reason = f"slamming {fix_pct:.1%} >= {fix_target:.0%}"
            break
        if plateau.step(obj_cur, n_new):
            stop_reason = f"plateau: {plateau_iters} iters + no new slams"
            break

    r_star  = project(r_bar, At)
    obj_PH  = compute_obj(r_star, S, B, P)
    gain    = (obj_PH - obj_irp) / max(abs(obj_irp), 1e-12) * 100
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  Stop: {stop_reason}")
        print(f"  obj={obj_PH:.2f}  IRP={obj_irp:.2f}  "
              f"gain={gain:.2f}%  slammed={fix_pct:.1%}  "
              f"iters={n_iter}  time={elapsed:.1f}s")
        print(f"{'─'*84}")

    return {
        "r_star":          r_star,
        "obj_PH":          float(obj_PH),
        "obj_IRP":         float(obj_irp),
        "gain_pct":        float(gain),
        "n_iter":          n_iter,
        "elapsed":         float(elapsed),
        "stop_reason":     stop_reason,
        "rho_init":        float(rho_init),
        "rho_final":       float(rho),
        "fix_pct_final":   float(fix_pct),
        "r_total_by_time": r_star.sum(axis=0).tolist(),
        "convergence":     conv_log,
    }

# Main

def run_all(data_dir=DATA_DIR, n_scenarios=N_SCENARIOS, seed=RANDOM_SEED,
            regimes=("I", "II", "III"), alphas=ALPHA_VALUES,
            use_gurobi=True, save_results=True, output_dir=".",
            n_workers=N_WORKERS):

    global DATA_DIR
    DATA_DIR = data_dir

    paper_ref = {
        "I":  {"SRP": {"1918":27619.4,"1928":27635.1,"1957":27650.9,
                        "1968":27680.2,"2009":27693.6,"2020":27601.2},
               "IRP": {"1918":24175.3,"1928":24100.5,"1957":23886.0,
                        "1968":24207.6,"2009":24235.9,"2020":24126.8}},
        "II": {"SRP": {"1918":27376.8,"1928":27383.0,"1957":27416.4,
                        "1968":27392.1,"2009":27418.7,"2020":27367.5},
               "IRP": {"1918":26226.2,"1928":26207.0,"1957":26245.0,
                        "1968":26256.1,"2009":26260.6,"2020":26186.3}},
        "III":{"SRP": {"1918":27640.0,"1928":27663.7,"1957":27650.8,
                        "1968":27679.9,"2009":27702.6,"2020":27595.4},
               "IRP": {"1918":25189.9,"1928":20091.0,"1957":25271.7,
                        "1968":23637.3,"2009":25123.4,"2020":20069.7}},
    }

    P = np.ones(n_scenarios) / n_scenarios

    all_results = {}

    for regime in regimes:
        At = make_At(regime)
        for alpha in alphas:
            print(f"\n{'━'*84}")
            print(f"  REGIME {regime}  |  α = {alpha}")
            print(f"{'━'*84}")

            for pandemic in PANDEMIC_CODES:
                try:
                    S, B = load_scenarios(pandemic, n_scenarios, seed, alpha)
                except FileNotFoundError as e:
                    print(f"  [SKIP {pandemic}]: {e}")
                    continue

                res = ph_v7(pandemic, regime, S, B, At, P,
                            n_workers=n_workers,
                            use_gurobi=use_gurobi, verbose=True)

                key = (pandemic, regime, alpha)
                all_results[key] = {
                    "pandemic":       pandemic,
                    "regime":         regime,
                    "alpha":          alpha,
                    "obj_PH":         res["obj_PH"],
                    "obj_IRP":        res["obj_IRP"],
                    "gain_pct":       res["gain_pct"],
                    "paper_SRP":      paper_ref[regime]["SRP"].get(pandemic),
                    "paper_IRP":      paper_ref[regime]["IRP"].get(pandemic),
                    "n_iterations":   res["n_iter"],
                    "solve_time_sec": res["elapsed"],
                    "stop_reason":    res["stop_reason"],
                    "rho_init":       res["rho_init"],
                    "rho_final":      res["rho_final"],
                    "fix_pct":        res["fix_pct_final"],
                    "r_total_by_time":res["r_total_by_time"],
                    "convergence":    res["convergence"],
                }

    _print_summary(all_results, regimes, alphas)
    if save_results:
        _save(all_results, regimes, alphas, output_dir)
    return all_results


def _print_summary(all_results, regimes, alphas):
    W = 96
    for regime in regimes:
        for alpha in alphas:
            print(f"\n{'='*W}")
            print(f"  PH — Regime {regime}  |  α = {alpha}")
            print(f"{'='*W}")
            print(f"  {'Pandemic':<10} {'PH':>10} {'IRP':>10} {'Gain%':>7} "
                  f"{'SRP_paper':>10} {'k':>5} {'fixed%':>7} {'time':>7}  Stop reason")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*10} "
                  f"{'-'*5} {'-'*7} {'-'*7}  {'-'*20}")
            for pandemic in PANDEMIC_CODES:
                key = (pandemic, regime, alpha)
                if key not in all_results:
                    continue
                res  = all_results[key]
                pSRP = res.get("paper_SRP", "")
                sr   = res.get("stop_reason", "")[:22]
                print(f"  {pandemic:<10} {res['obj_PH']:>10.1f} {res['obj_IRP']:>10.1f} "
                      f"{res['gain_pct']:>7.1f} {str(pSRP):>10} "
                      f"{res['n_iterations']:>5} {res['fix_pct']:>6.1%} "
                      f"{res['solve_time_sec']:>6.1f}s  {sr}")

    print(f"\n{'='*W}")
    print("  CROSS-α: PH Objective")
    print(f"{'='*W}")
    for regime in regimes:
        print(f"\n  Regime {regime}")
        hdr = f"  {'Pandemic':<10}" + "".join(f"  {'α='+str(a):>11}" for a in alphas)
        print(hdr)
        print(f"  {'-'*10}" + "".join(f"  {'-'*11}" for _ in alphas))
        for pandemic in PANDEMIC_CODES:
            row = f"  {pandemic:<10}"
            for alpha in alphas:
                key = (pandemic, regime, alpha)
                v   = all_results[key]["obj_PH"] if key in all_results else float("nan")
                row += f"  {v:>11.1f}"
            print(row)
    print(f"{'='*W}")


def _save(all_results, regimes, alphas, output_dir):
    safe = {}
    for (p, r, a), v in all_results.items():
        safe[f"{p}__R{r}__a{int(a*100)}"] = {
            kk: vv for kk, vv in v.items() if kk != "convergence"
        }
    path = os.path.join(output_dir, "ph_results.json")
    with open(path, "w") as f:
        json.dump(safe, f, indent=2, default=str)

    path2 = os.path.join(output_dir, "ph_alpha_table.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "pandemic", "alpha", "obj_PH", "obj_IRP",
                    "gain_pct", "n_iter", "fix_pct", "solve_time",
                    "stop_reason", "paper_SRP", "paper_IRP"])
        for regime in regimes:
            for pandemic in PANDEMIC_CODES:
                for alpha in alphas:
                    key = (pandemic, regime, alpha)
                    if key not in all_results:
                        continue
                    res = all_results[key]
                    w.writerow([
                        regime, pandemic, alpha,
                        f"{res['obj_PH']:.2f}", f"{res['obj_IRP']:.2f}",
                        f"{res['gain_pct']:.2f}", res["n_iterations"],
                        f"{res['fix_pct']:.3f}", f"{res['solve_time_sec']:.2f}",
                        res.get("stop_reason", ""),
                        res.get("paper_SRP", ""),
                        res.get("paper_IRP", ""),
                    ])
    print(f"  Results saved to {output_dir}")

# Entry Point
if __name__ == "__main__":
    results = run_all(
        data_dir    = "/Users/hengsichen/Downloads/ResourceAllocation-main/Data",
        n_scenarios = 50,
        seed        = 42,
        regimes     = ("I", "II", "III"),
        alphas      = [0.0, 0.10, 0.15, 0.20],
        use_gurobi  = True,
        save_results= True,
        output_dir  = ".",
        n_workers   = None,
    )import os, glob, time, json, random, csv, multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except ImportError:
    HAVE_GUROBI = False
