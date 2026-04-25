"""
Adversarial Robust Optimisation — Heuristic C&CG  (CLEAN RESTART VERSION)
=========================================================
新增控制开关：
  FORCE_RESTART = True   → 忽略/删除旧checkpoint，从头开始
  FORCE_RESTART = False  → 如有旧checkpoint则续跑（原有行为）

自动续存：每完成一个instance立即写入checkpoint，中途中断可续跑。

Framework (unchanged from paper):
  SUB-PROBLEM  : heuristic attacker — pool scan over |Omega|=50 SEIR
                 scenarios, O(|Omega|*|C|*|T|) via Theorem 1, no solver.
  MASTER MILP  : exact MILP with binary x_{c,t}^k (NOT relaxed),
                 per equations (mp_obj)-(mp_vars).
  C&CG LOOP    : heuristic — attacker proposes worst pool scenario,
                 master optimises against accumulated set W.

Bug fix vs original:
  - in_W updated correctly on replacement path
"""

import os, glob, time, json, csv, random
import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
    print("Gurobi found.")
except ImportError:
    HAVE_GUROBI = False
    print("Gurobi not available.")

# =============================================================================
# 0. RESTART CONTROL  ← 改这里
# =============================================================================

FORCE_RESTART = True   # True = 从头跑，忽略一切旧checkpoint
                       # False = 有checkpoint就续跑

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_DIR    = "/Users/hengsichen/Downloads/ResourceAllocation-main/Data"
N_TIMES     = 15
N_COUNTIES  = 254
N_SCENARIOS = 50
RANDOM_SEED = 42
TOTAL_DOSES = 1_000_000

PANDEMIC_CODES = {
    "1918": "c1", "1928": "c2", "1957": "c3",
    "1968": "c4", "2009": "c5", "2020": "c6",
}
BETA_G   = np.array([0.18, -0.03, 0.05, 0.03, 0.01])
GAMMA_WW = -float(BETA_G.mean())
ALPHA_VALUES = [0.0, 0.10, 0.15, 0.20]
RHO = 0.30

# C&CG parameters
K_MAX       = 10
K_INIT      = 3
K_SCENE_CAP = 15
N_POOL_ADD  = 3

TAU_0    = 300.0
GAMMA_TL = 1.25
TAU_MAX  = 600.0

EPS_TOTAL = 0.01

PAPER_REF = {
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

# =============================================================================
# 2. DATA LOADING
# =============================================================================

def make_At(regime: str) -> np.ndarray:
    At = np.zeros(N_TIMES)
    if   regime == "I":   At[0]  = TOTAL_DOSES
    elif regime == "II":  At[2]  = TOTAL_DOSES
    elif regime == "III": At[0]  = TOTAL_DOSES // 2; At[2] = TOTAL_DOSES // 2
    else: raise ValueError(f"Unknown regime '{regime}'")
    return At


def list_scenario_files(pandemic: str) -> list:
    code = PANDEMIC_CODES[pandemic]
    pat  = os.path.join(DATA_DIR,
                        f"infectious-*_{code}_ic*_population_monthly.csv")
    pops = sorted(glob.glob(pat))
    if not pops:
        raise FileNotFoundError(f"No files for pandemic {pandemic}.")
    pairs = []
    for pf in pops:
        bf = pf.replace("_population_monthly.csv", "_benefit_monthly.csv")
        if os.path.exists(bf):
            pairs.append((pf, bf))
    return pairs


def load_single(pop_path: str, ben_path: str):
    S_raw = pd.read_csv(pop_path, index_col=0).values.astype(float)
    B_raw = pd.read_csv(ben_path, index_col=0).values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        B_til = np.where(S_raw > 0, B_raw / S_raw, 0.0)
    return S_raw.T.copy(), B_til.T.copy()   # (C, T)


def apply_alpha(U, Bt, alpha):
    if alpha == 0.0:
        return U.copy(), Bt.copy()
    return (1.0 + alpha) * U, (Bt + alpha * GAMMA_WW) / (1.0 + alpha)


def load_scenarios(pandemic, n=N_SCENARIOS, seed=RANDOM_SEED, alpha=0.0):
    pairs   = list_scenario_files(pandemic)
    rng     = random.Random(seed)
    sampled = rng.sample(pairs, min(n, len(pairs)))
    Ss, Bs  = [], []
    for pp, bp in sampled:
        U, Bt = load_single(pp, bp)
        Sa, Ba = apply_alpha(U, Bt, alpha)
        Ss.append(Sa); Bs.append(Ba)
    return np.stack(Ss), np.stack(Bs)

# =============================================================================
# 3. UNCERTAINTY SET
# =============================================================================

def build_uncertainty_set(S_all, B_all, rho=RHO):
    S_lb = S_all.min(0); S_ub = S_all.max(0)
    B_lb = B_all.min(0); B_ub = B_all.max(0)
    S_rng = np.where(S_ub > S_lb, S_ub - S_lb, 1.0)
    B_rng = np.where(B_ub > B_lb, B_ub - B_lb, 1.0)
    return dict(S_lb=S_lb, S_ub=S_ub, S_hat=(S_lb+S_ub)/2, S_rng=S_rng,
                B_lb=B_lb, B_ub=B_ub, B_hat=(B_lb+B_ub)/2, B_rng=B_rng,
                Gamma=rho * N_COUNTIES * N_TIMES)

# =============================================================================
# 4. ANALYTICAL EVALUATION  (Theorem 1)
# =============================================================================

def eval_phi(r, S, B):
    """O(|C|*|T|) forward pass — no solver."""
    obj = 0.0
    q   = np.zeros(N_COUNTIES)
    for t in range(N_TIMES):
        avail = q + r[:, t]
        s_t   = S[:, t]; b_t = B[:, t]
        with np.errstate(invalid="ignore", divide="ignore"):
            f_t = np.where(s_t > 0, np.minimum(1.0, avail / s_t), 1.0)
        obj += float(np.dot(b_t * s_t, f_t))
        q    = np.maximum(0.0, avail - f_t * s_t)
    return obj


def eval_pool(r, S_all, B_all):
    return np.array([eval_phi(r, S_all[i], B_all[i])
                     for i in range(len(S_all))])

# =============================================================================
# 5. IRP WARM-START
# =============================================================================

def compute_irp(S_all, B_all, At):
    r = np.zeros((N_COUNTIES, N_TIMES))
    for t in range(N_TIMES):
        if At[t] <= 0:
            continue
        exp_d = S_all[:, :, t].mean(0)
        tot   = exp_d.sum()
        r[:, t] = At[t] * exp_d / tot if tot > 1e-9 \
                  else np.full(N_COUNTIES, At[t] / N_COUNTIES)
    vals = eval_pool(r, S_all, B_all)
    return r, float(vals.mean()), vals

# =============================================================================
# 6. POOL SCAN ATTACKER
# =============================================================================

def pool_attack_topn(r, S_all, B_all, excluded, UB, n_return=N_POOL_ADD):
    vals  = eval_pool(r, S_all, B_all)
    order = np.argsort(vals)
    out   = []
    for idx in order:
        i = int(idx)
        if i in excluded:
            continue
        if vals[i] >= UB:
            break
        out.append((i, S_all[i], B_all[i], float(vals[i])))
        if len(out) >= n_return:
            break
    return out

# =============================================================================
# 7. MASTER MILP
# =============================================================================

class MasterMILP:
    """
    Persistent Gurobi MILP master with scenario replacement.
    Bug fix: in_W set is managed externally; replacement returns old_idx
    so the caller can update in_W correctly.
    """

    _gid = 0

    def __init__(self, At, r_init):
        self.At  = At
        self.M_t = np.cumsum(At)
        # Each slot stores the pool index it came from (for in_W bookkeeping)
        self._slots     = []   # list of slot dicts
        self._slot_pidx = []   # parallel list: pool index for each slot

        self._env = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 0)
        self._env.start()
        self._m = gp.Model(env=self._env)
        self._m.setParam("OutputFlag",   0)
        self._m.setParam("LogToConsole", 0)

        self._r   = self._m.addMVar((N_COUNTIES, N_TIMES), lb=0.0, name="r")
        self._eta = self._m.addVar(lb=-1e9, ub=1e9, name="eta")

        for tau in range(N_TIMES):
            self._m.addConstr(
                self._r[:, :tau+1].sum() <= float(At[:tau+1].sum()),
                name=f"bud_{tau}"
            )
        self._m.setObjective(self._eta, GRB.MAXIMIZE)

        for c in range(N_COUNTIES):
            for t in range(N_TIMES):
                self._r[c, t].Start = float(r_init[c, t])
        self._m.update()

    # ------------------------------------------------------------------
    def _build_slot(self, S_k, B_k):
        MasterMILP._gid += 1
        sid = MasterMILP._gid
        pfx = f"g{sid}"
        C, T, M_t = N_COUNTIES, N_TIMES, self.M_t

        f_k = self._m.addMVar((C, T), lb=0.0, ub=1.0,  name=f"f_{pfx}")
        q_k = self._m.addMVar((C, T), lb=0.0,           name=f"q_{pfx}")
        x_k = self._m.addMVar((C, T), vtype=GRB.BINARY, name=f"x_{pfx}")

        inv_cs  = []
        l2_cs   = []
        zero_cs = []

        for t in range(T):
            lhs = q_k[:, t-1] if t > 0 else 0.0
            c_inv = self._m.addConstr(
                q_k[:, t] == lhs + self._r[:, t] - f_k[:, t] * S_k[:, t],
                name=f"inv_{pfx}_{t}"
            )
            inv_cs.append(c_inv)

        c_l1 = self._m.addConstr(x_k <= f_k, name=f"l1_{pfx}")
        for t in range(T):
            if M_t[t] > 0:
                l2_cs.append(self._m.addConstr(
                    q_k[:, t] <= float(M_t[t]) * x_k[:, t],
                    name=f"l2_{pfx}_{t}"
                ))

        for t in range(T):
            zm = (S_k[:, t] == 0)
            if zm.any():
                idx = np.where(zm)[0]
                zero_cs.append(self._m.addConstr(f_k[idx, t] == 1.0))
                zero_cs.append(self._m.addConstr(x_k[idx, t] == 1.0))

        c_eta = self._m.addConstr(
            self._eta <= (B_k * S_k * f_k).sum(),
            name=f"eta_{pfx}"
        )
        self._m.update()

        return dict(sid=sid, f=f_k, q=q_k, x=x_k,
                    inv=inv_cs, l1=c_l1, l2=l2_cs, zeros=zero_cs,
                    eta=c_eta, S=S_k.copy(), B=B_k.copy())

    # ------------------------------------------------------------------
    def _remove_slot(self, slot):
        m = self._m
        for c in slot["inv"] + [slot["l1"]] + slot["l2"] + slot["zeros"] \
                + [slot["eta"]]:
            m.remove(c)
        for mvar in (slot["f"], slot["q"], slot["x"]):
            rows = mvar.tolist()
            flat = [v for row in rows for v in row] if rows and isinstance(rows[0], list) else rows
            for v in flat:
                m.remove(v)
        m.update()

    # ------------------------------------------------------------------
    def add_scenario(self, pool_idx, S_k, B_k):
        self._slots.append(self._build_slot(S_k, B_k))
        self._slot_pidx.append(pool_idx)

    # ------------------------------------------------------------------
    def try_replace_weakest(self, r_cur, pool_idx_new, S_new, B_new, new_val):
        """
        Replace weakest slot if new_val is more adversarial.
        Returns (replaced: bool, evicted_pool_idx: int or None).
        The caller must update in_W accordingly.
        """
        phis    = np.array([eval_phi(r_cur, s["S"], s["B"]) for s in self._slots])
        pos     = int(np.argmax(phis))
        old_val = float(phis[pos])
        if new_val >= old_val:
            return False, None
        evicted_pidx = self._slot_pidx[pos]
        self._remove_slot(self._slots[pos])
        self._slots[pos]     = self._build_slot(S_new, B_new)
        self._slot_pidx[pos] = pool_idx_new
        return True, evicted_pidx

    # ------------------------------------------------------------------
    def phi_over_W(self, r):
        return np.array([eval_phi(r, s["S"], s["B"]) for s in self._slots])

    # ------------------------------------------------------------------
    def solve(self, tau_k, eps_m):
        self._m.setParam("TimeLimit", tau_k)
        self._m.setParam("MIPGap",    eps_m)
        self._m.optimize()
        ok = self._m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
        if ok:
            r_opt = self._r.X.copy()
            for c in range(N_COUNTIES):
                for t in range(N_TIMES):
                    self._r[c, t].Start = float(r_opt[c, t])
            return r_opt, float(self._m.ObjVal), float(self._m.ObjBound), True
        return None, None, None, False

    def n_scenarios(self):
        return len(self._slots)

    def dispose(self):
        self._m.dispose()
        self._env.dispose()

# =============================================================================
# 8. MAIN C&CG LOOP
# =============================================================================

def solve_aro_hccg(pandemic, regime, S_all, B_all, At,
                   alpha=0.0, rho=RHO,
                   k_max=K_MAX, k_init=K_INIT, k_cap=K_SCENE_CAP,
                   n_pool_add=N_POOL_ADD,
                   tau_0=TAU_0, gamma_tl=GAMMA_TL, tau_max=TAU_MAX,
                   eps=EPS_TOTAL,
                   verbose=True):
    t_wall = time.perf_counter()
    if not HAVE_GUROBI:
        return {"error": "Gurobi required"}

    r_irp, obj_irp, irp_vals = compute_irp(S_all, B_all, At)
    sep = "─" * 100

    if verbose:
        print(f"\n{sep}")
        print(f"  ARO H-CCG — {pandemic} / Regime {regime} / alpha={alpha}")
        print(f"  |C|={N_COUNTIES} |T|={N_TIMES} |Omega|={len(S_all)} "
              f"rho={rho} k_cap={k_cap} eps={eps*100:.1f}%")
        print(f"  IRP expected={obj_irp:.1f}  IRP worst={irp_vals.min():.1f}")
        print(f"  {'k':>3}  {'UB':>11}  {'LB':>11}  {'gap%':>7}  "
              f"{'added':>6}  {'repl':>5}  {'|W|':>4}  {'tau_k':>6}  {'t_MP':>7}")
        print(f"  {sep[2:]}")

    # ── Initialise ────────────────────────────────────────────────────────────
    mp   = MasterMILP(At, r_irp)
    in_W = set()   # pool indices currently represented in W

    sorted_init = np.argsort(irp_vals)
    for i in range(min(k_init, len(S_all))):
        idx = int(sorted_init[i])
        mp.add_scenario(idx, S_all[idx], B_all[idx])
        in_W.add(idx)

    LB = float(mp.phi_over_W(r_irp).min())
    UB = float("inf")
    r_k = r_irp.copy()
    log = []

    # ── C&CG iterations ───────────────────────────────────────────────────────
    for k in range(1, k_max + 1):

        # [1] Solve master MILP
        tau_k = min(tau_0 * (gamma_tl ** k), tau_max)
        t0    = time.perf_counter()
        r_new, obj_val, best_bd, ok = mp.solve(tau_k, eps)
        t_mp  = time.perf_counter() - t0

        if ok:
            r_k = r_new
            UB  = best_bd

        # [2] Re-evaluate LB from W under current r_k
        if mp.n_scenarios() > 0:
            w_phis = mp.phi_over_W(r_k)
            LB     = float(w_phis.min())

        gap = (UB - LB) / abs(LB) if (np.isfinite(UB) and abs(LB) > 1e-9) \
              else float("inf")

        # [3] Pool scan (threshold = UB)
        candidates = pool_attack_topn(r_k, S_all, B_all, in_W, UB,
                                      n_return=n_pool_add)

        n_added = 0
        n_repl  = 0
        for (p_idx, p_S, p_B, p_val) in candidates:
            if mp.n_scenarios() < k_cap:
                mp.add_scenario(p_idx, p_S, p_B)
                in_W.add(p_idx)
                n_added += 1
            else:
                # ── BUG FIX: update in_W on replacement ──────────────────────
                replaced, evicted_pidx = mp.try_replace_weakest(
                    r_k, p_idx, p_S, p_B, p_val)
                if replaced:
                    n_repl += 1
                    if evicted_pidx is not None:
                        in_W.discard(evicted_pidx)   # allow evicted back in pool
                    in_W.add(p_idx)                  # mark new as in W
            LB = min(LB, p_val)

        # Recompute gap after LB update
        gap = (UB - LB) / abs(LB) if (np.isfinite(UB) and abs(LB) > 1e-9) \
              else float("inf")

        if verbose:
            ub_s  = f"{UB:11.1f}" if np.isfinite(UB) else "        inf"
            gap_s = f"{gap*100:7.3f}" if np.isfinite(gap) else "    inf"
            print(f"  {k:>3}  {ub_s}  {LB:11.1f}  {gap_s}  "
                  f"{n_added:>6}  {n_repl:>5}  {mp.n_scenarios():>4}"
                  f"  {tau_k:6.1f}  {t_mp:6.1f}s")

        log.append({"k": k, "UB": float(UB) if np.isfinite(UB) else None,
                    "LB": float(LB), "gap_pct": float(gap*100) if np.isfinite(gap) else None,
                    "n_added": n_added, "n_repl": n_repl, "n_W": mp.n_scenarios(),
                    "tau_k": tau_k, "t_mp": t_mp,
                    "elapsed": time.perf_counter() - t_wall})

        # ── Termination ───────────────────────────────────────────────────────
        if np.isfinite(gap) and gap <= eps:
            if verbose:
                print(f"  ✓ Converged at k={k}: gap={gap*100:.3f}%")
            break

        if len(candidates) == 0:
            if verbose:
                print(f"  ✓ Pool exhausted at k={k} (no Phi < UB={UB:.1f} outside W)")
            break

    else:
        if verbose:
            print(f"  ✗ Iteration limit ({k_max}) reached.")

    mp.dispose()

    # ── Final evaluation ──────────────────────────────────────────────────────
    vals_final   = eval_pool(r_k, S_all, B_all)
    obj_expected = float(vals_final.mean())
    obj_wc50     = float(vals_final.min())
    gain_exp     = (obj_expected - obj_irp) / max(abs(obj_irp), 1e-12) * 100
    elapsed      = time.perf_counter() - t_wall

    if verbose:
        print(f"\n  Expected (50 scenarios) : {obj_expected:.1f}")
        print(f"  Worst-case (pool min)   : {obj_wc50:.1f}")
        print(f"  Robust LB (W min)       : {LB:.1f}")
        print(f"  UB (master bound)       : {UB:.1f}")
        print(f"  Final gap               : {(UB-LB)/abs(LB)*100:.2f}%")
        print(f"  IRP benchmark           : {obj_irp:.1f}")
        print(f"  Gain vs IRP             : {gain_exp:.2f}%")
        print(f"  Total time              : {elapsed:.1f}s")
        print(sep)

    return {"pandemic": pandemic, "regime": regime, "alpha": alpha, "rho": rho,
            "obj_ARO_expected": obj_expected, "obj_ARO_worst_50": obj_wc50,
            "obj_robust_LB": float(LB),
            "obj_UB": float(UB) if np.isfinite(UB) else None,
            "obj_IRP": obj_irp, "gain_expected_pct": gain_exp,
            "n_iter": k, "n_scenes_final": mp.n_scenarios(),
            "solve_time_sec": elapsed,
            "r_total_by_time": r_k.sum(axis=0).tolist(),
            "convergence": log}

# =============================================================================
# 9. CHECKPOINT HELPERS
# =============================================================================

CACHE_FILE = "aro_hccg_checkpoint.json"

def _cache_key(pandemic, regime, alpha):
    return f"{pandemic}__R{regime}__a{int(round(alpha * 100))}"


def _load_cache(out_dir, force_restart):
    """
    Load checkpoint.
    If force_restart=True, delete any existing checkpoint and return empty dict.
    """
    path = os.path.join(out_dir, CACHE_FILE)

    if force_restart:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [FORCE RESTART] Deleted old checkpoint: {path}")
        else:
            print(f"  [FORCE RESTART] No checkpoint found, starting fresh.")
        return {}

    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        cache = {}
        for k, v in raw.items():
            parts    = k.split("__")
            pandemic = parts[0]
            regime   = parts[1][1:]
            alpha    = int(parts[2][1:]) / 100
            cache[(pandemic, regime, alpha)] = v
        print(f"  [RESUME] Loaded {len(cache)} completed results from {path}")
        return cache
    return {}


def _save_cache(cache_dict, out_dir):
    """Atomic write of checkpoint."""
    path = os.path.join(out_dir, CACHE_FILE)
    flat = {}
    for (p, reg, a), v in cache_dict.items():
        key = _cache_key(p, reg, a)
        flat[key] = {kk: vv for kk, vv in v.items() if kk != "convergence"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    os.replace(tmp, path)

# =============================================================================
# 10. EXPERIMENT RUNNER
# =============================================================================

def run_all(data_dir=DATA_DIR, regimes=("I","II","III"), alphas=ALPHA_VALUES,
            rho=RHO, k_max=K_MAX, save=True, out_dir=".",
            force_restart=FORCE_RESTART):

    global DATA_DIR
    DATA_DIR = data_dir
    os.makedirs(out_dir, exist_ok=True)

    print("="*80)
    print("  ARO H-CCG | exact MILP master · heuristic pool attacker")
    print(f"  FORCE_RESTART={force_restart}")
    print(f"  rho={rho} regimes={regimes} alphas={alphas} K_max={k_max}")
    print("="*80)

    results = _load_cache(out_dir, force_restart)

    total = len(regimes) * len(alphas) * len(PANDEMIC_CODES)
    done  = sum(1 for (p, r, a) in results if r in regimes and a in alphas)
    todo  = total - done
    print(f"  Progress: {done}/{total} done, {todo} remaining\n")

    for regime in regimes:
        At = make_At(regime)
        for alpha in alphas:
            print(f"\n{'━'*80}\n  REGIME {regime}  |  alpha={alpha}\n{'━'*80}")
            for pandemic in PANDEMIC_CODES:

                if (pandemic, regime, alpha) in results:
                    prev = results[(pandemic, regime, alpha)]
                    ub   = prev.get("obj_UB") or 0
                    lb   = prev.get("obj_robust_LB", 1)
                    gap_s = f"{(ub-lb)/abs(lb)*100:.2f}%" if (ub and lb) else "n/a"
                    print(f"  [SKIP — already done] {pandemic}  "
                          f"ARO_exp={prev['obj_ARO_expected']:.1f}  "
                          f"gap={gap_s}  time={prev['solve_time_sec']:.0f}s")
                    continue

                try:
                    S_all, B_all = load_scenarios(pandemic, N_SCENARIOS,
                                                   RANDOM_SEED, alpha)
                except FileNotFoundError as e:
                    print(f"  [SKIP {pandemic}]: {e}"); continue

                res = solve_aro_hccg(pandemic, regime, S_all, B_all, At,
                                     alpha=alpha, rho=rho, k_max=k_max,
                                     verbose=True)

                if "error" not in res:
                    res["paper_SRP"] = PAPER_REF[regime]["SRP"].get(pandemic)
                    res["paper_IRP"] = PAPER_REF[regime]["IRP"].get(pandemic)
                    results[(pandemic, regime, alpha)] = res
                    _save_cache(results, out_dir)
                    done += 1
                    print(f"  [CHECKPOINT] {done}/{total} saved → "
                          f"{os.path.join(out_dir, CACHE_FILE)}")

    _print_summary(results, regimes, alphas)
    if save:
        _save_results(results, regimes, alphas, out_dir)
    return results


# =============================================================================
# 11. REPORTING
# =============================================================================

def _print_summary(results, regimes, alphas):
    W = 115
    for regime in regimes:
        for alpha in alphas:
            print(f"\n{'='*W}\n  Regime {regime}  |  alpha={alpha}\n{'='*W}")
            print(f"  {'Pandemic':<10} {'ARO(exp)':>10} {'ARO(wc50)':>10} "
                  f"{'LB':>10} {'UB':>10} {'gap%':>6} {'IRP':>10} "
                  f"{'SRP_paper':>10} {'Gain%':>7} {'k':>4} {'Time':>8}")
            print("  " + "-"*(W-4))
            for pandemic in PANDEMIC_CODES:
                key = (pandemic, regime, alpha)
                if key not in results or "error" in results[key]:
                    continue
                r   = results[key]
                srp = r.get("paper_SRP", "N/A")
                ub  = r["obj_UB"]
                lb  = r["obj_robust_LB"]
                ub_s  = f"{ub:10.1f}" if ub else "       inf"
                gap_s = f"{(ub-lb)/abs(lb)*100:6.2f}" if (ub and lb) else "  n/a"
                print(f"  {pandemic:<10} {r['obj_ARO_expected']:>10.1f} "
                      f"{r['obj_ARO_worst_50']:>10.1f} "
                      f"{lb:>10.1f} {ub_s} {gap_s} "
                      f"{r['obj_IRP']:>10.1f} {str(srp):>10} "
                      f"{r['gain_expected_pct']:>7.1f} "
                      f"{r['n_iter']:>4} {r['solve_time_sec']:>7.1f}s")

    print(f"\n{'='*W}\n  Cross-alpha summary (ARO expected objective)\n{'='*W}")
    for regime in regimes:
        print(f"\n  Regime {regime}")
        print(f"  {'Pandemic':<10}" + "".join(f"  {'a='+str(a):>12}" for a in alphas))
        for pandemic in PANDEMIC_CODES:
            row = f"  {pandemic:<10}"
            for alpha in alphas:
                key = (pandemic, regime, alpha)
                v = results[key]["obj_ARO_expected"] \
                    if key in results and "error" not in results[key] \
                    else float("nan")
                row += f"  {v:>12.1f}"
            print(row)
    print(f"{'='*W}")


def _save_results(results, regimes, alphas, out_dir):
    safe = {}
    for (p, reg, a), v in results.items():
        if "error" not in v:
            safe[f"{p}__R{reg}__a{int(a*100)}"] = {
                kk: vv for kk, vv in v.items() if kk != "convergence"}
    j_path = os.path.join(out_dir, "aro_hccg_final_results.json")
    with open(j_path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    print(f"\n  Results  -> {j_path}")

    c_path = os.path.join(out_dir, "aro_hccg_final_table.csv")
    fields = ["regime","pandemic","alpha","obj_ARO_expected","obj_ARO_worst_50",
              "obj_robust_LB","obj_UB","obj_IRP","gain_expected_pct",
              "n_iter","solve_time_sec","paper_SRP","paper_IRP"]
    with open(c_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for regime in regimes:
            for pandemic in PANDEMIC_CODES:
                for alpha in alphas:
                    key = (pandemic, regime, alpha)
                    if key not in results or "error" in results[key]:
                        continue
                    row = dict(results[key])
                    row.update({"regime": regime, "pandemic": pandemic, "alpha": alpha})
                    w.writerow(row)
    print(f"  Table    -> {c_path}")

# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all(
        data_dir      = DATA_DIR,
        regimes       = ("I", "II", "III"),
        alphas        = [0.0, 0.10, 0.15, 0.20],
        rho           = RHO,
        k_max         = K_MAX,
        save          = True,
        out_dir       = ".",
        force_restart = FORCE_RESTART,   # 控制是否从头跑
    )