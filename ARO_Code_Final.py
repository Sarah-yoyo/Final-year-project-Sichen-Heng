"""
Adversarial Robust Optimisation — Heuristic C&CG v4
=====================================================
Architecture: identical to v3 except for five targeted fixes that address
the "W-full stagnation" problem observed in the v3 logs (iters 13-20).

MASTER MILP: UNCHANGED — binary x_{c,t}^k preserved, no LP relaxation.

Five changes vs v3
------------------
[A] Scenario replacement (fixes the root cause of stagnation)
    When |W| = k_cap, instead of discarding new adversarial scenarios,
    the new scenario REPLACES the weakest scenario currently in W.
    "Weakest" = highest Phi value under the current r^k (least adversarial).
    Implementation: Gurobi variables and constraints for the weakest
    scenario are removed via model.remove(), then fresh ones are added.
    Justification: Zeng & Zhao (2013), Remark 3 — the master remains a
    valid outer approximation as long as W contains feasible scenarios.
    No additional proof required.

[B] Top-N pool scan per iteration
    v3 added only 1 scenario per iteration. v4 adds up to n_pool_add=3
    new pool scenarios per iteration (all passing dual-bound screening),
    giving the master more cutting information per solve. Consistent with
    the pool-evaluation principle of Tonissen & Arts (2018).

[C] LB re-evaluation each iteration
    At the start of each iteration, after obtaining r^k from the master,
    we re-evaluate Phi(r^k; S^w, B^w) for ALL scenarios currently in W.
    LB = min over W of these values (true worst-case under r^k within W).
    This closes the gap faster because LB is no longer stale from old r^j.

[D] Stagnation early stopping
    If UB decreases by less than stag_tol=0.05% for stag_patience=5
    consecutive iterations, terminate. Avoids spending 1800s/iter when
    the solver is clearly stuck.

[E] Re-calibrated time limits
    tau_0=30s (was 120s), gamma=1.2 (was 1.4), tau_max=600s (was 1800s).
    Early iterations involve small MILPs (3-5 scenarios) that solve in
    <15s; the 120s budget was wasted. Later iterations grow to 600s.
    The gap propagation (MIPGap=eps) from v3 is retained.
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
    print("Gurobi not available — cannot run.")

# =============================================================================
# 1. GLOBAL CONFIGURATION
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
K_MAX        = 30        # iteration cap
K_INIT       = 3         # initial scenarios seeded from IRP warm-start
K_SCENE_CAP  = 15        # max |W|
N_POOL_ADD   = 3         # [B] max new pool scenarios added per iteration

# [E] Re-calibrated time limits
TAU_0    = 30.0          # seconds (was 120)
GAMMA_TL = 1.2           # growth factor (was 1.4)
TAU_MAX  = 600.0         # per-iteration cap (was 1800)

# [D] Stagnation detection
STAG_TOL      = 5e-4     # 0.05% relative UB change threshold
STAG_PATIENCE = 5        # consecutive non-improving iters before stopping

EPS_TOTAL = 0.01         # 1% convergence target

# Paper reference values
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
# 2. DATA LOADING  (unchanged from v3)
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
        raise FileNotFoundError(f"No files found for pandemic {pandemic}.")
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
    return S_raw.T.copy(), B_til.T.copy()   # shape (C, T)


def apply_alpha(U: np.ndarray, Bt: np.ndarray, alpha: float):
    if alpha == 0.0:
        return U.copy(), Bt.copy()
    S = (1.0 + alpha) * U
    B = (Bt + alpha * GAMMA_WW) / (1.0 + alpha)
    return S, B


def load_scenarios(pandemic: str, n: int = N_SCENARIOS,
                   seed: int = RANDOM_SEED, alpha: float = 0.0):
    pairs   = list_scenario_files(pandemic)
    rng     = random.Random(seed)
    sampled = rng.sample(pairs, min(n, len(pairs)))
    Ss, Bs  = [], []
    for pp, bp in sampled:
        U, Bt = load_single(pp, bp)
        Sa, Ba = apply_alpha(U, Bt, alpha)
        Ss.append(Sa); Bs.append(Ba)
    return np.stack(Ss), np.stack(Bs)   # (n, C, T)

# =============================================================================
# 3. UNCERTAINTY SET  (unchanged)
# =============================================================================

def build_uncertainty_set(S_all: np.ndarray, B_all: np.ndarray,
                           rho: float = RHO) -> dict:
    S_lb = S_all.min(0); S_ub = S_all.max(0)
    B_lb = B_all.min(0); B_ub = B_all.max(0)
    S_rng = np.where(S_ub > S_lb, S_ub - S_lb, 1.0)
    B_rng = np.where(B_ub > B_lb, B_ub - B_lb, 1.0)
    Gamma = rho * N_COUNTIES * N_TIMES
    return dict(
        S_lb=S_lb, S_ub=S_ub, S_hat=(S_lb+S_ub)/2, S_rng=S_rng,
        B_lb=B_lb, B_ub=B_ub, B_hat=(B_lb+B_ub)/2, B_rng=B_rng,
        Gamma=Gamma,
    )

# =============================================================================
# 4. ANALYTICAL EVALUATION  (Theorem 1, unchanged)
# =============================================================================

def eval_phi(r: np.ndarray, S: np.ndarray, B: np.ndarray) -> float:
    """O(|C|*|T|) sequential forward pass per Theorem 1."""
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


def eval_pool(r: np.ndarray, S_all: np.ndarray,
              B_all: np.ndarray) -> np.ndarray:
    return np.array([eval_phi(r, S_all[i], B_all[i])
                     for i in range(len(S_all))])

# =============================================================================
# 5. IRP WARM-START  (unchanged)
# =============================================================================

def compute_irp(S_all: np.ndarray, B_all: np.ndarray,
                At: np.ndarray):
    n = len(S_all)
    P = np.ones(n) / n
    r = np.zeros((N_COUNTIES, N_TIMES))
    for t in range(N_TIMES):
        if At[t] <= 0:
            continue
        exp_d = (P[:, None] * S_all[:, :, t]).sum(0)
        tot   = exp_d.sum()
        r[:, t] = (At[t] * exp_d / tot if tot > 1e-9
                   else np.full(N_COUNTIES, At[t] / N_COUNTIES))
    vals = eval_pool(r, S_all, B_all)
    return r, float(vals.mean()), vals

# =============================================================================
# 6. POOL SCAN ATTACKER — returns top-N candidates  [B]
# =============================================================================

def pool_attack_topn(r: np.ndarray, S_all: np.ndarray, B_all: np.ndarray,
                     excluded: set, thresh: float,
                     n_return: int = N_POOL_ADD) -> list:
    """
    Evaluate all pool scenarios, return up to n_return worst ones
    that pass screening: val < thresh and not in excluded.
    Sorted ascending by val.
    """
    vals  = eval_pool(r, S_all, B_all)
    order = np.argsort(vals)
    out   = []
    for idx in order:
        i = int(idx)
        if i in excluded:
            continue
        if vals[i] >= thresh:
            break
        out.append((i, S_all[i], B_all[i], float(vals[i])))
        if len(out) >= n_return:
            break
    return out

# =============================================================================
# 7. MASTER MILP  (exact binary MILP + scenario replacement)
# =============================================================================

class MasterMILP:
    """
    Persistent Gurobi MILP master.

    Variables per scenario slot s in W:
        f^s_{c,t}  in [0,1]   — fraction of demand satisfied
        q^s_{c,t}  >= 0       — surplus resources
        x^s_{c,t}  in {0,1}   — =1 iff demand fully met  (BINARY, exact)

    Global:
        r_{c,t}  >= 0
        eta      (worst-case benefit lower envelope)

    [A] Scenario replacement:
        self._slots is a list where each entry stores all Gurobi handles
        (variables + constraints) for one scenario, plus the (S, B) data.
        When W is full and a better scenario is found, _remove_slot()
        deletes the weakest slot's objects from the Gurobi model, then
        _build_slot() creates fresh ones for the replacement scenario.
        The model remains a valid outer approximation throughout.
    """

    # internal slot counter — monotonically increasing for unique naming
    _global_slot_id = 0

    def __init__(self, At: np.ndarray, r_init: np.ndarray):
        self.At  = At
        self.M_t = np.cumsum(At)   # M_t[t] = sum_{t'<=t} A_{t'}
        self._slots: list = []     # list of slot dicts

        self._env = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 0)
        self._env.start()
        self._m = gp.Model(env=self._env)
        self._m.setParam("OutputFlag",   0)
        self._m.setParam("LogToConsole", 0)

        # First-stage variables
        self._r   = self._m.addMVar((N_COUNTIES, N_TIMES), lb=0.0, name="r")
        self._eta = self._m.addVar(lb=-1e9, ub=1e9, name="eta")

        # Release feasibility  sum_{t'<=tau} sum_c r_{c,t'} <= sum_{t'<=tau} A_{t'}
        for tau in range(N_TIMES):
            self._m.addConstr(
                self._r[:, :tau+1].sum() <= float(At[:tau+1].sum()),
                name=f"budget_{tau}"
            )
        self._m.setObjective(self._eta, GRB.MAXIMIZE)

        # Warm-start from IRP [acceleration (i)]
        for c in range(N_COUNTIES):
            for t in range(N_TIMES):
                self._r[c, t].Start = float(r_init[c, t])
        self._m.update()

    # ------------------------------------------------------------------
    def _build_slot(self, S_k: np.ndarray, B_k: np.ndarray) -> dict:
        """
        Add variables + constraints for one scenario to the Gurobi model.
        Returns a slot dict with all handles needed for later removal.
        """
        MasterMILP._global_slot_id += 1
        sid  = MasterMILP._global_slot_id
        pfx  = f"g{sid}"
        C, T = N_COUNTIES, N_TIMES
        M_t  = self.M_t

        f_k = self._m.addMVar((C, T), lb=0.0, ub=1.0,  name=f"f_{pfx}")
        q_k = self._m.addMVar((C, T), lb=0.0,           name=f"q_{pfx}")
        x_k = self._m.addMVar((C, T), vtype=GRB.BINARY, name=f"x_{pfx}")

        inv_cs   = []
        log2_cs  = []
        zero_cs  = []

        # Inventory balance: q_{c,t} = q_{c,t-1} + r_{c,t} - f_{c,t}*S_{c,t}
        for t in range(T):
            s_t  = S_k[:, t]
            lhs  = q_k[:, t-1] if t > 0 else 0.0
            c_inv = self._m.addConstr(
                q_k[:, t] == lhs + self._r[:, t] - f_k[:, t] * s_t,
                name=f"inv_{pfx}_{t}"
            )
            inv_cs.append(c_inv)

        # Complementarity linearisation: q*(1-f)=0
        #   x <= f           (x=1 => f=1)
        #   q <= M_t * x     (q>0 => x=1 => f=1)
        c_l1 = self._m.addConstr(x_k <= f_k, name=f"l1_{pfx}")
        for t in range(T):
            if M_t[t] > 0:
                c_l2 = self._m.addConstr(
                    q_k[:, t] <= float(M_t[t]) * x_k[:, t],
                    name=f"l2_{pfx}_{t}"
                )
                log2_cs.append(c_l2)

        # Boundary: f=x=1 when S_{c,t}=0
        for t in range(T):
            zm = (S_k[:, t] == 0)
            if zm.any():
                idx = np.where(zm)[0]
                zero_cs.append(self._m.addConstr(f_k[idx, t] == 1.0))
                zero_cs.append(self._m.addConstr(x_k[idx, t] == 1.0))

        # eta-cut:  eta <= sum_{c,t} B_{c,t} * S_{c,t} * f_{c,t}
        c_eta = self._m.addConstr(
            self._eta <= (B_k * S_k * f_k).sum(),
            name=f"eta_{pfx}"
        )

        self._m.update()
        return dict(sid=sid, f=f_k, q=q_k, x=x_k,
                    inv=inv_cs, l1=c_l1, l2=log2_cs, zeros=zero_cs,
                    eta=c_eta, S=S_k.copy(), B=B_k.copy())

    # ------------------------------------------------------------------
    def _remove_slot(self, slot: dict):
        """Remove all Gurobi objects belonging to a slot."""
        m = self._m
        # Remove constraints (must happen before variables)
        all_cs = slot["inv"] + [slot["l1"]] + slot["l2"] + slot["zeros"] \
                 + [slot["eta"]]
        for c in all_cs:
            m.remove(c)
        # Remove MVar objects (each MVar wraps a matrix of Var)
        for mvar in (slot["f"], slot["q"], slot["x"]):
            rows = mvar.tolist()
            # tolist() on a 2-D MVar returns list of lists
            if rows and isinstance(rows[0], list):
                flat = [v for row in rows for v in row]
            else:
                flat = rows
            for v in flat:
                m.remove(v)
        m.update()

    # ------------------------------------------------------------------
    def add_scenario(self, S_k: np.ndarray, B_k: np.ndarray):
        """Append a new scenario (only when |W| < k_cap)."""
        slot = self._build_slot(S_k, B_k)
        self._slots.append(slot)

    # ------------------------------------------------------------------
    def try_replace_weakest(self, r_current: np.ndarray,
                             S_new: np.ndarray,
                             B_new: np.ndarray,
                             new_val: float) -> tuple:
        """
        [A] If new_val < Phi of the weakest scenario in W (under r_current),
        replace that scenario with (S_new, B_new).

        Returns (replaced: bool, old_val: float).
        """
        phi_W   = np.array([eval_phi(r_current, s["S"], s["B"])
                             for s in self._slots])
        max_pos = int(np.argmax(phi_W))
        old_val = float(phi_W[max_pos])

        if new_val >= old_val:
            return False, old_val   # new scenario is NOT more adversarial

        # Replace
        self._remove_slot(self._slots[max_pos])
        new_slot = self._build_slot(S_new, B_new)
        self._slots[max_pos] = new_slot
        return True, old_val

    # ------------------------------------------------------------------
    def current_W_phi(self, r: np.ndarray) -> np.ndarray:
        """[C] Return Phi(r; S^s, B^s) for all slots. O(|W|*C*T)."""
        return np.array([eval_phi(r, s["S"], s["B"])
                         for s in self._slots])

    # ------------------------------------------------------------------
    def solve(self, tau_k: float, eps_m: float) -> tuple:
        """
        Solve master MILP.
        Accelerations (iii): adaptive time limit + gap propagation (MIPGap).
        Returns (r_opt, eta_opt, best_bound, status_ok).
        """
        self._m.setParam("TimeLimit", tau_k)
        self._m.setParam("MIPGap",    eps_m)
        self._m.optimize()

        ok = self._m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
        if ok:
            r_opt      = self._r.X.copy()
            best_bound = float(self._m.ObjBound)
            # Warm-start for next iteration
            for c in range(N_COUNTIES):
                for t in range(N_TIMES):
                    self._r[c, t].Start = float(r_opt[c, t])
            return r_opt, float(self._m.ObjVal), best_bound, True
        return None, None, None, False

    def n_scenarios(self) -> int:
        return len(self._slots)

    def dispose(self):
        self._m.dispose()
        self._env.dispose()

# =============================================================================
# 8. MAIN C&CG LOOP
# =============================================================================

def solve_aro_hccg(pandemic: str, regime: str,
                   S_all: np.ndarray, B_all: np.ndarray,
                   At: np.ndarray,
                   alpha: float    = 0.0,
                   rho: float      = RHO,
                   k_max: int      = K_MAX,
                   k_init: int     = K_INIT,
                   k_cap: int      = K_SCENE_CAP,
                   n_pool_add: int = N_POOL_ADD,
                   tau_0: float    = TAU_0,
                   gamma_tl: float = GAMMA_TL,
                   tau_max: float  = TAU_MAX,
                   eps: float      = EPS_TOTAL,
                   stag_tol: float = STAG_TOL,
                   stag_patience: int = STAG_PATIENCE,
                   verbose: bool   = True) -> dict:
    """
    Heuristic C&CG v4 — fixed MIPGap, terminates when attacker finds no worse scenario.
    """
    t_wall = time.perf_counter()
    if not HAVE_GUROBI:
        return {"error": "Gurobi required"}

    r_irp, obj_irp, irp_vals = compute_irp(S_all, B_all, At)

    sep = "─" * 100
    if verbose:
        print(f"\n{sep}")
        print(f"  ARO H-CCG v4 (fixed MIPGap) — {pandemic} / Regime {regime} / alpha={alpha}")
        print(f"  |C|={N_COUNTIES}  |T|={N_TIMES}  |Omega|={len(S_all)}"
              f"  rho={rho}  Gamma={rho*N_COUNTIES*N_TIMES:.0f}")
        print(f"  k_cap={k_cap}  n_pool_add={n_pool_add}"
              f"  tau_0={tau_0}s  gamma={gamma_tl}  tau_max={tau_max}s"
              f"  eps={eps*100:.1f}% (fixed)")
        print(f"  IRP expected={obj_irp:.1f}   IRP worst={irp_vals.min():.1f}")
        print(f"  {'k':>3}  {'UB':>11}  {'LB':>11}  {'gap%':>7}  "
              f"{'added':>6}  {'repl':>5}  {'|W|':>4}"
              f"  {'tau_k':>6}  {'t_MP':>7}")
        print(f"  {sep[2:]}")

    # ── Initialise ────────────────────────────────────────────────────────────
    mp = MasterMILP(At, r_irp)
    in_W = set()

    # Seed W with k_init worst pool scenarios under r_IRP
    for i in range(min(k_init, len(S_all))):
        idx = int(np.argsort(irp_vals)[i])
        mp.add_scenario(S_all[idx], B_all[idx])
        in_W.add(idx)

    # Initial LB: min over W under r_irp
    w_phi_init = mp.current_W_phi(r_irp)
    LB = float(w_phi_init.min())
    UB = float("inf")
    r_k = r_irp.copy()
    log = []
    stag_cnt = 0
    prev_UB = float("inf")

    # ── C&CG iterations ───────────────────────────────────────────────────────
    for k in range(1, k_max + 1):

        # 固定 MIPGap
        current_mip_gap = eps

        # [1] Solve master MILP
        tau_k = min(tau_0 * (gamma_tl ** k), tau_max)
        t0 = time.perf_counter()
        r_new, _, best_bd, ok = mp.solve(tau_k, current_mip_gap)
        t_mp = time.perf_counter() - t0

        if ok:
            r_k = r_new
            UB = best_bd

        # [2] Re-evaluate LB under current r_k using all scenarios in W
        if mp.n_scenarios() > 0:
            w_phi = mp.current_W_phi(r_k)
            LB = float(w_phi.min())

        # [3] Attacker: find new adversarial scenarios (threshold = LB)
        candidates = pool_attack_topn(r_k, S_all, B_all, in_W,
                                      thresh=LB, n_return=n_pool_add)

        # [4] Termination: no worse scenario found -> converged
        if not candidates:
            if verbose:
                print(f"  Converged at k={k}: no adversarial scenario (LB={LB:.1f})")
            break

        # [5] Add or replace scenarios
        n_added = 0
        n_repl = 0
        for (p_idx, p_S, p_B, p_val) in candidates:
            if mp.n_scenarios() < k_cap:
                mp.add_scenario(p_S, p_B)
                in_W.add(p_idx)
                n_added += 1
            else:
                replaced, old_val = mp.try_replace_weakest(r_k, p_S, p_B, p_val)
                if replaced:
                    n_repl += 1
            # Update LB with the new scenario's value (it is <= current LB by construction)
            LB = min(LB, p_val)

        # [6] Stagnation detection (based on UB improvement)
        if np.isfinite(UB) and np.isfinite(prev_UB):
            rel_chg = abs(UB - prev_UB) / max(abs(prev_UB), 1e-12)
            stag_cnt = stag_cnt + 1 if rel_chg < stag_tol else 0
        else:
            stag_cnt = 0
        prev_UB = UB
        if stag_cnt >= stag_patience:
            if verbose:
                print(f"  Early stop (stagnation) at k={k}")
            break

        # [7] Logging
        gap = ((UB - LB) / abs(LB)) if (np.isfinite(UB) and abs(LB) > 1e-12) else float("inf")
        if verbose:
            ub_s = f"{UB:11.1f}" if np.isfinite(UB) else "        inf"
            gap_s = f"{gap*100:7.3f}" if np.isfinite(gap) else "    inf"
            print(f"  {k:>3}  {ub_s}  {LB:11.1f}  {gap_s}  "
                  f"{n_added:>6}  {n_repl:>5}  {mp.n_scenarios():>4}"
                  f"  {tau_k:6.1f}  {t_mp:6.1f}s")

        log.append({
            "k": k,
            "UB":      float(UB) if np.isfinite(UB) else None,
            "LB":      float(LB),
            "gap_pct": float(gap * 100) if np.isfinite(gap) else None,
            "n_added": n_added, "n_repl": n_repl,
            "n_W":     mp.n_scenarios(),
            "tau_k":   tau_k, "t_mp": t_mp,
            "elapsed": time.perf_counter() - t_wall,
        })

    else:
        if verbose:
            print(f"  Iteration limit ({k_max}) reached.")

    mp.dispose()

    # ── Final evaluation on all 50 scenarios ──────────────────────────────────
    vals_final   = eval_pool(r_k, S_all, B_all)
    obj_expected = float(vals_final.mean())
    obj_wc50     = float(vals_final.min())
    gain_exp     = (obj_expected - obj_irp) / max(abs(obj_irp), 1e-12) * 100
    elapsed      = time.perf_counter() - t_wall

    if verbose:
        print(f"\n  Expected (50 scenarios) : {obj_expected:.1f}")
        print(f"  Worst-case (50 scen.)   : {obj_wc50:.1f}")
        print(f"  Robust LB (over W)      : {LB:.1f}")
        print(f"  UB (master bound)       : {UB:.1f}")
        print(f"  IRP benchmark           : {obj_irp:.1f}")
        print(f"  Gain (expected vs IRP)  : {gain_exp:.2f}%")
        print(f"  Total time              : {elapsed:.1f}s")
        print(sep)

    return {
        "pandemic": pandemic, "regime": regime,
        "alpha": alpha, "rho": rho,
        "obj_ARO_expected":  obj_expected,
        "obj_ARO_worst_50":  obj_wc50,
        "obj_robust_LB":     float(LB),
        "obj_UB":            float(UB) if np.isfinite(UB) else None,
        "obj_IRP":           obj_irp,
        "gain_expected_pct": gain_exp,
        "n_iter":            k,
        "n_scenes_final":    mp.n_scenarios(),
        "solve_time_sec":    elapsed,
        "r_total_by_time":   r_k.sum(axis=0).tolist(),
        "convergence":       log,
    }
# =============================================================================
# 9. EXPERIMENT RUNNER  (unchanged structure from v3)
# =============================================================================

def run_all(data_dir: str   = DATA_DIR,
            regimes: tuple  = ("I", "II", "III"),
            alphas: list    = ALPHA_VALUES,
            rho: float      = RHO,
            k_max: int      = K_MAX,
            save: bool      = True,
            out_dir: str    = ".") -> dict:

    global DATA_DIR
    DATA_DIR = data_dir

    print("=" * 80)
    print("  ARO H-CCG v4  |  exact MILP · scenario replacement · stagnation stop")
    print(f"  rho={rho}  Gamma={rho*N_COUNTIES*N_TIMES:.0f}"
          f"  regimes={regimes}  alphas={alphas}  K_max={k_max}")
    print("=" * 80)

    results = {}
    for regime in regimes:
        At = make_At(regime)
        for alpha in alphas:
            print(f"\n{'━'*80}\n  REGIME {regime}  |  alpha={alpha}\n{'━'*80}")
            for pandemic in PANDEMIC_CODES:
                try:
                    S_all, B_all = load_scenarios(pandemic, N_SCENARIOS,
                                                   RANDOM_SEED, alpha)
                except FileNotFoundError as e:
                    print(f"  [SKIP {pandemic}]: {e}")
                    continue
                res = solve_aro_hccg(
                    pandemic, regime, S_all, B_all, At,
                    alpha=alpha, rho=rho, k_max=k_max, verbose=True)
                if "error" not in res:
                    res["paper_SRP"] = PAPER_REF[regime]["SRP"].get(pandemic)
                    res["paper_IRP"] = PAPER_REF[regime]["IRP"].get(pandemic)
                results[(pandemic, regime, alpha)] = res

    _print_summary(results, regimes, alphas)
    if save:
        _save_results(results, regimes, alphas, out_dir)
    return results


def _print_summary(results: dict, regimes: tuple, alphas: list):
    W = 110
    for regime in regimes:
        for alpha in alphas:
            print(f"\n{'='*W}\n  Regime {regime}  |  alpha={alpha}\n{'='*W}")
            print(f"  {'Pandemic':<10} {'ARO(exp)':>10} {'ARO(wc50)':>10} "
                  f"{'LB':>10} {'UB':>10} {'IRP':>10} {'SRP_paper':>10} "
                  f"{'Gain%':>7} {'k':>4} {'Time':>8}")
            print("  " + "-"*(W-4))
            for pandemic in PANDEMIC_CODES:
                key = (pandemic, regime, alpha)
                if key not in results or "error" in results[key]:
                    continue
                r   = results[key]
                srp = r.get("paper_SRP", "N/A")
                ub_s = f"{r['obj_UB']:10.1f}" if r["obj_UB"] else "       inf"
                print(f"  {pandemic:<10} {r['obj_ARO_expected']:>10.1f} "
                      f"{r['obj_ARO_worst_50']:>10.1f} "
                      f"{r['obj_robust_LB']:>10.1f} {ub_s} "
                      f"{r['obj_IRP']:>10.1f} {str(srp):>10} "
                      f"{r['gain_expected_pct']:>7.1f} "
                      f"{r['n_iter']:>4} {r['solve_time_sec']:>7.1f}s")
    print(f"\n{'='*W}\n  Cross-alpha summary\n{'='*W}")
    for regime in regimes:
        print(f"\n  Regime {regime}")
        print(f"  {'Pandemic':<10}" + "".join(
            f"  {'a='+str(a):>12}" for a in alphas))
        for pandemic in PANDEMIC_CODES:
            row = f"  {pandemic:<10}"
            for alpha in alphas:
                key = (pandemic, regime, alpha)
                v = (results[key]["obj_ARO_expected"]
                     if key in results and "error" not in results[key]
                     else float("nan"))
                row += f"  {v:>12.1f}"
            print(row)
    print(f"{'='*W}")


def _save_results(results: dict, regimes: tuple, alphas: list, out_dir: str):
    safe = {}
    for (p, reg, a), v in results.items():
        if "error" not in v:
            safe[f"{p}__R{reg}__a{int(a*100)}"] = {
                kk: vv for kk, vv in v.items() if kk != "convergence"}
    j_path = os.path.join(out_dir, "aro_hccg_v4_results.json")
    with open(j_path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    print(f"\n  Results  -> {j_path}")

    c_path = os.path.join(out_dir, "aro_hccg_v4_table.csv")
    fields = ["regime","pandemic","alpha",
              "obj_ARO_expected","obj_ARO_worst_50","obj_robust_LB","obj_UB",
              "obj_IRP","gain_expected_pct",
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
                    row.update({"regime": regime, "pandemic": pandemic,
                                "alpha": alpha})
                    w.writerow(row)
    print(f"  Table    -> {c_path}")

# =============================================================================
# 10. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all(
        data_dir = DATA_DIR,
        regimes  = ("I", "II", "III"),
        alphas   = [0.0, 0.10, 0.15, 0.20],
        rho      = RHO,
        k_max    = K_MAX,
        save     = True,
        out_dir  = ".",
    )
