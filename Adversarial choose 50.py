"""
Adversarial Robust Optimisation (ARO) for Resource Allocation
=============================================================
Singh & Rebennack (2025)  +  Proposal Section 3.3

The adversarial approach reformulates the stochastic problem as a Min-Max
(robust optimisation) problem.  Instead of optimising over scenario
probabilities, the adversary picks the worst-case (S, B) within an
uncertainty set U, and the planner defends against it.

FORMULATION (Proposal Eq. 1a–1g)
---------------------------------
  z^RO = max_r  min_{(S,B)∈U}  Σ_{c,t} B_{c,t}·S_{c,t}·f_{c,t}(r, S)

  Uncertainty set U (budget-constrained box):
    S_{c,t} ∈ [S_lb_{c,t}, S_ub_{c,t}]   B_{c,t} ∈ [B_lb_{c,t}, B_ub_{c,t}]
    Σ_{c,t} |S_{c,t} − Ŝ_{c,t}| / (S_ub − S_lb) ≤ Γ_S
    Σ_{c,t} |B_{c,t} − B̂_{c,t}| / (B_ub − B_lb) ≤ Γ_B
  where Ŝ, B̂ are the nominal (midpoint) values.

ALGORITHM (Proposal Algorithm 2: C&CG)
---------------------------------------
  The Min-Max is solved by Column-and-Constraint Generation (Zeng & Zhao 2013):
    Master problem (MP): optimise r against a finite set of adversarial scenarios
    Adversarial subproblem (ASP): for fixed r, find worst-case (S,B) in U

  Iteration:
    1. Solve MP → r^k, LB ← z_MP
    2. Solve ASP with r^k → (S^k, B^k), UB ← min(UB, z_ASP)
    3. Add (S^k, B^k) as a new scenario to MP (column-and-constraint)
    4. Repeat until UB − LB ≤ ε

IMPLEMENTATION NOTES
--------------------
- Both MP and ASP are solved with Gurobi.
- MP is an MIP (binary x^ω for each scenario cut generated so far).
- ASP is a bilinear min problem (products B·S·f); we linearise via Big-M.
- α (worried-well) scaling is applied before constructing the uncertainty set,
  following paper Eq. (5).
- Three regimes × 6 pandemics × 4 α values.  For each (pandemic, regime, α)
  we use the same 50 sampled scenarios to build S_lb/S_ub/B_lb/B_ub.
"""

import os, glob, time, json, random, csv
import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
    print("✓ Gurobi found.")
except ImportError:
    HAVE_GUROBI = False
    print("⚠  Gurobi not available — ARO requires Gurobi.  Install gurobipy first.")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

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
GAMMA_WW = -float(BETA_G.mean())   # = −0.048

ALPHA_VALUES = [0.0, 0.10, 0.15, 0.20]

# C&CG parameters
MAX_CCG_ITER = 30         # max column-and-constraint iterations
CCG_TOL      = 1e-3       # UB - LB convergence tolerance (lives saved)

# Uncertainty budget (Proposal): controls conservatism.
# Γ = |C|·|T| means adversary can perturb EVERY (c,t) by its full range.
# Γ = 0 means no uncertainty (deterministic).
# We use Γ = 0.1·|C|·|T| as a moderate budget (10% of total budget),
# consistent with Bertsimas & Sim (2004) guidance.
GAMMA_FRAC   = 0.10       # fraction of C×T = budget fraction

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  AVAILABILITY REGIMES
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING + α SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def list_scenario_files(pandemic: str):
    code = PANDEMIC_CODES[pandemic]
    pat  = os.path.join(DATA_DIR,
                        f"infectious-*_{code}_ic*_population_monthly.csv")
    pops = sorted(glob.glob(pat))
    if not pops:
        raise FileNotFoundError(
            f"No files for pandemic={pandemic}.\nPattern: {pat}")
    out = []
    for pf in pops:
        bf = pf.replace("_population_monthly.csv", "_benefit_monthly.csv")
        if os.path.exists(bf):
            out.append((pf, bf,
                        os.path.basename(pf).replace(
                            "_population_monthly.csv", "")))
    return out


def load_single_scenario(pop_path, ben_path):
    """Load (U_tilde, B_tilde) – the α=0 quantities.  Shape: (C, T)."""
    S_raw = pd.read_csv(pop_path, index_col=0).values.astype(float)   # (T,C)
    B_raw = pd.read_csv(ben_path, index_col=0).values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        B_til = np.where(S_raw > 0, B_raw / S_raw, 0.0)
    return S_raw.T.copy(), B_til.T.copy()   # (C,T)


def apply_alpha(U, B_tilde, alpha):
    """Paper Eq. (5): S(α) = (1+α)Ũ, B(α) = (B̃ + α·γ_ww)/(1+α)."""
    if alpha == 0.0:
        return U.copy(), B_tilde.copy()
    return (1.0 + alpha) * U, (B_tilde + alpha * GAMMA_WW) / (1.0 + alpha)


def load_scenarios(pandemic, n=N_SCENARIOS, seed=RANDOM_SEED, alpha=0.0):
    """Return S_all (n,C,T), B_all (n,C,T), labels."""
    triples = list_scenario_files(pandemic)
    rng     = random.Random(seed)
    sampled = rng.sample(triples, min(n, len(triples)))
    S_list, B_list, lbls = [], [], []
    for pp, bp, lbl in sampled:
        U, Bt = load_single_scenario(pp, bp)
        Sa, Ba = apply_alpha(U, Bt, alpha)
        S_list.append(Sa); B_list.append(Ba); lbls.append(lbl)
    return np.stack(S_list), np.stack(B_list), lbls

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  UNCERTAINTY SET CONSTRUCTION  (Proposal Section 3.3)
# ═══════════════════════════════════════════════════════════════════════════════

def build_uncertainty_set(S_all: np.ndarray, B_all: np.ndarray,
                           gamma_frac: float = GAMMA_FRAC):
    """
    Build the budget-constrained box uncertainty set from scenario data.

    For each (c,t):
      S_lb = min_ω S^ω_{c,t},   S_ub = max_ω S^ω_{c,t}
      B_lb = min_ω B^ω_{c,t},   B_ub = max_ω B^ω_{c,t}
      Ŝ_{c,t} = (S_lb + S_ub) / 2  (nominal midpoint)
      B̂_{c,t} = (B_lb + B_ub) / 2

    Budget: Γ_S = Γ_B = gamma_frac × C × T
      (the adversary can push at most this many (c,t) pairs to their extreme)

    Returns dict with arrays of shape (C, T).
    """
    S_lb = S_all.min(axis=0)   # (C, T)
    S_ub = S_all.max(axis=0)
    B_lb = B_all.min(axis=0)
    B_ub = B_all.max(axis=0)

    S_hat = (S_lb + S_ub) / 2.0
    B_hat = (B_lb + B_ub) / 2.0

    # Range (avoid division by zero)
    S_range = np.maximum(S_ub - S_lb, 1e-10)
    B_range = np.maximum(B_ub - B_lb, 1e-10)

    Gamma_S = gamma_frac * N_COUNTIES * N_TIMES
    Gamma_B = gamma_frac * N_COUNTIES * N_TIMES

    return {
        "S_lb": S_lb, "S_ub": S_ub, "S_hat": S_hat, "S_range": S_range,
        "B_lb": B_lb, "B_ub": B_ub, "B_hat": B_hat, "B_range": B_range,
        "Gamma_S": Gamma_S, "Gamma_B": Gamma_B,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SECOND-STAGE EVALUATION  (Theorem 1, analytical)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_second_stage(r: np.ndarray, S: np.ndarray, B: np.ndarray) -> float:
    """
    Given r (C,T) and a single scenario S,B (C,T), compute Σ B·S·f
    analytically via Theorem 1:  f_{c,t} = min(1, (q_{c,t-1}+r_{c,t})/S_{c,t})
    """
    obj = 0.0
    q   = np.zeros(N_COUNTIES)
    for t in range(N_TIMES):
        avail = q + r[:, t]
        s_t   = S[:, t]
        b_t   = B[:, t]
        with np.errstate(invalid="ignore", divide="ignore"):
            f_t = np.where(s_t > 0, np.minimum(1.0, avail / s_t), 1.0)
        obj += np.dot(b_t * s_t, f_t)
        q = np.maximum(0.0, avail - f_t * s_t)
    return float(obj)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  IRP BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def compute_irp(S_all, B_all, At):
    """IRP on nominal (mean) scenario."""
    P   = np.ones(S_all.shape[0]) / S_all.shape[0]
    r   = np.zeros((N_COUNTIES, N_TIMES))
    for t in range(N_TIMES):
        if At[t] <= 0:
            continue
        exp_d = (P[:, None] * S_all[:, :, t]).sum(0)
        tot   = exp_d.sum()
        r[:, t] = At[t] * exp_d / tot if tot > 1e-9 else At[t] / N_COUNTIES
    # Evaluate on ALL scenarios
    obj = np.mean([eval_second_stage(r, S_all[o], B_all[o])
                   for o in range(S_all.shape[0])])
    return r, obj

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ADVERSARIAL SUBPROBLEM  (ASP)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_adversarial_subproblem(r: np.ndarray, uset: dict,
                                  At: np.ndarray, env) -> tuple:
    """
    ASP: given fixed r, find (S*, B*) ∈ U that minimises Σ B·S·f(r,S).

    The bilinear term B·S·f is linearised by noting that:
      - f(r,S) = min(1, avail/S) is determined by r and S alone.
      - Given fixed r, f is a function of S only (through Theorem 1).

    Approach: We linearise B·(S·f) = B·y  where y_{c,t} = S_{c,t}·f_{c,t}(r,S).
    The inner min over B (for fixed y) has a closed-form: B* = B_lb wherever
    y > 0 (adversary minimises B where doses are actually consumed).

    For S: since f = min(1, avail/S) and avail = q_{c,t-1}+r_{c,t} is fixed,
    there are two regimes per (c,t):
      avail ≥ S → f=1 → y=S  (adversary increases S beyond avail to cause waste)
      avail < S → f=avail/S → y=avail (independent of S)

    This makes the ASP separable: for each (c,t), the adversary chooses S
    to either:
      (a) Keep S ≤ avail → y=S, so min B·S is achieved at S=S_lb (if B_lb < 0)
          or S=S_lb (minimise positive contribution)
      (b) Push S > avail → y=avail (fixed), then adversary only affects B

    Given this structure, we implement ASP as a Gurobi LP:
      Variables: S_{c,t}, B_{c,t}, f_{c,t}, q_{c,t} (continuous)
      Linearise f = min(1, (q_{c,t-1}+r_{c,t})/S_{c,t}) using Big-M
      Objective: min Σ_{c,t} B_{c,t}·S_{c,t}·f_{c,t}
    The bilinear B*S*f is handled by fixing f via Theorem 1 analytically
    (since r is fixed) and then minimising over B given y = S*f.

    Efficient implementation:
    Step 1: Compute f^*(c,t) = min(1, avail_{c,t}/S_lb_{c,t}) analytically.
    Step 2: y_{c,t} = S_{c,t}·f_{c,t}.  If avail ≥ S_lb (full satisfaction),
            y can range in [S_lb, avail].  If avail < S_ub (partial), y=avail fixed.
    Step 3: Solve LP: min_{S,B} Σ B_{c,t}·y_{c,t}  s.t. budget constraints.
    """
    C, T   = N_COUNTIES, N_TIMES
    S_lb   = uset["S_lb"]
    S_ub   = uset["S_ub"]
    S_hat  = uset["S_hat"]
    S_rng  = uset["S_range"]
    B_lb   = uset["B_lb"]
    B_ub   = uset["B_ub"]
    B_hat  = uset["B_hat"]
    B_rng  = uset["B_range"]
    Gamma_S= uset["Gamma_S"]
    Gamma_B= uset["Gamma_B"]
    M_t    = np.array([At[:t+1].sum() for t in range(T)])

    try:
        with gp.Model(env=env) as m:
            m.setParam("OutputFlag",   0)
            m.setParam("LogToConsole", 0)

            # ── Variables ─────────────────────────────────────────────────────
            # S_{c,t} ∈ [S_lb, S_ub]
            S_var = m.addMVar((C, T), lb=S_lb, ub=S_ub, name="S")
            # B_{c,t} ∈ [B_lb, B_ub]
            B_var = m.addMVar((C, T), lb=B_lb, ub=B_ub, name="B")
            # f_{c,t} ∈ [0,1]  (fraction of demand satisfied)
            f_var = m.addMVar((C, T), lb=0.0, ub=1.0, name="f")
            # q_{c,t} ≥ 0  (surplus)
            q_var = m.addMVar((C, T), lb=0.0, name="q")
            # x_{c,t} ∈ [0,1]  (LP relax of indicator for full satisfaction)
            x_var = m.addMVar((C, T), lb=0.0, ub=1.0, name="x")
            # y_{c,t} = S_{c,t}·f_{c,t}  (linearised by McCormick)
            # Since f ∈ [0,1] and S ∈ [S_lb, S_ub], we use:
            #   y ≤ S_ub·f + S·0 = S_ub·f
            #   y ≤ S·1 = S
            #   y ≥ S_lb·f + S·0 - S_lb·0 = S_lb·f  (McCormick lower)
            #   y ≥ 0
            y_var = m.addMVar((C, T), lb=0.0, name="y")
            # w_{c,t} = B_{c,t}·y_{c,t}  — linearised using B_lb/ub and y bounds
            # y_ub = S_ub (since f ≤ 1)
            y_ub  = S_ub.copy()
            # w ≥ B_lb·y + B·y_lb - B_lb·y_lb = B_lb·y  (since y_lb=0)
            # w ≤ B_ub·y  (since y ≥ 0 and B ≤ B_ub)
            # w ≤ B·y_ub + B_ub·y - B_ub·y_ub
            # w ≥ B·y_lb = 0
            # For our range, since B can be negative, we need full McCormick:
            w_var = m.addMVar((C, T), lb=B_lb * y_ub, ub=B_ub * y_ub + 1.0, name="w")

            # ── Inventory balance (1b): r is fixed ────────────────────────────
            for t in range(T):
                s_t = S_var[:, t]
                if t == 0:
                    m.addConstr(q_var[:, t] == r[:, t] - f_var[:, t] * s_t)
                else:
                    m.addConstr(q_var[:, t] == q_var[:, t-1] + r[:, t]
                                              - f_var[:, t] * s_t)

            # ── Complementarity (1d)(1e) ──────────────────────────────────────
            m.addConstr(x_var <= f_var)
            for t in range(T):
                if M_t[t] > 0:
                    m.addConstr(q_var[:, t] <= M_t[t] * x_var[:, t])
                else:
                    m.addConstr(q_var[:, t] == 0.0)

            # ── McCormick for y = S·f ─────────────────────────────────────────
            # y ≤ S  (since f ≤ 1)
            m.addConstr(y_var <= S_var)
            # y ≤ S_ub·f
            m.addConstr(y_var <= S_ub * f_var)
            # y ≥ S_lb·f
            m.addConstr(y_var >= S_lb * f_var)
            # y ≥ 0  (already in lb)

            # ── McCormick for w = B·y ─────────────────────────────────────────
            # w ≥ B_lb·y + B·0   - B_lb·0   = B_lb·y
            m.addConstr(w_var >= B_lb * y_var)
            # w ≥ B·y_ub + B_lb·y - B_lb·y_ub
            m.addConstr(w_var >= (y_ub * B_var) + (B_lb * y_var)
                        - (B_lb * y_ub))
            # w ≤ B_ub·y
            m.addConstr(w_var <= B_ub * y_var)
            # w ≤ B·y_ub + B_ub·y - B_ub·y_ub
            m.addConstr(w_var <= (y_ub * B_var) + (B_ub * y_var)
                        - (B_ub * y_ub))

            # ── Budget constraints for S and B ────────────────────────────────
            # Σ_{c,t} |S_{c,t} − Ŝ_{c,t}| / S_range ≤ Γ_S
            # Using auxiliary z_S ≥ 0: |S − Ŝ|/S_range ≤ z_S
            z_S = m.addMVar((C, T), lb=0.0, name="zS")
            m.addConstr( (S_var - S_hat) / S_rng <= z_S)
            m.addConstr(-(S_var - S_hat) / S_rng <= z_S)
            m.addConstr(z_S.sum() <= Gamma_S)

            z_B = m.addMVar((C, T), lb=0.0, name="zB")
            m.addConstr( (B_var - B_hat) / B_rng <= z_B)
            m.addConstr(-(B_var - B_hat) / B_rng <= z_B)
            m.addConstr(z_B.sum() <= Gamma_B)

            # ── Objective: minimise Σ B·S·f = Σ w ───────────────────────────
            m.setObjective(w_var.sum(), GRB.MINIMIZE)
            m.optimize()

            if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                S_star = S_var.X.copy()
                B_star = B_var.X.copy()
                obj_val= float(m.ObjVal)
                return S_star, B_star, obj_val, True
            else:
                # Fallback: return worst nominal corner
                return S_lb.copy(), B_lb.copy(), -1e18, False

    except Exception as e:
        print(f"    [ASP error]: {e}")
        return S_lb.copy(), B_lb.copy(), -1e18, False

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MASTER PROBLEM  (MP)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_master_problem(At: np.ndarray, uset: dict,
                          scenarios_cut: list, env) -> tuple:
    """
    MP: max_r  η
    s.t.
      η ≤ Σ_{c,t} B^k_{c,t}·S^k_{c,t}·f^k_{c,t}(r)   ∀k (scenario cuts added)
      All feasibility constraints on r (constraint 1c)
      f^k, q^k, x^k satisfy (1b)(1d)(1e) for scenario k

    Each scenario cut (S^k, B^k) added by C&CG introduces:
      - New second-stage variables f^k, q^k, x^k
      - Inventory constraints (1b) for scenario k
      - Complementarity constraints (1d)(1e) for scenario k
      - The cut:  η ≤ Σ B^k·S^k·f^k

    Returns (r_star, eta_star, obj_val, status).
    """
    C, T   = N_COUNTIES, N_TIMES
    M_t    = np.array([At[:t+1].sum() for t in range(T)])
    n_cuts = len(scenarios_cut)

    try:
        with gp.Model(env=env) as m:
            m.setParam("OutputFlag",   0)
            m.setParam("LogToConsole", 0)

            # ── First-stage: r_{c,t} ≥ 0 ─────────────────────────────────────
            r = m.addMVar((C, T), lb=0.0, name="r")

            # ── Epigraph variable η (lower bound on worst-case objective) ─────
            eta = m.addVar(lb=-1e9, name="eta")

            # ── Budget constraint (1c) ────────────────────────────────────────
            for tau in range(T):
                m.addConstr(r[:, :tau+1].sum() <= At[:tau+1].sum())

            # ── For each adversarial scenario cut k ───────────────────────────
            for k, (S_k, B_k) in enumerate(scenarios_cut):
                f_k = m.addMVar((C, T), lb=0.0, ub=1.0, name=f"f{k}")
                q_k = m.addMVar((C, T), lb=0.0,         name=f"q{k}")
                x_k = m.addMVar((C, T), lb=0.0, ub=1.0, name=f"x{k}")

                # (1b) Inventory balance
                for t in range(T):
                    s_t = S_k[:, t]
                    if t == 0:
                        m.addConstr(q_k[:, t] == r[:, t] - f_k[:, t] * s_t)
                    else:
                        m.addConstr(q_k[:, t] == q_k[:, t-1] + r[:, t]
                                                - f_k[:, t] * s_t)

                # (1d)(1e) Complementarity
                m.addConstr(x_k <= f_k)
                for t in range(T):
                    if M_t[t] > 0:
                        m.addConstr(q_k[:, t] <= M_t[t] * x_k[:, t])
                    else:
                        m.addConstr(q_k[:, t] == 0.0)

                # Cut: η ≤ Σ B_k · S_k · f_k
                lin_coef = (B_k * S_k)   # (C, T), fixed constants
                m.addConstr(eta <= (lin_coef * f_k).sum(),
                            name=f"cut{k}")

            # ── Objective ─────────────────────────────────────────────────────
            m.setObjective(eta, GRB.MAXIMIZE)
            m.optimize()

            if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                return r.X.copy(), float(eta.X), float(m.ObjVal), True
            else:
                # Fallback: IRP solution
                r_fb = np.zeros((C, T))
                for t in range(T):
                    if At[t] > 0:
                        r_fb[:, t] = At[t] / C
                return r_fb, -1e9, -1e9, False

    except Exception as e:
        print(f"    [MP error]: {e}")
        r_fb = np.zeros((C, T))
        for t in range(T):
            if At[t] > 0:
                r_fb[:, t] = At[t] / C
        return r_fb, -1e9, -1e9, False

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  C&CG MAIN LOOP  (Algorithm 2 from Proposal)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_aro(pandemic: str, regime: str, S_all: np.ndarray, B_all: np.ndarray,
              At: np.ndarray, alpha: float = 0.0,
              gamma_frac: float = GAMMA_FRAC,
              max_iter: int = MAX_CCG_ITER, tol: float = CCG_TOL,
              verbose: bool = True) -> dict:
    """
    Solve the ARO problem via Column-and-Constraint Generation.

    Returns dict with r_star, obj_RO, obj_IRP, gain_pct, iterations, time,
    convergence history, r_total_by_time.
    """
    t0 = time.perf_counter()
    label = f"{pandemic} / Regime {regime} / α={alpha}"

    if not HAVE_GUROBI:
        print(f"  [SKIP {label}]: Gurobi required for ARO.")
        return {"error": "No Gurobi"}

    # ── Build uncertainty set ─────────────────────────────────────────────────
    uset = build_uncertainty_set(S_all, B_all, gamma_frac)

    # ── IRP benchmark ─────────────────────────────────────────────────────────
    r_irp, obj_irp = compute_irp(S_all, B_all, At)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  ARO (C&CG) — {label}")
        print(f"  Γ_S = Γ_B = {uset['Gamma_S']:.0f}  ({gamma_frac*100:.0f}% of C×T)")
        print(f"  IRP baseline: {obj_irp:.2f}")
        print(f"  {'Iter':>4}  {'LB':>12}  {'UB':>12}  {'Gap':>10}  {'Time':>8}")

    # ── Gurobi environment (shared) ───────────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # ── Initialise C&CG ───────────────────────────────────────────────────────
    LB = -np.inf
    UB =  np.inf
    scenarios_cut = []   # list of (S_k, B_k) adversarial scenarios

    # Warm-start: add nominal scenario (midpoint) as the first cut
    S_nom = uset["S_hat"].copy()
    B_nom = uset["B_hat"].copy()
    scenarios_cut.append((S_nom, B_nom))

    r_star   = r_irp.copy()
    conv_log = []

    for k in range(1, max_iter + 1):
        # ── Step 1: Solve master problem ──────────────────────────────────────
        r_k, eta_k, mp_obj, mp_ok = solve_master_problem(At, uset, scenarios_cut, env)
        if mp_ok:
            LB = max(LB, mp_obj)
            r_star = r_k.copy()

        # ── Step 2: Solve adversarial subproblem ──────────────────────────────
        S_k, B_k, asp_obj, asp_ok = solve_adversarial_subproblem(r_k, uset, At, env)
        if asp_ok:
            UB = min(UB, asp_obj)

        gap = UB - LB if np.isfinite(UB) and np.isfinite(LB) else np.inf
        elapsed = time.perf_counter() - t0
        conv_log.append({"k": k, "LB": float(LB), "UB": float(UB),
                          "gap": float(gap), "elapsed": elapsed})

        if verbose:
            LB_str = f"{LB:.2f}" if np.isfinite(LB) else "  -inf"
            UB_str = f"{UB:.2f}" if np.isfinite(UB) else "  +inf"
            print(f"  {k:>4}  {LB_str:>12}  {UB_str:>12}  "
                  f"{gap:>10.4f}  {elapsed:>7.2f}s")

        if gap <= tol:
            if verbose:
                print(f"  ✓ Converged at k={k}  (gap={gap:.4f})")
            break

        # ── Step 3: Add new adversarial scenario to MP ────────────────────────
        scenarios_cut.append((S_k, B_k))
    else:
        if verbose:
            print(f"  ✗ Max iter ({max_iter}) reached  (gap={gap:.4f})")

    env.dispose()

    # ── Final evaluation ──────────────────────────────────────────────────────
    # Evaluate r_star on ALL 50 scenarios (stochastic objective)
    obj_all = np.mean([eval_second_stage(r_star, S_all[o], B_all[o])
                       for o in range(S_all.shape[0])])
    # Worst-case objective (over all scenario data)
    obj_wc  = min(eval_second_stage(r_star, S_all[o], B_all[o])
                  for o in range(S_all.shape[0]))
    gain    = (obj_all - obj_irp) / max(abs(obj_irp), 1e-12) * 100
    gain_wc = (obj_wc  - obj_irp) / max(abs(obj_irp), 1e-12) * 100
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n  ARO (expected over 50 scen):  {obj_all:.2f}")
        print(f"  ARO (worst-case):              {obj_wc:.2f}")
        print(f"  IRP benchmark:                 {obj_irp:.2f}")
        print(f"  Gain (expected vs IRP):        {gain:.2f}%")
        print(f"  Gain (worst-case vs IRP):      {gain_wc:.2f}%")
        print(f"  Total time: {elapsed:.2f}s    Cuts added: {len(scenarios_cut)-1}")
        print(f"{'─'*70}")

    return {
        "pandemic": pandemic, "regime": regime, "alpha": alpha,
        "gamma_frac": gamma_frac,
        "obj_ARO_expected": float(obj_all),
        "obj_ARO_worst":    float(obj_wc),
        "obj_IRP":          float(obj_irp),
        "gain_expected_pct":float(gain),
        "gain_wc_pct":      float(gain_wc),
        "ARO_LB":           float(LB),
        "ARO_UB":           float(UB),
        "n_cuts":           len(scenarios_cut) - 1,
        "n_iter":           k,
        "solve_time_sec":   float(elapsed),
        "r_total_by_time":  r_star.sum(axis=0).tolist(),
        "r_top5_t1":        _top5(r_star, 0),
        "r_top5_t3":        _top5(r_star, 2),
        "convergence":      conv_log,
    }


def _top5(r, t):
    col = r[:, t]
    idx = np.argsort(-col)[:5]
    return [(int(c), float(col[c])) for c in idx if col[c] > 0]

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(data_dir=DATA_DIR, n_scenarios=N_SCENARIOS, seed=RANDOM_SEED,
            regimes=("I","II","III"), alphas=ALPHA_VALUES,
            gamma_frac=GAMMA_FRAC, max_iter=MAX_CCG_ITER,
            save_results=True, output_dir="."):
    """
    Run ARO for all 6 pandemics × 3 regimes × 4 α values.
    """
    global DATA_DIR
    DATA_DIR = data_dir

    paper_ref = {
        "I":   {"SRP": {"1918":27619.4,"1928":27635.1,"1957":27650.9,
                         "1968":27680.2,"2009":27693.6,"2020":27601.2},
                "IRP": {"1918":24175.3,"1928":24100.5,"1957":23886.0,
                         "1968":24207.6,"2009":24235.9,"2020":24126.8}},
        "II":  {"SRP": {"1918":27376.8,"1928":27383.0,"1957":27416.4,
                         "1968":27392.1,"2009":27418.7,"2020":27367.5},
                "IRP": {"1918":26226.2,"1928":26207.0,"1957":26245.0,
                         "1968":26256.1,"2009":26260.6,"2020":26186.3}},
        "III": {"SRP": {"1918":27640.0,"1928":27663.7,"1957":27650.8,
                         "1968":27679.9,"2009":27702.6,"2020":27595.4},
                "IRP": {"1918":25189.9,"1928":20091.0,"1957":25271.7,
                         "1968":23637.3,"2009":25123.4,"2020":20069.7}},
    }

    print("=" * 70)
    print("  Adversarial Robust Optimisation — Singh & Rebennack (2025)")
    print(f"  Regimes {regimes}  |  α ∈ {alphas}  |  Γ = {gamma_frac*100:.0f}% of C×T")
    print(f"  {n_scenarios} scenarios per pandemic  |  seed = {seed}")
    print("=" * 70)

    all_results = {}   # (pandemic, regime, alpha) → result dict

    for regime in regimes:
        At = make_At(regime)

        for alpha in alphas:
            print(f"\n\n{'━'*70}")
            print(f"  REGIME {regime}  |  α = {alpha}")
            print(f"{'━'*70}")

            for pandemic in PANDEMIC_CODES:
                try:
                    S_all, B_all, _ = load_scenarios(
                        pandemic, n_scenarios, seed, alpha)
                except FileNotFoundError as e:
                    print(f"  [SKIP {pandemic}]: {e}")
                    continue

                res = solve_aro(pandemic, regime, S_all, B_all, At,
                                alpha=alpha, gamma_frac=gamma_frac,
                                max_iter=max_iter, verbose=True)

                if "error" not in res:
                    res["paper_SRP"] = paper_ref[regime]["SRP"].get(pandemic)
                    res["paper_IRP"] = paper_ref[regime]["IRP"].get(pandemic)
                all_results[(pandemic, regime, alpha)] = res

    _print_summary(all_results, regimes, alphas)

    if save_results:
        _save(all_results, regimes, alphas, output_dir)

    return all_results


def _print_summary(all_results, regimes, alphas):
    W = 92
    for regime in regimes:
        for alpha in alphas:
            print(f"\n{'='*W}")
            print(f"  ARO RESULTS — Regime {regime}  |  α = {alpha}")
            print(f"{'='*W}")
            print(f"  {'Pandemic':<10} {'ARO(exp)':>10} {'ARO(wc)':>9} "
                  f"{'IRP':>10} {'Gain(exp)%':>11} {'k':>4} "
                  f"{'Time':>7}  r releases")
            print(f"  {'-'*10} {'-'*10} {'-'*9} {'-'*10} {'-'*11} "
                  f"{'-'*4} {'-'*7}  {'-'*28}")
            for pandemic in PANDEMIC_CODES:
                key = (pandemic, regime, alpha)
                if key not in all_results or "error" in all_results[key]:
                    continue
                res = all_results[key]
                r_t = res["r_total_by_time"]
                nz  = [f"t{t+1}:{v/1e3:.0f}k"
                       for t,v in enumerate(r_t) if v > 0.5]
                print(f"  {pandemic:<10} {res['obj_ARO_expected']:>10.1f} "
                      f"{res['obj_ARO_worst']:>9.1f} "
                      f"{res['obj_IRP']:>10.1f} "
                      f"{res['gain_expected_pct']:>11.1f} "
                      f"{res['n_iter']:>4} "
                      f"{res['solve_time_sec']:>6.1f}s  "
                      f"{'  '.join(nz)}")

    # Cross-α table
    print(f"\n\n{'='*W}")
    print("  CROSS-α: ARO Expected Objective  (lives saved)")
    print(f"{'='*W}")
    for regime in regimes:
        print(f"\n  Regime {regime}")
        hdr = (f"  {'Pandemic':<10}"
               + "".join(f"  {'α='+str(a):>11}" for a in alphas))
        print(hdr)
        print(f"  {'-'*10}" + "".join(f"  {'-'*11}" for _ in alphas))
        for pandemic in PANDEMIC_CODES:
            row = f"  {pandemic:<10}"
            for alpha in alphas:
                key = (pandemic, regime, alpha)
                if key in all_results and "error" not in all_results[key]:
                    row += f"  {all_results[key]['obj_ARO_expected']:>11.1f}"
                else:
                    row += f"  {'N/A':>11}"
            print(row)
    print(f"{'='*W}")


def _save(all_results, regimes, alphas, output_dir):
    # JSON
    safe = {}
    for (p, r, a), v in all_results.items():
        k = f"{p}__R{r}__a{int(a*100)}"
        if "error" not in v:
            safe[k] = {kk: vv for kk, vv in v.items()
                       if kk != "convergence"}
    path = os.path.join(output_dir, "aro_results.json")
    with open(path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    print(f"\n  Full results → {path}")

    # Summary CSV
    path2 = os.path.join(output_dir, "aro_alpha_table.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "pandemic", "alpha",
                    "obj_ARO_expected", "obj_ARO_worst", "obj_IRP",
                    "gain_expected_pct", "gain_wc_pct",
                    "n_iter", "solve_time_sec",
                    "paper_SRP", "paper_IRP"])
        for regime in regimes:
            for pandemic in PANDEMIC_CODES:
                for alpha in alphas:
                    key = (pandemic, regime, alpha)
                    if key not in all_results or "error" in all_results[key]:
                        continue
                    res = all_results[key]
                    w.writerow([regime, pandemic, alpha,
                                f"{res['obj_ARO_expected']:.2f}",
                                f"{res['obj_ARO_worst']:.2f}",
                                f"{res['obj_IRP']:.2f}",
                                f"{res['gain_expected_pct']:.2f}",
                                f"{res['gain_wc_pct']:.2f}",
                                res["n_iter"],
                                f"{res['solve_time_sec']:.2f}",
                                res.get("paper_SRP",""),
                                res.get("paper_IRP","")])
    print(f"  α comparison table → {path2}")

    # r_total CSVs
    for regime in regimes:
        for alpha in alphas:
            path3 = os.path.join(output_dir,
                                  f"aro_r_R{regime}_a{int(alpha*100)}.csv")
            with open(path3, "w", newline="") as f:
                w = csv.writer(f)
                pandemics = list(PANDEMIC_CODES.keys())
                w.writerow(["t"] + pandemics)
                for t in range(N_TIMES):
                    row = [t + 1]
                    for p in pandemics:
                        key = (p, regime, alpha)
                        if key in all_results and "error" not in all_results[key]:
                            row.append(
                                f"{all_results[key]['r_total_by_time'][t]:.2f}")
                        else:
                            row.append("N/A")
                    w.writerow(row)
            print(f"  r_total (R{regime}, α={alpha}) → {path3}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIRECTORY = "/Users/hengsichen/Downloads/ResourceAllocation-main/Data"
    OUTPUT_DIR     = "."

    results = run_all(
        data_dir    = DATA_DIRECTORY,
        n_scenarios = 50,
        seed        = 42,
        regimes     = ("I", "II", "III"),
        alphas      = [0.0, 0.10, 0.15, 0.20],
        gamma_frac  = 0.10,    # 10% budget: moderate conservatism
        max_iter    = 30,
        save_results= True,
        output_dir  = OUTPUT_DIR,
    )