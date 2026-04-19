"""
Sensitivity Analysis — PH-v7: Scenario Count vs Solve Time / Objective
=======================================================================
For each pandemic (6) × each alpha (4) → 2 figures:
  Fig 1: Solve time  vs n_scenarios  (3 lines = Regime I/II/III)
  Fig 2: Objective   vs n_scenarios  (3 lines = Regime I/II/III)

Total figure count: 6 pandemics × 4 alphas × 2 figures = 48 figures.
Figures are grouped into per-pandemic PDF pages for readability.

Scenario grid: [5, 10, 20, 30, 40, 50]
Alpha values:  [0.0, 0.10, 0.15, 0.20]

Methodological notes
--------------------
1. The IRP baseline (compute_irp) is a population-proportional allocation —
   this correctly replicates Singh & Rebennack (2025)'s IRP definition.
   Do NOT change it to an optimised variant.

2. Objectives reported are in-sample (optimised and evaluated on the same
   n scenarios). The paper's SRP uses the full 1200-scenario GAMS solution.
   The gain% vs IRP is valid WITHIN each (alpha, n) cell but NOT comparable
   across alpha levels because apply_alpha() scales S by (1+alpha), degrading
   IRP's proportional allocation more than PH's hedge-aware allocation.

3. At n=5/10, the plateau detector typically fires (algorithm exits without
   reaching 80% slamming). These points are plotted with a different marker
   (open circle) to visually distinguish the two stopping-regime behaviours.

4. For the sensitivity narrative: focus on n >= 20 (convergence regime).
   n=5/10 are "insufficient scenario" regime and should be noted as such.

Usage
-----
    python sensitivity_by_alpha.py

Outputs (in OUTPUT_DIR):
  - sensitivity_alpha{a}_{pandemic}_time.{pdf|png}   — time figures
  - sensitivity_alpha{a}_{pandemic}_obj.{pdf|png}    — objective figures
  - sensitivity_all_results.json                      — raw data cache
  - sensitivity_summary_table.csv                     — tabular summary
"""

import os, sys, time, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PH_Code_final import (
    load_scenarios, make_At, ph_v7,
    PANDEMIC_CODES, DATA_DIR,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_GRID = [5, 10, 20, 30, 40, 50]
ALPHA_VALUES  = [0.0, 0.10, 0.15, 0.20]
REGIMES       = ["I", "II", "III"]
RANDOM_SEED   = 42
OUTPUT_DIR    = "."

# ── Styling ───────────────────────────────────────────────────────────────────
REGIME_COLORS  = {"I": "#185FA5", "II": "#993C1D", "III": "#3B6D11"}
REGIME_LABELS  = {"I": "Regime I",  "II": "Regime II", "III": "Regime III"}
MARKERS_SOLID  = {"I": "o", "II": "s", "III": "^"}   # n>=20 (slamming exits)
MARKERS_OPEN   = {"I": "o", "II": "s", "III": "^"}   # n<=10 (plateau exits)
LINEWIDTH      = 1.8
MARKERSIZE     = 6
FIGSIZE        = (5.5, 4.0)

PANDEMIC_NAMES = {
    "1918": "1918 Spanish Flu",  "1928": "1928 Pandemic",
    "1957": "1957 Asian Flu",    "1968": "1968 Hong Kong Flu",
    "2009": "2009 Swine Flu",    "2020": "2020 COVID-19",
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER — load or compute all results
# ═══════════════════════════════════════════════════════════════════════════════

def _key(pandemic, regime, n, alpha):
    return f"{pandemic}__R{regime}__n{n}__a{int(alpha*100):02d}"


def run_sensitivity(data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                    scenario_grid=SCENARIO_GRID, alpha_values=ALPHA_VALUES,
                    seed=RANDOM_SEED, n_workers=None, verbose=True,
                    cache_path=None):
    """
    Run ph_v7 for all (pandemic, regime, n_scenarios, alpha) combinations.
    Results are cached to JSON so re-runs skip already-computed cells.

    Returns: dict keyed by _key(pandemic, regime, n, alpha)
    """
    if cache_path is None:
        cache_path = os.path.join(output_dir, "sensitivity_all_results.json")

    # Load existing cache
    results = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)
        if verbose:
            print(f"  Loaded {len(results)} cached results from {cache_path}")

    total = (len(PANDEMIC_CODES) * len(REGIMES)
             * len(scenario_grid) * len(alpha_values))
    done = 0

    for alpha in alpha_values:
        for pandemic in PANDEMIC_CODES:
            for regime in REGIMES:
                At = make_At(regime)
                for n_scen in scenario_grid:
                    k = _key(pandemic, regime, n_scen, alpha)
                    done += 1

                    if k in results:
                        if verbose:
                            print(f"  [{done}/{total}] SKIP (cached)  "
                                  f"{pandemic} R{regime} n={n_scen} α={alpha}")
                        continue

                    if verbose:
                        print(f"\n  [{done}/{total}]  "
                              f"pandemic={pandemic}  regime={regime}  "
                              f"n={n_scen}  α={alpha}")
                    try:
                        S, B = load_scenarios(pandemic, n=n_scen,
                                              seed=seed, alpha=alpha)
                        P = np.ones(n_scen) / n_scen

                        res = ph_v7(pandemic, regime, S, B, At, P,
                                    n_workers=n_workers,
                                    use_gurobi=True, verbose=False)

                        results[k] = {
                            "pandemic":    pandemic,
                            "regime":      regime,
                            "n_scenarios": n_scen,
                            "alpha":       alpha,
                            "obj_PH":      res["obj_PH"],
                            "obj_IRP":     res["obj_IRP"],
                            "gain_pct":    res["gain_pct"],
                            "solve_time":  res["elapsed"],
                            "n_iter":      res["n_iter"],
                            "fix_pct":     res["fix_pct_final"],
                            "stop_reason": res["stop_reason"],
                        }

                        # Save after every cell (safe against crashes)
                        with open(cache_path, "w") as f:
                            json.dump(results, f, indent=2, default=str)

                        if verbose:
                            sr = res["stop_reason"][:30]
                            print(f"  → obj={res['obj_PH']:.1f}  "
                                  f"time={res['elapsed']:.1f}s  "
                                  f"k={res['n_iter']}  "
                                  f"fix={res['fix_pct_final']:.1%}  "
                                  f"[{sr}]")

                    except Exception as e:
                        print(f"  [ERROR {k}]: {e}")
                        results[k] = None

    # Final save
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  All results saved → {cache_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#BBBBBB")
    ax.set_axisbelow(True)


def _is_plateau(stop_reason):
    """True if the run exited via plateau (not slamming)."""
    return stop_reason is not None and "plateau" in stop_reason.lower()


def _get_series(results, pandemic, regime, alpha, field, scenario_grid):
    """
    Extract (x_values, y_values, plateau_mask) for one (pandemic, regime, alpha).
    plateau_mask[i] is True when point i came from a plateau exit.
    """
    xs, ys, plateau = [], [], []
    for n in scenario_grid:
        k = _key(pandemic, regime, n, alpha)
        v = results.get(k)
        if v is None:
            continue
        xs.append(n)
        ys.append(v[field])
        plateau.append(_is_plateau(v.get("stop_reason", "")))
    return xs, ys, plateau


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING — one call per (pandemic, alpha) → 2 figures
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_one_pair(results, pandemic, alpha, field, ylabel, title_suffix,
                   scenario_grid, output_dir, legend_loc="upper left"):
    """
    Create one figure for (pandemic, alpha, field) with 3 lines (regimes).
    Solid markers = slamming exit (n>=20 typically).
    Open markers  = plateau exit  (n<=10 typically).

    Returns list of saved file paths.
    """
    pname = PANDEMIC_NAMES.get(pandemic, pandemic)
    a_str = f"α={alpha:.2f}"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _spine_style(ax)

    for regime in REGIMES:
        xs, ys, plateau = _get_series(results, pandemic, regime,
                                      alpha, field, scenario_grid)
        if not xs:
            continue

        col = REGIME_COLORS[regime]
        lbl = REGIME_LABELS[regime]

        # Draw the connecting line first
        ax.plot(xs, ys, color=col, linewidth=LINEWIDTH,
                label=lbl, zorder=2)

        # Draw markers: solid fill for slamming, open for plateau
        for xi, yi, is_plt in zip(xs, ys, plateau):
            mk = MARKERS_SOLID[regime]
            if is_plt:
                # Open marker for plateau exits
                ax.plot(xi, yi, marker=mk, color=col,
                        markersize=MARKERSIZE, markerfacecolor="white",
                        markeredgewidth=1.5, zorder=4)
            else:
                # Filled marker for slamming exits
                ax.plot(xi, yi, marker=mk, color=col,
                        markersize=MARKERSIZE, zorder=4)

    ax.set_xlabel("Number of scenarios", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{pname}  ({a_str})\n{title_suffix}",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xticks(scenario_grid)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Legend: regime lines
    handles, labels = ax.get_legend_handles_labels()
    # Add marker-style legend entries
    solid_patch = Line2D([0], [0], marker="o", color="gray",
                         markersize=5, markerfacecolor="gray",
                         linestyle="None", label="Slamming exit")
    open_patch  = Line2D([0], [0], marker="o", color="gray",
                         markersize=5, markerfacecolor="white",
                         markeredgewidth=1.5, linestyle="None",
                         label="Plateau exit (n≤10)")
    ax.legend(handles=handles + [solid_patch, open_patch],
              labels=labels + ["Slamming exit", "Plateau exit (n≤10)"],
              fontsize=7, framealpha=0.8, edgecolor="#CCCCCC",
              loc=legend_loc, ncol=1)

    fig.tight_layout(pad=1.5)

    alpha_tag = f"a{int(alpha*100):02d}"
    stem = f"sensitivity_{alpha_tag}_{pandemic}_{field}"
    saved = []
    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


def plot_all(results, scenario_grid=SCENARIO_GRID, alpha_values=ALPHA_VALUES,
             output_dir=OUTPUT_DIR, verbose=True):
    """
    Generate all (pandemic, alpha) pairs → 2 figures each.
    Total: 6 × 4 × 2 = 48 figures.
    """
    saved_all = []

    for alpha in alpha_values:
        for pandemic in PANDEMIC_CODES:

            # --- Figure 1: Solve time ---
            paths = _plot_one_pair(
                results, pandemic, alpha,
                field="solve_time",
                ylabel="Solve time (seconds)",
                title_suffix="Solve time vs scenario count",
                scenario_grid=scenario_grid,
                output_dir=output_dir,
                legend_loc="upper left",
            )
            saved_all.extend(paths)

            # --- Figure 2: PH-v7 Objective ---
            paths = _plot_one_pair(
                results, pandemic, alpha,
                field="obj_PH",
                ylabel="Expected objective (PH-v7)",
                title_suffix="Objective value vs scenario count",
                scenario_grid=scenario_grid,
                output_dir=output_dir,
                legend_loc="upper right",
            )
            saved_all.extend(paths)

            if verbose:
                print(f"  Plotted: pandemic={pandemic}  α={alpha}")

    return saved_all


# ═══════════════════════════════════════════════════════════════════════════════
# CSV SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(results, output_dir=OUTPUT_DIR,
             scenario_grid=SCENARIO_GRID, alpha_values=ALPHA_VALUES):
    path = os.path.join(output_dir, "sensitivity_summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pandemic", "regime", "alpha", "n_scenarios",
                    "obj_PH", "obj_IRP", "gain_pct",
                    "solve_time_s", "n_iter", "fix_pct", "stop_reason"])
        for alpha in alpha_values:
            for regime in REGIMES:
                for pandemic in PANDEMIC_CODES:
                    for n in scenario_grid:
                        k = _key(pandemic, regime, n, alpha)
                        v = results.get(k)
                        if v is None:
                            continue
                        w.writerow([
                            pandemic, regime, alpha, n,
                            f"{v['obj_PH']:.2f}",
                            f"{v['obj_IRP']:.2f}",
                            f"{v['gain_pct']:.2f}",
                            f"{v['solve_time']:.2f}",
                            v["n_iter"],
                            f"{v['fix_pct']:.3f}",
                            v.get("stop_reason", ""),
                        ])
    print(f"  CSV summary → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY — quick check on key patterns
# ═══════════════════════════════════════════════════════════════════════════════

def print_console_summary(results, scenario_grid=SCENARIO_GRID,
                          alpha_values=ALPHA_VALUES):
    """
    Print a compact overview: for each alpha, show avg obj and time
    at n=20 and n=50 across all pandemics and regimes.
    Highlights the two stopping regimes.
    """
    print("\n" + "="*80)
    print("  Quick summary: avg obj / avg time  by alpha × n")
    print("  (across all 6 pandemics × 3 regimes = 18 cells per row)")
    print("="*80)
    hdr = f"  {'α':>5}  {'n':>4}  {'avg_obj':>10}  {'avg_time_s':>10}  " \
          f"{'plateau_exits':>14}  {'slamming_exits':>15}"
    print(hdr)
    print("  " + "-"*75)

    for alpha in alpha_values:
        for n in scenario_grid:
            objs, times, plat, slam = [], [], 0, 0
            for pandemic in PANDEMIC_CODES:
                for regime in REGIMES:
                    k = _key(pandemic, regime, n, alpha)
                    v = results.get(k)
                    if v is None:
                        continue
                    objs.append(v["obj_PH"])
                    times.append(v["solve_time"])
                    if _is_plateau(v.get("stop_reason", "")):
                        plat += 1
                    else:
                        slam += 1
            if not objs:
                continue
            avg_obj  = sum(objs)  / len(objs)
            avg_time = sum(times) / len(times)
            print(f"  {alpha:>5.2f}  {n:>4d}  "
                  f"{avg_obj:>10.1f}  {avg_time:>10.1f}  "
                  f"{'plateau: '+str(plat):>14}  "
                  f"{'slamming: '+str(slam):>15}")

    print("="*80)
    print("  NOTE: objective values are in-sample (optimised and evaluated")
    print("  on the same n scenarios). gain% vs IRP is valid within each")
    print("  (alpha, n) cell but NOT comparable across alpha values.")
    print("="*80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 72)
    print("  PH-v7 Sensitivity Analysis: Time + Objective vs Scenario Count")
    print(f"  Scenarios: {SCENARIO_GRID}")
    print(f"  Alpha:     {ALPHA_VALUES}")
    print(f"  Regimes:   {REGIMES}")
    print(f"  Pandemics: {list(PANDEMIC_CODES.keys())}")
    total = (len(PANDEMIC_CODES) * len(REGIMES)
             * len(SCENARIO_GRID) * len(ALPHA_VALUES))
    print(f"  Total solver runs: {total}")
    print(f"  Figures to produce: {len(PANDEMIC_CODES) * len(ALPHA_VALUES) * 2}")
    print("=" * 72)

    # Step 1: Run / load cached results
    results = run_sensitivity(
        data_dir      = DATA_DIR,
        output_dir    = OUTPUT_DIR,
        scenario_grid = SCENARIO_GRID,
        alpha_values  = ALPHA_VALUES,
        seed          = RANDOM_SEED,
        n_workers     = None,    # None = all cores; set 1 for serial debug
        verbose       = True,
    )

    # Step 2: Console summary
    print_console_summary(results, SCENARIO_GRID, ALPHA_VALUES)

    # Step 3: CSV
    save_csv(results, OUTPUT_DIR, SCENARIO_GRID, ALPHA_VALUES)

    # Step 4: Figures
    print("\nGenerating figures...")
    saved = plot_all(results, SCENARIO_GRID, ALPHA_VALUES, OUTPUT_DIR,
                     verbose=True)
    print(f"\nSaved {len(saved)} figure files:")
    for p in saved[:12]:
        print(f"  {p}")
    if len(saved) > 12:
        print(f"  ... and {len(saved)-12} more.")