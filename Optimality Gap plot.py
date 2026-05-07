import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Data

# PH optimality gaps [pandemic order: 1918,1928,1957,1968,2009,2020]
ph = {
    'I':   np.array([[10.65, 5.24, 2.74, 4.46, 6.76, 8.21],
                     [ 2.81, 2.79, 3.34, 3.22, 3.25, 3.36],
                     [ 2.84, 3.21, 3.09, 3.06, 3.01, 3.14],
                     [ 2.81, 2.79, 3.46, 2.84, 3.67, 3.00]]),
    'II':  np.array([[ 7.53, 7.41, 7.78, 7.38, 7.15, 7.87],
                     [ 7.15, 7.58, 8.51, 7.79, 8.94, 9.38],
                     [ 9.05, 6.83, 9.68, 7.69, 7.70, 8.68],
                     [ 7.41, 8.47, 8.91, 7.50, 7.65,10.64]]),
    'III': np.array([[ 4.00, 4.10, 3.88, 3.91, 4.08, 4.62],
                     [ 3.59, 3.88, 3.52, 3.64, 3.90, 3.38],
                     [ 3.71, 3.60, 3.86, 3.62, 4.01, 3.67],
                     [ 3.47, 3.24, 3.79, 3.58, 3.72, 3.27]]),
}

# ARO optimality gaps — updated with new results
aro = {
    'I':   np.array([
                     [0.5, 2.01, 0.91, 0.72, 1.18, 1.49],
                     [0.45, 1.58, 0.99, 1.01, 1.07, 0.69],
                     [0.88, 0.41, 0.96, 1.21, 1.11, 0.86],
                     [1.73, 0.78, 1.2, 0.91, 2.29, 1.48],
                    ]),
    'II':  np.array([
                     [0.1, 0.33, 0.78, 0.17, 0.23, 0.33],
                     [0.23, 0.47, 0.5, 0.94, 1.1, 0.17],
                     [0.0, 0.52, 0.78, 1.37, 1.18, 0.1],
                     [0.59, 0.61, 1.67, 1.41, 1.34, 0.71],
                    ]),
    'III': np.array([
                     [0.23, 0.6, 0.86, 0.68, 0.38, 1.45],
                     [0.13, 1.75, 1.07, 0.97, 1.18, 0.54],
                     [0.82, 1.9, 1.15, 2.68, 0.59, 0.61],
                     [0.69, 0.71, 0.96, 1.34, 1.99, 0.62],
                    ]),
}

alphas     = [0, 1, 2, 3]
alpha_lbls = [r'$\alpha=0$', r'$\alpha=0.10$', r'$\alpha=0.15$', r'$\alpha=0.20$']
regimes    = ['I', 'II', 'III']
colours    = {'I': '#2166AC', 'II': '#D6604D', 'III': '#1A9850'}
jitter_sd  = 0.07

# Legend
legend_handles = []
for reg in regimes:
    legend_handles.append(
        mlines.Line2D([], [],
                      color=colours[reg], linestyle='-',
                      marker='o', markersize=7, linewidth=2,
                      label=f'Regime {reg}')
    )
legend_handles.append(
    mlines.Line2D([], [],
                  color='grey', linestyle='none',
                  marker='o', markersize=6, alpha=0.5,
                  label='Individual pandemic')
)

# Plotting function
def make_plot(data, title, ymax, yticks, ax):
    rng = np.random.default_rng(42)
    for reg in regimes:
        mat   = data[reg]
        means = mat.mean(axis=1)
        col   = colours[reg]

        for ai in alphas:
            jitter = rng.normal(0, jitter_sd, mat.shape[1])
            ax.scatter(
                np.full(mat.shape[1], ai) + jitter,
                mat[ai],
                color=col, alpha=0.30, s=28, zorder=2,
                linewidths=0
            )

        ax.plot(
            alphas, means,
            color=col, linestyle='-',
            marker='o', markersize=7,
            linewidth=2, zorder=3
        )

    ax.set_xticks(alphas)
    ax.set_xticklabels(alpha_lbls, fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{v:.1f}%' for v in yticks], fontsize=10)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel('Optimality gap (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.grid(axis='y', color='grey', alpha=0.2, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# PH figure
fig1, ax1 = plt.subplots(figsize=(7, 5))
make_plot(ph, 'SRP Optimality Gap: Exact vs PH', ymax=13, yticks=[0,2,4,6,8,10,12], ax=ax1)
fig1.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=4,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08)
)
fig1.savefig('optimality_gap_ph.pdf', bbox_inches='tight', dpi=300)
fig1.savefig('optimality_gap_ph.png', bbox_inches='tight', dpi=300)

# ARO figure
fig2, ax2 = plt.subplots(figsize=(7, 5))
make_plot(aro, 'SRP Optimality Gap: Exact vs ARO', ymax=3.5, yticks=[0,0.5,1.0,1.5,2.0,2.5,3.0], ax=ax2)
fig2.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=4,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08)
)
fig2.savefig('optimality_gap_aro.pdf', bbox_inches='tight', dpi=300)
fig2.savefig('optimality_gap_aro.png', bbox_inches='tight', dpi=300)

print("Done.")