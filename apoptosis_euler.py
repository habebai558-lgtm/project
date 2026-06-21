"""
Apoptosis ODE Model - Euler Method Solver
Course: SBEG108 Numerical Methods in Biomedical Engineering
Chapter 3: Apoptosis (Schiesser, 2014)

3 Cases implemented:
  Case 1: Base IC (y_hif=1, rest=0), constant a12
  Case 2: Zero IC  (all=0),          constant a12
  Case 3: Zero IC  (all=0),          a12*exp(-0.1*t)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────
#  Parameters (Table 3.3)
# ─────────────────────────────────────────
p = {
    'a_hif': 1.52, 'a_o2': 1.8,  'a_p53': 0.05,
    'a3':    0.9,  'a4':   0.2,   'a5':    0.001,
    'a7':    0.7,  'a8':   0.06,  'a9':    0.1,
    'a10':   0.7,  'a11':  0.2,   'a12':   0.1,
    'a13':   0.1,  'a14':  0.05,
}

# ─────────────────────────────────────────
#  Book reference values (Tables 3.4, 3.5, 3.6)
#  [y_hif, y_o2, y_p300, y_p53, y_casp, y_kp]
# ─────────────────────────────────────────
book_ref = {
    1: {  # Case 1 - Table 3.4
        0.0:   [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5420, 2.6645, 0.4706, 0.4576, 1.2735, 0.5706],
        50.0:  [0.5038, 2.8407, 0.5771, 0.4940, 1.4708, 0.5267],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 1.4968, 0.5219],
    },
    2: {  # Case 2 - Table 3.5
        0.0:   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5392, 2.6779, 0.4800, 0.4575, 1.2735, 0.5740],
        50.0:  [0.5036, 2.8419, 0.5783, 0.4940, 1.4708, 0.5269],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 1.4968, 0.5219],
    },
    3: {  # Case 3 - Table 3.6 ncase=2
        0.0:   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5392, 2.6779, 0.4800, 0.4575, 0.5608, 1.1832],
        50.0:  [0.5036, 2.8419, 0.5783, 0.4940, 0.5112, 1.3837],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 0.4973, 1.4390],
    },
}

labels = ['y_hif', 'y_o2', 'y_p300', 'y_p53', 'y_casp', 'y_kp']
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#00BCD4']
titles = ['HIF', 'Oxygen', 'p300', 'p53', 'Caspase', 'Potassium']
case_names = {
    1: 'Case 1: Base IC (y_hif=1), constant a12',
    2: 'Case 2: Zero IC, constant a12',
    3: 'Case 3: Zero IC, a12*exp(-0.1t)',
}

# ─────────────────────────────────────────
#  ODE system
# ─────────────────────────────────────────
def apoptosis(y, t, p, ncase):
    hif, o2, p300, p53, casp, kp = y
    dhif  = p['a_hif'] - p['a3']*o2*hif   - p['a4']*hif*p300  - p['a7']*p53*hif
    do2   = p['a_o2']  - p['a3']*o2*hif   + p['a4']*hif*p300  - p['a11']*o2
    dp300 = -p['a4']*hif*p300 - p['a5']*p300*p53 + p['a8']
    dp53  = p['a_p53'] - p['a5']*p300*p53 - p['a9']*p53
    # Case 3: a12 varies exponentially with t
    if ncase == 3:
        dcasp = p['a9']*p53 + p['a12']*np.exp(-0.1*t) - p['a13']*casp
    else:
        dcasp = p['a9']*p53 + p['a12'] - p['a13']*casp
    dkp   = -p['a10']*casp*kp + p['a11']*o2 - p['a14']*kp
    return np.array([dhif, do2, dp300, dp53, dcasp, dkp])

# ─────────────────────────────────────────
#  Euler solver
# ─────────────────────────────────────────
def euler_solve(ncase):
    # Initial conditions
    if ncase == 1:
        y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    h  = 0.01
    t0, tf = 0.0, 100.0
    t_arr = np.linspace(t0, tf, int((tf - t0) / h) + 1)
    N = len(t_arr)

    y_arr = np.zeros((N, 6))
    y_arr[0] = y0

    start = time.perf_counter()
    for i in range(N - 1):
        y_arr[i+1] = y_arr[i] + h * apoptosis(y_arr[i], t_arr[i], p, ncase)
    elapsed = time.perf_counter() - start

    return t_arr, y_arr, elapsed

# ─────────────────────────────────────────
#  Run all 3 cases
# ─────────────────────────────────────────
results = {}
for nc in [1, 2, 3]:
    t_arr, y_arr, elapsed = euler_solve(nc)
    results[nc] = {'t': t_arr, 'y': y_arr, 'time': elapsed}
    print(f"Case {nc} done — {elapsed*1000:.1f} ms")

# ─────────────────────────────────────────
#  Plot 1: All 6 variables for each case (3 figures)
# ─────────────────────────────────────────
for nc in [1, 2, 3]:
    t_arr = results[nc]['t']
    y_arr = results[nc]['y']
    ref   = book_ref[nc]
    ref_t = list(ref.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Apoptosis — Euler Method\n{case_names[nc]}', fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ref_vals = [ref[t][i] for t in ref_t]
        ax.plot(t_arr, y_arr[:, i], color=colors[i], lw=1.5, label='Euler')
        ax.scatter(ref_t, ref_vals, color='black', s=60, zorder=5, marker='D', label='Book RKF45')
        ax.set_title(titles[i])
        ax.set_xlabel('t')
        ax.set_ylabel(labels[i])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────
#  Plot 2: Compare all 3 cases on same axes (one plot per variable)
# ─────────────────────────────────────────
case_colors = ['blue', 'green', 'red']
case_styles = ['-', '--', '-.']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Apoptosis — Euler Method: All 3 Cases Compared', fontsize=13, fontweight='bold')

for i, ax in enumerate(axes.flat):
    for nc in [1, 2, 3]:
        t_arr = results[nc]['t']
        y_arr = results[nc]['y']
        ax.plot(t_arr, y_arr[:, i],
                color=case_colors[nc-1],
                linestyle=case_styles[nc-1],
                lw=1.5, label=f'Case {nc}')
    ax.set_title(titles[i])
    ax.set_xlabel('t')
    ax.set_ylabel(labels[i])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
#  Print comparison tables
# ─────────────────────────────────────────
check_times = [0.0, 25.0, 50.0, 100.0]
h = 0.01

for nc in [1, 2, 3]:
    y_arr = results[nc]['y']
    ref   = book_ref[nc]
    print(f"\n{'='*70}")
    print(f"  {case_names[nc]}")
    print(f"  Euler runtime: {results[nc]['time']*1000:.2f} ms")
    print(f"{'='*70}")
    for tc in check_times:
        idx = int(round(tc / h))
        euler_v = y_arr[idx]
        ref_v   = ref[tc]
        print(f"\n  t = {tc:.1f}")
        print(f"  {'Var':<8} {'Euler':>10} {'Book RKF45':>12} {'Abs Error':>12}")
        print(f"  {'-'*44}")
        for j in range(6):
            err = abs(euler_v[j] - ref_v[j])
            print(f"  {labels[j]:<8} {euler_v[j]:>10.4f} {ref_v[j]:>12.4f} {err:>12.4e}")
