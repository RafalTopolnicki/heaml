"""Fig. 3: composition of the champion of each mincomp run (stacked bars), set A and set B (corr2)."""
import numpy as np
import matplotlib.pyplot as plt
import style
import common as C

series = {'A': (C.load_series('A'), C.SET_A), 'B': (C.load_series('B_corr2'), C.SET_B)}
fig, ax = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.9))
for a, (name, (runs, elements)) in zip(ax, series.items()):
    champs = [C.champion(r) for r in runs]
    labels = [f"$k\\geq{r['mc']}$: {ch.Tc:.1f} K" for r, ch in zip(runs, champs)]
    order = sorted(elements, key=lambda el: -np.mean([ch[el] for ch in champs]))   # majority elements first
    left = np.zeros(len(champs))
    for el in order:
        vals = np.array([ch[el] for ch in champs])
        a.barh(labels, vals, left=left, color=style.EL_COLORS[el], edgecolor='white', lw=0.3, label=el)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.045:
                a.text(l + v / 2, i, el, ha='center', va='center', fontsize=5.5,
                       color='white' if el in ('Nb', 'Ta', 'Hf', 'Re', 'Mo') else 'k')
        left += vals
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel='atomic fraction', title=f'({"ab"[list(series).index(name)]}) set {name}')
    a.tick_params(axis='y', length=0)
plt.tight_layout()
style.save(fig, 'fig_champions')
