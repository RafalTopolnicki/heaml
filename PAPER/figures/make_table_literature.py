"""SI table of the literature alloys used for validation -> figures/table_literature.tex (cite keys as in make_bib.py).
Also numbers/init_sets.json: size and composition statistics of the initial (random) datasets."""
import re
import sys
import numpy as np
import pandas as pd
import common as C

sys.path.insert(0, str(C.ROOT / 'PAPER'))
from make_bib import literature_dois  # noqa: E402

B = C.load_baseline()
keys = {doi: key for key, doi in literature_dois().items()}


def doi_key(doi):
    d = str(doi).strip()
    if d.lower().startswith('arxiv:'):
        d = '10.48550/arXiv.' + d.split(':', 1)[1]
    return keys.get(d, '')


cells = []
for _, r in B[B.in_A | B.in_B].sort_values('Tc_exp', ascending=False, kind='stable').iterrows():
    comp = C.comp_tex(r, C.XLSX_ELEMENTS)
    flag = '$^{\\dagger}$' if r.caution else ''
    cells.append(f"{comp}{flag} & {r.Tc_exp:.2f} & {r['Tc_corr2']:.2f} & \\cite{{{doi_key(r.doi)}}}")
# two entries per table row, reading order: down the left half, then down the right half
half = (len(cells) + 1) // 2
rows = [f"{cells[i]} & {cells[i + half] if i + half < len(cells) else ' & & & '} \\\\" for i in range(half)]
(C.FIG / 'table_literature.tex').write_text('\n'.join(rows) + '\n')
print(len(cells), 'entries in', len(rows), 'rows; missing keys:', sum('cite{}' in x for x in rows))

# initial datasets
out = {}
for name, f, elements in [('A', 'sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv', C.SET_A),
                          ('B', 'sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv', C.SET_B),
                          ('A_interior', 'sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.interior.csv', C.SET_A)]:
    db = pd.read_csv(C.ROOT / 'databases' / f)
    x = db[elements].fillna(0.0).values
    n_nonzero = (x > 1e-6).sum(axis=1)
    n_sig = (x > C.RATIO).sum(axis=1)
    src = db['source'].value_counts().to_dict() if 'source' in db else {}
    tc_col = 'Tc_mu0.1' if 'Tc_mu0.1' in db else None
    out[name] = dict(n=len(db), nonzero_hist={int(k): int(v) for k, v in zip(*np.unique(n_nonzero, return_counts=True))},
                     significant_hist={int(k): int(v) for k, v in zip(*np.unique(n_sig, return_counts=True))},
                     sources=src, max_fraction_hist={'>0.7': int((x.max(axis=1) > 0.7).sum())},
                     best_Tc=float(db[tc_col].max()) if tc_col else None,
                     best_name=str(db.loc[db[tc_col].idxmax(), 'name']) if tc_col else None)
    print(name, out[name])
C.save_numbers('init_sets', out)
