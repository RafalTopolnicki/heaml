#!/usr/bin/env python3
"""Composition list for the bcc/fcc/hcp comparison (scripts/structure_compare.py) of the optimisation champions.

Run on the workstation (needs opt.ratios.gl/), commit the CSV, run the array on Topola (slurm/submit_td_champions.sh).

Champion of a run = composition with the highest target among the compositions with >= mincomp elements above 5 %
(the rule of utils.log_iteration_summary and of the OptGL_* notebooks), over init + all iterations.

  python scripts/td_champion_list.py                        # default runs (below) -> slurm/compositions/td_champions_gl.csv
  python scripts/td_champion_list.py --runs '<glob>:<target>' ...

Columns: tag, run, target, Tc, n_el, elements, concentrations (space-separated, only fractions > 0), outdir, dirname
(= the directory structure_compare.py creates under outdir; the array task skips it when structure_summary.json exists).
The pure elements (references for formation energies, all 12 incl. V and Sc) are added as rows with outdir
thermodynamic/elements.sra.ew0.6.topola (same binary and settings as the champions).
"""
import argparse
import glob
import json
import os
import re
import sys
import tarfile

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from src.elements import HEAClass            # noqa: E402
from src.utils import generate_dirname       # noqa: E402

RATIO = 0.05
DEFAULT_RUNS = [
    # set A, GL, Tc_mu0.1, mincomp 2..10, beta 2
    ('opt.ratios.gl/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp*.r0.05.max0.7.face0.1', 'Tc_mu0.1'),
    # set B, GL, Tc_corr (corr2), mincomp 2..10, beta 2
    ('opt.ratios.gl/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.corr2.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp*.r0.05.max0.7.face0.1', 'Tc_corr'),
    # set B, GL, Tc_mu0.1 (no correction): whatever has finished so far
    ('opt.ratios.gl/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp*.r0.05.max0.7.face0.1', 'Tc_mu0.1'),
]
ELEMENT_REFS = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru', 'V', 'Sc']


def parse_name(name):
    return {el: float(x) for el, x in re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', name)}


def iter_results(it_dir):
    tf = os.path.join(it_dir, 'computation.tar')
    if os.path.exists(tf):
        with tarfile.open(tf) as t:
            for m in t.getmembers():
                if m.name.endswith('results.json'):
                    yield json.load(t.extractfile(m))
        return
    for rf in glob.glob(os.path.join(it_dir, 'computation', '*', 'results.json')):
        with open(rf) as f:
            yield json.load(f)


def champion(run_dir, target):
    params = json.load(open(os.path.join(run_dir, 'parameters.json')))
    mc = params['min_components']
    elements = re.findall(r'[A-Z][a-z]?', os.path.basename(run_dir).split('.')[1])   # sra.<TiNbZr...>.kp10...
    init = pd.read_csv(os.path.join(run_dir, 'init_tc_comparison.csv'))
    cand = [(parse_name(n), tc) for n, tc in zip(init['name'], init[f'{target}_computed'])]
    for it_dir in glob.glob(os.path.join(run_dir, 'iteration_*')):
        for r in iter_results(it_dir):
            cand.append(({el: float(r.get(el) or 0.0) for el in elements}, r.get(target)))
    best = None
    for comp, tc in cand:
        if tc is None or pd.isna(tc):
            continue
        if sum(v > RATIO for v in comp.values()) < mc:
            continue
        if best is None or tc > best[1]:
            best = (comp, tc)
    early = 'EARLY STOP' in open(os.path.join(run_dir, 'optimization_log.txt')).read()
    return mc, best, early


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', help="'<glob>:<target>' (default: set A GL, set B corr2, set B mu0.1)")
    ap.add_argument('--out', default='slurm/compositions/td_champions_gl.csv')
    ap.add_argument('--outroot', default='thermodynamic')
    ap.add_argument('--elements_outdir', default='elements.sra.ew0.6.topola',
                    help='pure-element references; separate from the workstation elements.sra.ew0.6 so that an rsync back '
                         'does not overwrite them (and both can be compared)')
    ap.add_argument('--include_running', action='store_true', help='also runs without EARLY STOP (default: skipped)')
    args = ap.parse_args()
    runs = [tuple(s.rsplit(':', 1)) for s in args.runs] if args.runs else DEFAULT_RUNS

    out = []
    for pattern, target in runs:
        for run_dir in sorted(glob.glob(pattern), key=lambda d: int(re.search(r'mincomp(\d+)', d).group(1))):
            mc, best, early = champion(run_dir, target)
            name = os.path.basename(run_dir)
            if not early and not args.include_running:
                print(f'[skip, no early stop] {name}')
                continue
            comp, tc = best
            els = [el for el, v in comp.items() if v > 0]
            conc = [comp[el] for el in els]
            hea = HEAClass(labels=els, concentrations=conc)
            setname = 'A' if 'ReRu' in name else 'B'
            variant = 'corr2' if '.corr2.' in name else 'mu0.1'
            out.append(dict(tag=f'{setname}_{variant}_mc{mc}', run=name, target=target, Tc=round(tc, 4),
                            n_el=sum(v > RATIO for v in conc), elements=' '.join(els),
                            concentrations=' '.join(f'{c:.6f}' for c in conc),
                            outdir=f'{args.outroot}/{name}', dirname=generate_dirname(els, hea.concentrations)))
            print(f"{out[-1]['tag']:>16}  {target:>8} {tc:6.2f} K  {out[-1]['dirname']}")
    for el in ELEMENT_REFS:
        out.append(dict(tag=f'el_{el}', run='', target='', Tc=float('nan'), n_el=1, elements=el, concentrations='1.000000',
                        outdir=f'{args.outroot}/{args.elements_outdir}', dirname=generate_dirname([el], [1.0])))
    df = pd.DataFrame(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'{len(df)} rows ({(df.run != "").sum()} champions + {len(ELEMENT_REFS)} elements) -> {args.out}')


if __name__ == '__main__':
    main()
