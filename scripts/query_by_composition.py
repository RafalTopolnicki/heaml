#!/usr/bin/env python3
"""
Find computed HEA calculations closest to a query composition.

Usage:
    python query_by_composition.py \
        --composition "Ti0.2Nb0.2Zr0.2Hf0.2Ta0.2" \
        /path/to/random.ratios/sra.kp10.ew0.6 \
        /path/to/opt.ratios/sra.kp10.ew0.6

Composition formats accepted:
    Ti0.2Nb0.2Zr0.2Hf0.2Ta0.2    (element+fraction concatenated)
    Ti:0.2,Nb:0.2,Zr:0.2          (colon-separated pairs)

Handles both random.ratios dirs (flat composition subdirs) and
opt.ratios dirs (iteration_N/computation/ structure).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ELEMENTS = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Sc', 'Mo', 'W', 'Y', 'La', 'Re', 'Ru']
BOHR_TO_ANG = 0.529177210903
COMP_RE = re.compile(r'(?:' + '|'.join(ELEMENTS) + r')[\d.]+')
ELEM_FRAC_RE = re.compile(r'(' + '|'.join(ELEMENTS) + r')([\d.]+)')


def parse_comp_from_name(name: str) -> dict[str, float]:
    matches = ELEM_FRAC_RE.findall(name)
    return {e: float(f) for e, f in matches}


def parse_comp_from_string(s: str) -> dict[str, float]:
    if ':' in s:
        result = {}
        for part in re.split(r'[,\s]+', s.strip()):
            if not part:
                continue
            elem, frac = part.split(':')
            result[elem.strip()] = float(frac.strip())
        return result
    matches = ELEM_FRAC_RE.findall(s)
    if not matches:
        raise ValueError(f"Cannot parse composition string: {s!r}")
    return {e: float(f) for e, f in matches}


def to_vec(comp: dict[str, float]) -> np.ndarray:
    return np.array([comp.get(e, 0.0) for e in ELEMENTS])


def composition_distance(a: dict, b: dict) -> float:
    return float(np.linalg.norm(to_vec(a) - to_vec(b)))


def is_comp_dir(path: Path) -> bool:
    return path.is_dir() and bool(COMP_RE.search(path.name))


def find_composition_dirs(base: Path) -> list[Path]:
    """Yield all composition dirs from a base dir (handles random and opt layouts)."""
    iteration_computations = sorted(base.glob('iteration_*/computation'))
    if iteration_computations:
        dirs = []
        for comp_dir_parent in iteration_computations:
            dirs.extend(p for p in comp_dir_parent.iterdir() if is_comp_dir(p))
        return dirs
    return [p for p in base.iterdir() if is_comp_dir(p)]


def load_results(comp_dir: Path) -> dict | None:
    path = comp_dir / 'results.json'
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_properties(data: dict, mu: str) -> dict:
    a0_ang = data.get('a0_bohr', float('nan')) * BOHR_TO_ANG
    return {
        'Tc':     data.get(f'Tc_mu{mu}', float('nan')),
        'debye':  data.get('thetaDB_K', data.get('mixture_debye_temperature', float('nan'))),
        'a0_ang': a0_ang,
        'B0':     data.get('mixture_bulk_modulus', float('nan')),
        'lam':    data.get('lambda', float('nan')),
    }


def format_comp(comp: dict) -> str:
    return ' '.join(f'{e}{v:.4f}' for e in ELEMENTS if (v := comp.get(e, 0.0)) > 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Find computed HEA closest to a query composition.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('dirs', nargs='+', metavar='DIR',
                        help='Directories containing HEA calculations')
    parser.add_argument('--composition', '-c', required=True,
                        help='Query composition string')
    parser.add_argument('--top', '-n', type=int, default=10,
                        help='Number of results to show (default: 10)')
    parser.add_argument('--mu', default='0.1', choices=['0.1', '0.2', '0.3'],
                        help='Coulomb pseudopotential μ for Tc (default: 0.1)')
    parser.add_argument('--distance', default='l2', choices=['l2', 'l1'],
                        help='Composition distance metric (default: l2)')
    args = parser.parse_args()

    try:
        query = parse_comp_from_string(args.composition)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1

    total = sum(query.values())
    if abs(total - 1.0) > 0.02:
        print(f'Warning: composition sums to {total:.4f}, normalizing.', file=sys.stderr)
        query = {e: v / total for e, v in query.items()}

    dist_fn = composition_distance
    if args.distance == 'l1':
        def dist_fn(a, b):
            return float(np.sum(np.abs(to_vec(a) - to_vec(b))))

    records = []
    for dir_str in args.dirs:
        base = Path(dir_str)
        if not base.exists():
            print(f'Warning: {base} not found, skipping.', file=sys.stderr)
            continue
        comp_dirs = find_composition_dirs(base)
        if not comp_dirs:
            print(f'Warning: no composition dirs in {base}, skipping.', file=sys.stderr)
            continue
        for comp_dir in comp_dirs:
            comp = parse_comp_from_name(comp_dir.name)
            if not comp:
                continue
            data = load_results(comp_dir)
            if data is None:
                continue
            records.append({
                'dist':  dist_fn(query, comp),
                'comp':  comp,
                'props': extract_properties(data, args.mu),
                'path':  comp_dir,
            })

    if not records:
        print('No results found.', file=sys.stderr)
        return 1

    records.sort(key=lambda r: r['dist'])
    top = records[:args.top]

    # Header
    w = [5, 10, 10, 9, 10, 9, 8]
    sep = '  '
    hdr = (f"{'Rank':<{w[0]}}{sep}"
           f"{'Tc(K)':<{w[1]}}{sep}"
           f"{'Debye(K)':<{w[2]}}{sep}"
           f"{'a0(Å)':<{w[3]}}{sep}"
           f"{'B0(GPa)':<{w[4]}}{sep}"
           f"{'lambda':<{w[5]}}{sep}"
           f"{'dist':<{w[6]}}{sep}"
           f"Composition")
    print(f'\nQuery: {format_comp(query)}  (mu={args.mu}, top={args.top})\n')
    print(hdr)
    print('-' * (len(hdr) + 40))

    for i, r in enumerate(top, 1):
        p = r['props']
        print(f"{i:<{w[0]}}{sep}"
              f"{p['Tc']:<{w[1]}.3f}{sep}"
              f"{p['debye']:<{w[2]}.2f}{sep}"
              f"{p['a0_ang']:<{w[3]}.4f}{sep}"
              f"{p['B0']:<{w[4]}.2f}{sep}"
              f"{p['lam']:<{w[5]}.4f}{sep}"
              f"{r['dist']:<{w[6]}.4f}{sep}"
              f"{format_comp(r['comp'])}")
        print(f"      {r['path']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
