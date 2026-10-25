"""Composition lists for random datasets: generate once, commit, compute from the file.

The list (CSV: idx, family, seed, dirname, one column per element) is the single source of truth for a dataset run.
SLURM tasks take disjoint row ranges (run_random_compositions.py --composition_file FILE --sample_offset A
--number_of_samples N), so nothing depends on the random generator at run time, and missing compositions are simply
rows without results.json.

  generate: python scripts/composition_list.py generate --elements Ti,Nb,... --out FILE \
                --full N:SEED [--face k:N:SEED ...] [--min_nonzero_fraction 0.05]
            Same generators as run_random_compositions.py: full-simplex sample i uses seed SEED+i (flat Dirichlet);
            k-face sample i uses subset face_subset(n, k, i, SEED) (fixed random order of all C(n,k) subsets) and seed
            SEED+i for the flat Dirichlet on the subset (fractions >= min_nonzero_fraction).
  missing:  python scripts/composition_list.py missing FILE WORKDIR --out FILE2   rows of FILE without WORKDIR/<dirname>/results.json
  status:   python scripts/composition_list.py status FILE WORKDIR                 done / missing per family
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_random_compositions import generate_random_composition, face_subset, generate_face_composition  # noqa: E402
from src.utils import generate_dirname  # noqa: E402
from src.consts import composition_labels as ALL_ELEMENTS  # noqa: E402


def read_list(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty list")
    elements = [c for c in rows[0].keys() if c not in ("idx", "family", "seed", "dirname")]
    return elements, rows


def row_composition(row, elements):
    return np.array([float(row[e]) for e in elements])


def generate(args):
    elements = [e.strip() for e in args.elements.split(",")]
    bad = [e for e in elements if e not in ALL_ELEMENTS]
    if bad:
        raise ValueError(f"unknown elements {bad}; allowed: {ALL_ELEMENTS}")
    n = len(elements)
    rows, seen = [], set()
    families = [("full", None, *map(int, args.full.split(":")))] if args.full else []
    for spec in args.face or []:
        k, cnt, seed = map(int, spec.split(":"))
        if not 1 <= k < n:
            raise ValueError(f"--face {spec}: k must be in [1, {n - 1}]")
        families.append((f"k{k}", k, cnt, seed))
    for name, k, cnt, seed in families:
        for i in range(cnt):
            if k is None:
                x = generate_random_composition(elements, seed=seed + i)
            else:
                x = generate_face_composition(n, face_subset(n, k, i, seed), args.min_nonzero_fraction, seed=seed + i,
                                              max_tries=10**6)   # all fractions >= 0.05 is rare for k = 11 (p ~ 3e-4)
            x = np.round(x, 8); j = int(np.argmax(x)); x[j] = round(1.0 - (x.sum() - x[j]), 8)   # exact sum 1, zeros stay 0
            d = generate_dirname(elements, x)
            if d in seen:
                raise RuntimeError(f"duplicate dirname {d} (family {name}, sample {i})")
            seen.add(d)
            rows.append({"idx": len(rows), "family": name, "seed": seed + i, "dirname": d,
                         **{e: f"{v:.8f}" for e, v in zip(elements, x)}})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "family", "seed", "dirname"] + elements)
        w.writeheader(); w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    print(f"{args.out}: {len(rows)} compositions, {n} elements ({args.elements}); " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))


def split_done(path, workdir):
    elements, rows = read_list(path)
    done = [r for r in rows if os.path.exists(os.path.join(workdir, r["dirname"], "results.json"))]
    todo = [r for r in rows if not os.path.exists(os.path.join(workdir, r["dirname"], "results.json"))]
    return elements, rows, done, todo


def missing(args):
    elements, rows, done, todo = split_done(args.list, args.workdir)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "family", "seed", "dirname"] + elements)
        w.writeheader(); w.writerows(todo)
    print(f"{args.out}: {len(todo)} missing of {len(rows)} ({len(done)} done in {args.workdir})")


def status(args):
    elements, rows, done, todo = split_done(args.list, args.workdir)
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    print(f"{'family':8s} {'total':>6s} {'done':>6s} {'missing':>8s}")
    for fam in fams:
        t = sum(r["family"] == fam for r in rows); d = sum(r["family"] == fam for r in done)
        print(f"{fam:8s} {t:6d} {d:6d} {t - d:8d}")
    print(f"{'all':8s} {len(rows):6d} {len(done):6d} {len(todo):8d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--elements", required=True); g.add_argument("--out", required=True)
    g.add_argument("--full", default=None, help="N:SEED full-simplex samples")
    g.add_argument("--face", action="append", help="k:N:SEED, repeatable")
    g.add_argument("--min_nonzero_fraction", type=float, default=0.05)
    g.set_defaults(func=generate)
    m = sub.add_parser("missing"); m.add_argument("list"); m.add_argument("workdir"); m.add_argument("--out", required=True)
    m.set_defaults(func=missing)
    s = sub.add_parser("status"); s.add_argument("list"); s.add_argument("workdir"); s.set_defaults(func=status)
    a = p.parse_args(); a.func(a)
