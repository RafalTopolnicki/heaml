"""Build a flat CSV database (src.database) from finished KKR composition directories, for
run_opt_compositions.py --init_db. Reads plain or packed (kkr_raw.tar) entries; nothing is modified.

    python scripts/make_database.py random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl \
        --out databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv [--workers 12]

Several source directories go into one file. Entries without results.json + run_params.json (unfinished) are
skipped; entries that fail to process are listed. Rows with thetaDB=None (C' < 0) are kept with Tc=0;
the optimizer drops them when reading, as it did for directories.
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.database import database_row, write_database, read_database  # noqa: E402


def _one(args):
    path, name = args
    try:
        return name, database_row(path, name), None
    except Exception as e:
        return name, None, f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+', help='dataset directories with one subdirectory per composition')
    ap.add_argument('--out', required=True, help='output CSV')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    tasks = []
    for src in args.src:
        for name in sorted(os.listdir(src)):
            d = os.path.join(src, name)
            if os.path.isfile(os.path.join(d, 'results.json')) and os.path.isfile(os.path.join(d, 'run_params.json')):
                tasks.append((src, name))
    print(f'{len(tasks)} finished compositions in {len(args.src)} source dir(s)')
    with Pool(args.workers) as pool:
        out = pool.map(_one, tasks, chunksize=8)
    rows = [r for _, r, _ in out if r is not None]
    failed = [(n, e) for n, _, e in out if e is not None]
    for n, e in failed:
        print(f'FAILED {n}: {e}')
    seen = set()
    dup = [r['name'] for r in rows if r['name'] in seen or seen.add(r['name'])]
    if dup:
        print(f'WARNING: {len(dup)} duplicate names (same composition in several sources), e.g. {dup[:3]}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df = write_database(rows, args.out)
    usable, _ = read_database(args.out)
    print(f'wrote {args.out}: {len(df)} rows x {df.shape[1]} columns ({len(failed)} failed); '
          f'{len(usable)} usable by the optimizer (thetaDB present)')


if __name__ == '__main__':
    main()
