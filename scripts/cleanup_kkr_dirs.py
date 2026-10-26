"""Shrink already-finished KKR composition directories (random.ratios/..., opt.ratios*/...).

Same rules as the pipeline since 2026-10-21, applied to data computed before that:
  * delete <base>.inp.pot and <base>.inp.pot.info — only for <base>.inp(.gz) inputs present in the same
    directory (exact names derived from the inputs, no wildcard deletion);
  * gzip fort.50, fort.51 and *.out (kept, can be gunzipped to recompute eta);
  * with --pack (pipeline default since 2026-10-26, file quota on Topola): move lattice/, debye/, finalscf/ into
    <comp>/kkr_raw.tar (~40 inodes -> 4 per composition: dir, results.json, run_params.json, kkr_raw.tar);
    process_kkr reads the summaries from the archive, `tar -xf kkr_raw.tar` in <comp> restores the files.
Only directories with results.json AND run_params.json are touched (finished compositions);
anything still running or failed is skipped. Nothing read by process_kkr() or the notebooks is removed.

Dry run by default (prints what would happen); pass --apply to do it.

    python scripts/cleanup_kkr_dirs.py opt.ratios.mix/RUN1 opt.ratios.mix/RUN2
    python scripts/cleanup_kkr_dirs.py opt.ratios.mix/RUN1 --apply
    python scripts/cleanup_kkr_dirs.py random.ratios/RUN --pack --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import gzip_file, pack_kkr_files, KKR_PACK_SUBDIRS, KKR_KEEP_FILES  # noqa: E402

KKR_SUBDIRS = ["lattice", "debye", "finalscf"]
GZIP_NAMES = {"fort.50", "fort.51"}


def find_composition_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root):  # does not follow symlinks
        if "results.json" in filenames and "run_params.json" in filenames:
            dirnames[:] = []  # composition dirs are leaves of interest
            yield dirpath


def plan_subdir(subdir):
    """Return (files_to_delete, files_to_gzip) for one lattice/debye/finalscf directory."""
    names = set(os.listdir(subdir))
    delete, compress = [], []
    for name in sorted(names):
        if name.endswith(".inp") or name.endswith(".inp.gz"):
            inp = name[:-3] if name.endswith(".gz") else name
            for suffix in (".pot", ".pot.info"):
                if inp + suffix in names:
                    delete.append(os.path.join(subdir, inp + suffix))
    for name in sorted(names):
        path = os.path.join(subdir, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        if (name in GZIP_NAMES or name.endswith(".out")) and name + ".gz" not in names:
            compress.append(path)
    return delete, compress


def count_packable(comp):
    n = 0
    for sub in KKR_PACK_SUBDIRS:
        subdir = os.path.join(comp, sub)
        if os.path.isdir(subdir) and not os.path.islink(subdir):
            for dirpath, _, filenames in os.walk(subdir):
                n += sum(os.path.relpath(os.path.join(dirpath, f), comp).replace(os.sep, "/") not in KKR_KEEP_FILES
                         for f in filenames)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="run directories (searched recursively for composition dirs)")
    ap.add_argument("--apply", action="store_true", help="actually delete/compress (default: dry run)")
    ap.add_argument("--pack", action="store_true", help="also pack the raw KKR files into <comp>/kkr_raw.tar")
    ap.add_argument("--verbose", action="store_true", help="list every file")
    args = ap.parse_args()

    n_comp = n_del = n_gz = n_pack = 0
    bytes_del = bytes_gz_before = bytes_gz_after = 0
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"skip (not a directory): {root}")
            continue
        for comp in find_composition_dirs(root):
            n_comp += 1
            for sub in KKR_SUBDIRS:
                subdir = os.path.join(comp, sub)
                if not os.path.isdir(subdir) or os.path.islink(subdir):
                    continue
                delete, compress = plan_subdir(subdir)
                for f in delete:
                    bytes_del += os.path.getsize(f)
                    n_del += 1
                    if args.verbose:
                        print(f"delete {f}")
                    if args.apply:
                        os.remove(f)
                for f in compress:
                    size = os.path.getsize(f)
                    bytes_gz_before += size
                    n_gz += 1
                    if args.verbose:
                        print(f"gzip   {f}")
                    if args.apply:
                        bytes_gz_after += os.path.getsize(gzip_file(f))
            if args.pack:
                if args.apply:
                    n_pack += pack_kkr_files(comp)
                else:
                    n_pack += count_packable(comp)

    mode = "DONE" if args.apply else "DRY RUN (use --apply)"
    print(f"{mode}: {n_comp} finished composition dirs")
    print(f"  delete {n_del} potential files: {bytes_del / 1e9:.2f} GB")
    gz_info = f" -> {bytes_gz_after / 1e9:.2f} GB" if args.apply else ""
    print(f"  gzip   {n_gz} fort.5x/.out files: {bytes_gz_before / 1e9:.2f} GB{gz_info}")
    if args.pack:
        print(f"  pack   {n_pack} files into kkr_raw.tar archives")


if __name__ == "__main__":
    main()
