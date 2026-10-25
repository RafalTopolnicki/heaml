import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from process_hea import run_one_hea
from src.utils import generate_dirname, append_errorlog, save_dict_to_json
from src.consts import composition_labels as ALL_ELEMENTS


def generate_random_composition(elements, seed=None):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(len(elements)))


def face_subset(n_elements, n_nonzero, sample_id, base_seed):
    """Element subset (indices) for sample_id on the n_nonzero-faces of the simplex.

    All C(n, k) subsets are visited in a fixed random order (seeded by base_seed) before any
    is repeated, so e.g. 60 binaries over 10 elements cover all 45 pairs at least once.
    """
    subsets = list(itertools.combinations(range(n_elements), n_nonzero))
    order = np.random.default_rng(base_seed).permutation(len(subsets))
    return subsets[order[sample_id % len(subsets)]]


def generate_face_composition(n_elements, subset, min_nonzero_fraction=0.0, seed=None, max_tries=10000):
    """Flat Dirichlet on the elements in subset, the rest exactly 0.

    Draws with any nonzero fraction below min_nonzero_fraction are rejected, so a k-face point
    really has k components (same idea as --min_components_ratio in run_opt_compositions.py).
    """
    rng = np.random.default_rng(seed)
    k = len(subset)
    if min_nonzero_fraction * k >= 1.0:
        raise ValueError(f"min_nonzero_fraction={min_nonzero_fraction} impossible for {k} components")
    for _ in range(max_tries):
        x = rng.dirichlet(np.ones(k))
        if x.min() >= min_nonzero_fraction:
            break
    else:
        raise RuntimeError(f"no composition with all fractions >= {min_nonzero_fraction} after {max_tries} tries")
    composition = np.zeros(n_elements)
    composition[list(subset)] = x
    return composition


def compute_one_random_composition(task):
    args, sample_id, seed = task
    elements = args["elements"]

    exit_file = "EXIT"
    if os.path.exists(exit_file):
        print("EXIT file detected — stopping worker")
        return {"ok": False, "stopped": True}

    if args.get("composition_file"):
        composition_ratio = np.array(args["compositions"][sample_id])      # row sample_id of the list
    elif args.get("n_nonzero") is None:
        composition_ratio = generate_random_composition(elements, seed=seed)
    else:
        subset = face_subset(len(elements), args["n_nonzero"], sample_id, args["seed"])
        composition_ratio = generate_face_composition(len(elements), subset,
                                                      args["min_nonzero_fraction"], seed=seed)

    workdirname = generate_dirname(elements, composition_ratio)
    full_workdir = os.path.join(args["workdir"], workdirname)

    if os.path.exists(os.path.join(full_workdir, "results.json")):
        print(f"Skipping {workdirname} — results.json already exists")
        return {"ok": True, "workdirname": workdirname, "skipped": True}
    if args.get("dry_run"):
        print(f"[dry run] sample {sample_id} seed {seed} -> {workdirname}")
        return {"ok": True, "workdirname": workdirname, "dry_run": True}

    run_params = {
        "workdir": full_workdir,
        "element_labels": elements,
        "concentrations": composition_ratio,
        "task": args['task'],
        **( {"ew": args['ew']} if args['ew'] is not None else {} ),
        **({"debye_mode": args["debye_mode"]} if args.get("debye_mode", "kkr") != "kkr" else {}),
        "save_plots": args.get("save_plots", False),
    }

    try:
        run_one_hea(**run_params)
        return {"ok": True, "workdirname": workdirname}

    except Exception as exc:
        append_errorlog(args["errorlog"], workdirname)
        print(f"!!!! Error in {workdirname}: {exc}")
        return {"ok": False, "workdirname": workdirname, "mixtureerror": str(exc)}


def claim_loop(args):
    """Worker for --claim: take the next unclaimed row of the composition list, compute it, repeat.

    A row is claimed by creating its directory (atomic mkdir on a shared file system), so any number of SLURM
    tasks and workers can drain one list together and all finish at about the same time (no idle second wave).
    Rows whose directory exists are skipped: done (results.json) or being computed by someone else.  With
    --reclaim, directories without results.json are taken over (only when no other task works on the list).
    """
    elements = args["elements"]
    done = 0
    for idx in sorted(args["compositions"]):
        if idx < args["sample_offset"]:
            continue
        ratio = np.array(args["compositions"][idx])
        workdirname = generate_dirname(elements, ratio)
        full_workdir = os.path.join(args["workdir"], workdirname)
        if os.path.exists(os.path.join(full_workdir, "results.json")):
            continue
        if args.get("dry_run"):
            if not os.path.exists(full_workdir) or args.get("reclaim"):
                print(f"[dry run] would claim row {idx} -> {workdirname}")
            continue
        try:
            os.mkdir(full_workdir)
        except FileExistsError:
            if not args.get("reclaim"):
                continue
        print(f"claimed row {idx} -> {workdirname}", flush=True)
        compute_one_random_composition((args, idx, args["row_seeds"][idx]))
        done += 1
        if os.path.exists("EXIT"):   # same stop file as the workers
            break
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--errorlog", type=str, required=True)
    parser.add_argument("--number_of_samples", type=int, default=0,
                        help="Samples to compute (required without --composition_file; 0 = all rows with it)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="Global index of the first sample of this invocation. Sample i uses seed SEED+i, so parallel "
                             "SLURM tasks sharing SEED and taking disjoint offset ranges generate disjoint samples "
                             "and, for faces, keep one common C(n,k) subset order (full coverage).")
    parser.add_argument("--task", type=str, default="all", choices=["lattice", "all"])
    parser.add_argument("--ew", type=float, default=None,
                        help="Ewald parameter (overrides consts.py default if set)")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save McMillan orbital PNG plots for every composition (off by default; large output)")
    parser.add_argument("--debye_mode", type=str, default="kkr", choices=["kkr", "mix", "mjs_opt"],
                        help="Debye temperature source: 'kkr' (full KKR distortions, default), "
                             "'mix' (composition-weighted elemental θ_D, fast), "
                             "'mjs_opt' (MJS bulk-modulus formula with C=0.778, fast).")
    parser.add_argument(
        "--elements", type=str, default=None,
        help="Comma-separated subset of elements to use, e.g. Ti,Nb,Zr,Hf,Ta. "
             f"Must be a subset of: {','.join(ALL_ELEMENTS)}. Defaults to all elements.",
    )
    parser.add_argument("--n_nonzero", type=int, default=None,
                        help="Sample on the k-faces of the simplex: exactly k elements nonzero (flat Dirichlet "
                             "on them), the rest 0. Subsets are cycled so all C(n,k) are covered before repeats. "
                             "Default: None = flat Dirichlet on all elements.")
    parser.add_argument("--min_nonzero_fraction", type=float, default=0.05,
                        help="With --n_nonzero: reject draws where a nonzero fraction is below this value.")
    parser.add_argument("--composition_file", type=str, default=None,
                        help="CSV from scripts/composition_list.py: compute its rows instead of sampling. "
                             "--sample_offset is the first row and --number_of_samples the row count "
                             "(default: all remaining rows); --elements/--seed/--n_nonzero are ignored.")
    parser.add_argument("--dry_run", action="store_true", help="Print the sample -> directory mapping, compute nothing")
    parser.add_argument("--claim", action="store_true",
                        help="With --composition_file: workers claim rows dynamically (atomic mkdir of the composition "
                             "directory) instead of a fixed range, so many SLURM tasks drain the list together. "
                             "--sample_offset = first row considered; --number_of_samples is ignored.")
    parser.add_argument("--reclaim", action="store_true",
                        help="With --claim: also take rows whose directory exists but has no results.json (stale claims "
                             "of killed tasks). Use only when no other task is working on the list.")
    args = vars(parser.parse_args())

    if args["composition_file"]:
        from composition_list import read_list, row_composition
        file_elements, rows = read_list(args["composition_file"])
        args["elements"] = ",".join(file_elements)
        args["compositions"] = {int(r["idx"]): row_composition(r, file_elements).tolist() for r in rows}
        args["row_seeds"] = {int(r["idx"]): int(r["seed"]) for r in rows}
        if args["sample_offset"] >= len(rows):
            raise SystemExit(f"--sample_offset {args['sample_offset']} >= {len(rows)} rows: nothing to do")
        args["number_of_samples"] = min(args["number_of_samples"] if args["number_of_samples"] > 0 else len(rows),
                                        len(rows) - args["sample_offset"])
        args["n_nonzero"] = None
    elif args["number_of_samples"] <= 0:
        raise SystemExit("--number_of_samples must be > 0 without --composition_file")

    if args["elements"] is not None:
        elements = [e.strip() for e in args["elements"].split(",")]
        invalid = [e for e in elements if e not in ALL_ELEMENTS]
        if invalid:
            raise ValueError(f"Unknown elements: {invalid}. Allowed: {ALL_ELEMENTS}")
    else:
        elements = list(ALL_ELEMENTS)
    args["elements"] = elements
    if args["n_nonzero"] is not None and not 1 <= args["n_nonzero"] <= len(elements):
        raise ValueError(f"--n_nonzero must be in [1, {len(elements)}]")

    print(f"Running with elements: {elements}")
    os.makedirs(args["workdir"], exist_ok=True)
    if args["composition_file"]:
        params_name = f"parameters.{os.path.splitext(os.path.basename(args['composition_file']))[0]}.json"
    else:
        params_name = "parameters.json" if args["n_nonzero"] is None else f"parameters.face{args['n_nonzero']}.json"
    if os.path.exists(os.path.join(args["workdir"], params_name)):
        # extending an existing dataset: keep the original parameters, record this batch separately
        params_name = params_name.replace(".json", f".seed{args['seed']}.json")
    if args["sample_offset"]:
        params_name = params_name.replace(".json", f".off{args['sample_offset']}.json")
    save_dict_to_json({k: v for k, v in args.items() if k not in ("compositions", "row_seeds")}, os.path.join(args["workdir"], params_name))

    first = args["sample_offset"]
    if args["claim"]:
        if not args["composition_file"]:
            raise SystemExit("--claim needs --composition_file")
        if args["workers"] == 1:
            n = claim_loop(args)
        else:
            with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
                n = sum(f.result() for f in as_completed([executor.submit(claim_loop, args) for _ in range(args["workers"])]))
        print(f"claim mode: this invocation computed {n} compositions")
        raise SystemExit(0)
    tasks = [(args, i, args["row_seeds"][i] if args["composition_file"] else i + args["seed"])
             for i in range(first, first + args["number_of_samples"])]

    if args["workers"] == 1:
        for task in tasks:
            compute_one_random_composition(task)
    else:
        with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
            futures = [executor.submit(compute_one_random_composition, task) for task in tasks]
            for future in as_completed(futures):
                _ = future.result()
