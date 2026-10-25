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

    if args.get("n_nonzero") is None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--errorlog", type=str, required=True)
    parser.add_argument("--number_of_samples", type=int, default=2)
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
    args = vars(parser.parse_args())

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
    params_name = "parameters.json" if args["n_nonzero"] is None else f"parameters.face{args['n_nonzero']}.json"
    if os.path.exists(os.path.join(args["workdir"], params_name)):
        # extending an existing dataset: keep the original parameters, record this batch separately
        params_name = params_name.replace(".json", f".seed{args['seed']}.json")
    if args["sample_offset"]:
        params_name = params_name.replace(".json", f".off{args['sample_offset']}.json")
    save_dict_to_json(args, os.path.join(args["workdir"], params_name))

    first = args["sample_offset"]
    tasks = [(args, i, i + args["seed"]) for i in range(first, first + args["number_of_samples"])]

    if args["workers"] == 1:
        for task in tasks:
            compute_one_random_composition(task)
    else:
        with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
            futures = [executor.submit(compute_one_random_composition, task) for task in tasks]
            for future in as_completed(futures):
                _ = future.result()
