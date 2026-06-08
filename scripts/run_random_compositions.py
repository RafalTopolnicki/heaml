import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from process_hea import run_one_hea
from src.utils import generate_dirname, append_errorlog, save_dict_to_json
from src.consts import composition_labels as ALL_ELEMENTS


def generate_random_composition(elements, seed=None):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(len(elements)))


def compute_one_random_composition(task):
    args, sample_id, seed = task
    elements = args["elements"]

    exit_file = "EXIT"
    if os.path.exists(exit_file):
        print("EXIT file detected — stopping worker")
        return {"ok": False, "stopped": True}

    composition_ratio = generate_random_composition(elements, seed=seed)

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
    parser.add_argument("--task", type=str, default="all", choices=["lattice", "all"])
    parser.add_argument(
        "--elements", type=str, default=None,
        help="Comma-separated subset of elements to use, e.g. Ti,Nb,Zr,Hf,Ta. "
             f"Must be a subset of: {','.join(ALL_ELEMENTS)}. Defaults to all elements.",
    )
    args = vars(parser.parse_args())

    if args["elements"] is not None:
        elements = [e.strip() for e in args["elements"].split(",")]
        invalid = [e for e in elements if e not in ALL_ELEMENTS]
        if invalid:
            raise ValueError(f"Unknown elements: {invalid}. Allowed: {ALL_ELEMENTS}")
    else:
        elements = list(ALL_ELEMENTS)
    args["elements"] = elements

    print(f"Running with elements: {elements}")
    os.makedirs(args["workdir"], exist_ok=True)
    save_dict_to_json(args, os.path.join(args["workdir"], "parameters.json"))

    tasks = [(args, i, i + args["seed"]) for i in range(args["number_of_samples"])]

    if args["workers"] == 1:
        for task in tasks:
            compute_one_random_composition(task)
    else:
        with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
            futures = [executor.submit(compute_one_random_composition, task) for task in tasks]
            for future in as_completed(futures):
                _ = future.result()
