import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np

from process_hea import run_one_hea


composition_labels = ["Ti", "Nb", "Zr", "Hf", "Ta", "Sc", "Mo", "W", "Y", "La"]
minimal_compositions = {'Ti': 0, 'Nb': 0, 'Zr': 0, 'Hf': 0, 'Ta': 0, 'Sc': 0, 'Mo': 0, 'W': 0, 'Y': 0, 'La': 0}
maximal_compositions = {'Ti': 1, 'Nb': 1, 'Zr': 1, 'Hf': 1, 'Ta': 1, 'Sc': 1, 'Mo': 1, 'W': 1, 'Y': 1, 'La': 1}
assert len(composition_labels) <= len(minimal_compositions) # check actual labels
assert len(composition_labels) <= len(maximal_compositions)

def generate_random_composition(n_elements=5, seed=None):
    rng = np.random.default_rng(seed)
    while True:
        compostion = rng.dirichlet(np.ones(n_elements))
        for element, c in zip(composition_labels, compostion):
            if c >= minimal_compositions[element] and c <= maximal_compositions[element]:
                return compostion


def generate_dirname(composition_labels, composition_ratio):
    txt = ""
    for label, ratio in zip(composition_labels, composition_ratio):
        txt += f"{label}{ratio:.4f}"
    return txt


def append_errorlog(errorlog_path, workdirname):
    errorlog_dir = os.path.dirname(errorlog_path)
    if errorlog_dir:
        os.makedirs(errorlog_dir, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(errorlog_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {workdirname}\n")


import os

def compute_one_random_composition(task):
    args, sample_id, seed = task

    # global stop condition
    exit_file = "EXIT"
    if os.path.exists(exit_file):
        print("EXIT file detected — stopping worker")
        return {"ok": False, "stopped": True}

    # independent RNG per task
    composition_ratio = generate_random_composition(
        n_elements=len(composition_labels),
        seed=seed,
    )

    workdirname = generate_dirname(composition_labels, composition_ratio)
    full_workdir = os.path.join(args["workdir"], workdirname)

    run_params = {
        "workdir": full_workdir,
        "element_labels": composition_labels,
        "concentrations": composition_ratio,
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
    args = vars(parser.parse_args())

    os.makedirs(args["workdir"], exist_ok=True)

    tasks = [(args, i, i+args["seed"]) for i in range(args["number_of_samples"])]

    if args["workers"] == 1:
        for task in tasks:
            compute_one_random_composition(task)
    else:
        with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
            futures = [executor.submit(compute_one_random_composition, task) for task in tasks]
            for future in as_completed(futures):
                _ = future.result()