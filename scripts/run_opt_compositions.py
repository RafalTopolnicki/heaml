import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import math
from src.process_kkr import process_kkr
from src.utils import generate_dirname, append_errorlog, save_dict_to_json, log_iteration_summary
from src.ml import train_cb_model
from src.consts import composition_labels as ALL_ELEMENTS, ACQUISITION_ALPHA, ACQUISITION_METRIC, TARGET, CANDIDATE_COMPOSITIONS_N, MIN_NOVELTY_DIST, FRESH_FRACTION, LOCAL_TOP_K, LOCAL_NOISE_SCALE, MODEL_SUBSAMPLE_FRACTION
from src.sampling import generate_candidates_data
from process_hea import run_one_hea
import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_distances


def snapshot_python_code(workdir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(workdir, "code")

    copy_specs = [
        (script_dir, os.path.join(code_dir, "scripts")),
        (os.path.join(script_dir, "src"), os.path.join(code_dir, "scripts", "src")),
    ]

    for source_dir, target_dir in copy_specs:
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(source_dir):
            source_path = os.path.join(source_dir, filename)
            if os.path.isfile(source_path) and filename.endswith(".py"):
                shutil.copy2(source_path, os.path.join(target_dir, filename))


def read_experiments_from_directory(paths):
    if isinstance(paths, str):
        paths = [paths]
    results = []
    for path in paths:
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                res = process_kkr(path=path, dirname=entry)
                if res:
                    results.append(res)
    return results

def compute_min_distances(df_known, df_candidates, columns, metric="euclidean"):

    X_known = df_known[columns].values
    X_cand = df_candidates[columns].values

    if metric == "cosine":
        # sklearn already computes 1 - cosine_similarity
        dist = cosine_distances(X_cand, X_known)
        return dist.min(axis=1)

    # manual broadcasting for other metrics
    diff = X_cand[:, None, :] - X_known[None, :, :]

    if metric == "euclidean":
        dist = np.sqrt(np.sum(diff**2, axis=2))
    elif metric == "manhattan":
        dist = np.sum(np.abs(diff), axis=2)
    else:
        raise ValueError("Unknown metric")

    return np.min(dist, axis=1)

def compute_one_composition(composition_dict, workdir):
    # global stop condition
    exit_file = "EXIT"
    if os.path.exists(exit_file):
        print("EXIT file detected — stopping worker")
        return {"ok": False, "stopped": True}

    composition_labels = []
    composition_ratios = []
    for key, value in composition_dict.items():
        composition_labels.append(key)
        composition_ratios.append(value)

    workdirname = generate_dirname(composition_labels, composition_ratios)
    full_workdir = os.path.join(workdir, workdirname)

    if os.path.exists(os.path.join(full_workdir, "results.json")):
        print(f"Skipping {workdirname} — results.json already exists")
        return {"ok": True, "workdirname": workdirname, "skipped": True}

    run_params = {
        "workdir": full_workdir,
        "element_labels": composition_labels,
        "concentrations": composition_ratios,
        "task": 'all',
    }
    try:
        run_one_hea(**run_params)
        return {"ok": True, "workdirname": workdirname}

    except Exception as exc:
        append_errorlog(args["errorlog"], workdirname)
        print(f"!!!! Error in {workdirname}: {exc}")
        return {"ok": False, "workdirname": workdirname, "mixtureerror": str(exc)}

def compute_one_composition_task(task):
    comp_dict, computation_dir = task
    return compute_one_composition(comp_dict, computation_dir)

def find_largest_in_data(data):
    vals = [d[TARGET] for d in data if TARGET in d and pd.notna(d[TARGET])]
    return np.max(vals)


def select_diverse_top_candidates(
    df_known,
    df_candidates,
    columns,
    n_select,
    alpha,
    metric="euclidean",
    score_col="raw_acquisition",
    maximize=True,
):
    """
    Greedy batch selection:
    - pick one candidate
    - add it to known set
    - recompute distance penalty
    - pick next

    Returns
    -------
    pd.DataFrame
        Selected candidate rows, in chosen order.
    """
    known = df_known[columns].copy().reset_index(drop=True)
    remaining = df_candidates.copy().reset_index(drop=False)  # keep original index
    selected_rows = []

    for _ in range(min(n_select, len(remaining))):
        dists = compute_min_distances(
            df_known=known,
            df_candidates=remaining,
            columns=columns,
            metric=metric,
        )

        remaining["composition_distance"] = dists

        if maximize:
            remaining["acquisition"] = (
                remaining[score_col] + alpha * remaining["composition_distance"]
            )
            best_idx = remaining["acquisition"].idxmax()
        else:
            remaining["acquisition"] = (
                remaining[score_col] - alpha * remaining["composition_distance"]
            )
            best_idx = remaining["acquisition"].idxmin()

        best_row = remaining.loc[best_idx].copy()
        selected_rows.append(best_row)

        # add selected composition to known set before next round
        known = pd.concat(
            [known, pd.DataFrame([best_row[columns].to_dict()])],
            ignore_index=True,
        )

        # remove selected row from remaining pool
        remaining = remaining.drop(index=best_idx).reset_index(drop=True)

    return pd.DataFrame(selected_rows)

def filter_known_candidates(df_known, df_candidates, columns, min_dist=0.01, metric="euclidean"):
    """
    Remove candidates that are too close to already known points.
    """
    dists = compute_min_distances(
        df_known=df_known,
        df_candidates=df_candidates,
        columns=columns,
        metric=metric,
    )
    out = df_candidates.copy()
    out["dist_to_known"] = dists
    return out.loc[out["dist_to_known"] >= min_dist].copy()

def composition_key(row, columns, ndigits=5):
    return tuple(round(float(row[c]), ndigits) for c in columns)

def write_init_tc_comparison(init_data, initdirs, workdir, target=TARGET):
    if isinstance(initdirs, str):
        initdirs = [initdirs]
    rows = []
    for entry in init_data:
        name = entry.get("name", "")
        tc_computed = entry.get(target)
        tc_stored = None
        results_path = next(
            (os.path.join(d, name, "results.json") for d in initdirs
             if os.path.exists(os.path.join(d, name, "results.json"))),
            None,
        )
        if results_path is not None:
            try:
                with open(results_path) as f:
                    stored = json.load(f)
                tc_stored = stored.get(target)
            except Exception:
                pass
        rows.append({
            "name": name,
            f"{target}_computed": tc_computed,
            f"{target}_stored": tc_stored,
            "diff": (tc_computed - tc_stored) if (tc_computed is not None and tc_stored is not None) else None,
        })
    df = pd.DataFrame(rows)
    out_path = os.path.join(workdir, "init_tc_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"Init Tc comparison written to {out_path} ({len(df)} rows)")


def normalize_rows_to_elements(data, elements):
    """Fill missing element keys with 0.0 so all rows have the same composition columns."""
    for row in data:
        for el in elements:
            if el not in row:
                row[el] = 0.0
    return data

def deduplicate_known_data(data, columns, ndigits=5):
    seen = set()
    out = []
    for row in data:
        key = composition_key(row, columns, ndigits)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--initdir", type=str, nargs='+', required=True)
    parser.add_argument("--errorlog", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--champions_per_step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_of_models", type=int, default=10)
    parser.add_argument(
        "--elements", type=str, default=None,
        help="Comma-separated subset of elements to use, e.g. Ti,Nb,Zr,Hf,Ta. "
             f"Must be a subset of: {','.join(ALL_ELEMENTS)}. Defaults to all elements.",
    )
    parser.add_argument(
        "--min_element", type=str, action="append", default=[],
        help="Per-element minimum concentration, e.g. --min_element Nb=0.05. Can be repeated.",
    )
    parser.add_argument(
        "--max_element", type=str, action="append", default=[],
        help="Per-element maximum concentration, e.g. --max_element Sc=0.15. Can be repeated.",
    )
    parser.add_argument(
        "--min_components", type=int, default=1,
        help="Minimum number of elements that must have composition > min_components_ratio. "
             "Set to 0 or 1 to allow any composition including pure phases. Default: 1.",
    )
    parser.add_argument(
        "--min_components_ratio", type=float, default=0.0,
        help="Minimum composition fraction for an element to count toward min_components. Default: 0.0.",
    )
    parser.add_argument(
        "--acquisition_beta", type=float, default=2.0,
        help="Exploration coefficient in UCB acquisition: score = mu + beta * sigma. Default: 2.0.",
    )
    args = vars(parser.parse_args())

    if args["elements"] is not None:
        composition_labels = [e.strip() for e in args["elements"].split(",")]
        invalid = [e for e in composition_labels if e not in ALL_ELEMENTS]
        if invalid:
            raise ValueError(f"Unknown elements: {invalid}. Allowed: {ALL_ELEMENTS}")
    else:
        composition_labels = list(ALL_ELEMENTS)
    print(f"Running with elements: {composition_labels}")

    minimal_compositions = {e: 0.0 for e in composition_labels}
    maximal_compositions = {e: 0.6 for e in composition_labels}

    for spec in args["min_element"]:
        el, val = spec.split("=")
        el = el.strip()
        if el not in composition_labels:
            raise ValueError(f"--min_element unknown element: {el}")
        minimal_compositions[el] = float(val)

    for spec in args["max_element"]:
        el, val = spec.split("=")
        el = el.strip()
        if el not in composition_labels:
            raise ValueError(f"--max_element unknown element: {el}")
        maximal_compositions[el] = float(val)

    print(f"Composition bounds: min={minimal_compositions} max={maximal_compositions}")
    print(f"Component diversity: min_components={args['min_components']} min_components_ratio={args['min_components_ratio']}")

    champions_per_step = args['champions_per_step']
    workdir = args['workdir']
    iterations = args['iterations']
    os.makedirs(workdir, exist_ok=True)
    snapshot_python_code(workdir)
    save_dict_to_json(args, os.path.join(workdir, "parameters.json"))
    iteration_log_path = os.path.join(workdir, "optimization_log.txt")

    expected_composition_distance = np.power(math.factorial(len(composition_labels)), 1.0/len(composition_labels)) * np.power(CANDIDATE_COMPOSITIONS_N, 1.0/len(composition_labels))
    expected_composition_distance = 1.0/expected_composition_distance
    print(f'Expected axis distance: {expected_composition_distance}')


    # generate candidates
    #all_candidates = generate_candidates_data(min_comp=minimal_compositions, max_comp=maximal_compositions)

    # read initial computations
    init_data = read_experiments_from_directory(args["initdir"])
    normalize_rows_to_elements(init_data, composition_labels)
    write_init_tc_comparison(init_data, args["initdir"], workdir)
    known_data = init_data.copy()

    for iteration in range(1, iterations+1):
        print(f'(IIII) Iteration: ', iteration, datetime.datetime.now(), 'Number of datapoints: ', len(known_data), 'MaxTc:', find_largest_in_data(known_data))
        exit_file = "EXIT"
        if os.path.exists(exit_file):
            print("EXIT file detected — stopping all")
            sys.exit(0)

        iterationdir = os.path.join(workdir, f'iteration_{iteration}')
        computationdir = os.path.join(iterationdir, 'computation')
        os.makedirs(computationdir, exist_ok=True)

        # generate new candidates
        all_candidates = generate_candidates_data(
            known_data=known_data,
            min_comp=minimal_compositions,
            max_comp=maximal_compositions,
            n_candidates=CANDIDATE_COMPOSITIONS_N,
            fresh_fraction=FRESH_FRACTION,
            local_top_k=LOCAL_TOP_K,
            local_noise_scale=LOCAL_NOISE_SCALE,
            seed=args["seed"] + iteration,
            composition_labels=composition_labels,
            min_components=args["min_components"],
            min_components_ratio=args["min_components_ratio"],
        )

        # train multiple models, each on a random subset of known data for ensemble diversity
        model_training_metrics = []
        preds = []
        rng = np.random.default_rng(args["seed"] + iteration)
        for model_id in range(args["number_of_models"]):
            n = len(known_data)
            k = max(1, int(MODEL_SUBSAMPLE_FRACTION * n))
            indices = rng.choice(n, size=k, replace=False)
            sub_data = [known_data[i] for i in indices]
            print(f'(II) Training model: {model_id} on {k}/{n} points')
            model, metrics, pred_ = train_cb_model(sub_data, seed=100+model_id, predict_df=all_candidates, elements=composition_labels)
            preds.append(pred_)
            model_training_metrics.append({'metrics': metrics})
        preds = np.array(preds)
        # find acquisition and candidates for the new HAE
        mus = preds.mean(axis=0)
        sigmas = preds.std(axis=0)
        acquisitions = mus + args["acquisition_beta"] * sigmas
        df_candidates = all_candidates.copy()
        df_candidates["pred_target"] = mus
        df_candidates["pred_target_std"] = sigmas
        df_candidates["raw_acquisition"] = acquisitions

        # remove already-known or too-close candidates
        df_candidates = filter_known_candidates(
            df_known=pd.DataFrame(known_data),
            df_candidates=df_candidates,
            columns=composition_labels,
            min_dist=MIN_NOVELTY_DIST,
            metric="euclidean",
        ).reset_index(drop=True)

        df_top_candidates = select_diverse_top_candidates(

            df_known=pd.DataFrame(known_data),
            df_candidates=df_candidates,
            columns=composition_labels,
            n_select=champions_per_step,
            alpha=ACQUISITION_ALPHA,
            metric=ACQUISITION_METRIC,
            score_col="raw_acquisition",
            maximize=True,
        ).reset_index(drop=True)

        # log results
        save_dict_to_json(model_training_metrics, os.path.join(iterationdir, "model_training_metrics.json"))
        df_top_candidates.to_csv(os.path.join(iterationdir, "top_candidates.csv"), index=False)
        #df_candidates.to_csv(os.path.join(iterationdir, "all_candidates.csv"), index=False)

        # log selected candidates and write per-composition selection_info.json
        print(f"\n[Iter {iteration}] Selected {len(df_top_candidates)} candidates:")
        tasks = []
        for _, row in df_top_candidates.iterrows():
            comp_dict = dict(zip(composition_labels, row[composition_labels].values))
            workdirname = generate_dirname(composition_labels, list(comp_dict.values()))
            source = row.get("_source", "unknown")
            source_label = "local (top-K neighborhood)" if source == "local" else "global (random Sobol)"
            pred_target = float(row.get("pred_target", float("nan")))
            pred_target_std = float(row.get("pred_target_std", float("nan")))
            acq = float(row.get("raw_acquisition", float("nan")))
            print(
                f"  {workdirname:50s} | pred_target={pred_target:.4f}"
                f" | std={pred_target_std:.4f} | acq={acq:.4f} | source={source_label}"
            )
            comp_dir = os.path.join(computationdir, workdirname)
            os.makedirs(comp_dir, exist_ok=True)
            selection_info = {
                "iteration": iteration,
                "pred_target": pred_target,
                "pred_target_std": pred_target_std,
                "acquisition": acq,
                "source": source,
                "source_label": source_label,
                "target": TARGET,
                "composition_distance": float(row.get("composition_distance", float("nan"))),
                "composition": comp_dict,
            }
            save_dict_to_json(selection_info, os.path.join(comp_dir, "selection_info.json"))
            tasks.append((comp_dict, computationdir))

        if args["workers"] == 1:
            for task in tasks:
                compute_one_composition_task(task)
        else:
            with ProcessPoolExecutor(max_workers=args["workers"]) as executor:
                futures = [executor.submit(compute_one_composition_task, task) for task in tasks]
                for future in as_completed(futures):
                    _ = future.result()
        # now merge all new KKR results with the intial one
        new_data = [d for d in read_experiments_from_directory(computationdir) if
                    d.get(TARGET) is not None and not pd.isna(d.get(TARGET))]
        normalize_rows_to_elements(new_data, composition_labels)

        # create now list of all known data
        known_data = known_data + new_data
        known_data = deduplicate_known_data(known_data, composition_labels)

        # log the progress
        log_iteration_summary(
            log_path=iteration_log_path,
            iteration=iteration,
            known_data=known_data,
            new_data=new_data,
            composition_labels=composition_labels,
            target_col=TARGET,
        )
