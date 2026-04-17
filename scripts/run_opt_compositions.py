import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import math
from src.process_kkr import process_kkr
from src.utils import generate_dirname, append_errorlog, save_dict_to_json, log_iteration_summary
from src.ml import train_cb_model
from src.consts import composition_labels, ACQUISITION_ALPHA, ACQUISITION_METRIC, TARGET, CANDIDATE_COMPOSITIONS_N
from src.sampling import generate_candidates_data
from process_hea import run_one_hea
import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_distances

minimal_compositions = {'Ti': 0, 'Nb': 0, 'Zr': 0, 'Hf': 0, 'Ta': 0, 'Sc': 0, 'Mo': 0, 'W': 0, 'Y': 0, 'La': 0}
maximal_compositions = {'Ti': 0.6, 'Nb': 0.6, 'Zr': 0.6, 'Hf': 0.6, 'Ta': 0.6, 'Sc': 0.6, 'Mo': 0.6, 'W': 0.6, 'Y': 0.6, 'La': 0.6}
assert len(composition_labels) <= len(minimal_compositions) # check actual labels
assert len(composition_labels) <= len(maximal_compositions)


def read_experiments_from_directory(path):
    results = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):  # only directories
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
    vals = [d[TARGET] for d in data]
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

def filter_known_candidates(df_known, df_candidates, columns, tol=1e-6):
    """
    Remove candidate rows that are already present in known data
    (up to a tolerance in composition space).
    """
    dists = compute_min_distances(
        df_known=df_known,
        df_candidates=df_candidates,
        columns=columns,
        metric="euclidean",
    )
    return df_candidates.loc[dists > tol].copy()

def composition_key(row, columns, ndigits=6):
    return tuple(round(float(row[c]), ndigits) for c in columns)

def deduplicate_known_data(data, columns, ndigits=8):
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
    parser.add_argument("--initdir", type=str, required=True)
    parser.add_argument("--errorlog", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--champions_per_step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_of_models", type=int, default=10)
    args = vars(parser.parse_args())
    champions_per_step = args['champions_per_step']
    workdir = args['workdir']
    iterations = args['iterations']
    os.makedirs(workdir, exist_ok=True)
    save_dict_to_json(args, os.path.join(workdir, "parameters.json"))
    iteration_log_path = os.path.join(workdir, "optimization_log.txt")

    expected_composition_distance = np.power(math.factorial(len(composition_labels)), 1.0/len(composition_labels)) * np.power(CANDIDATE_COMPOSITIONS_N, 1.0/len(composition_labels))
    expected_composition_distance = 1.0/expected_composition_distance
    print(f'Expected axis distance: {expected_composition_distance}')


    # generate candidates
    all_candidates = generate_candidates_data(min_comp=minimal_compositions, max_comp=maximal_compositions)

    # read initial computations
    init_data = read_experiments_from_directory(args["initdir"])
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

        # train multiple models
        model_training_metrics = []
        preds = []
        for model_id in range(args["number_of_models"]):
            print(f'(II) Training model: {model_id}')
            model, metrics, pred_ = train_cb_model(known_data, seed=100+model_id, predict_df=all_candidates)
            preds.append(pred_)
            model_training_metrics.append({'metrics': metrics})
        preds = np.array(preds)
        # find acquisition and candidates for the new HAE
        mus = preds.mean(axis=0)
        sigmas = preds.std(axis=0)
        acquisitions = mus + 2*sigmas
        # df_candidates = all_candidates.copy()
        # df_candidates['raw_acquisition'] = acquisitions
        # df_candidates['composition_distance'] = compute_min_distances(df_known=pd.DataFrame(init_data), df_candidates=df_candidates, columns=composition_labels, metric=ACQUISITION_METRIC)
        # df_candidates['acquisition'] = df_candidates['raw_acquisition'] - ACQUISITION_ALPHA*df_candidates['composition_distance']
        # df_candidates = df_candidates.sort_values(by='acquisition', ascending=True)
        # df_top_candidates = df_candidates.head(champions_per_step).reset_index(drop=True)
        df_candidates = all_candidates.copy()
        df_candidates["raw_acquisition"] = acquisitions

        df_candidates = all_candidates.copy()
        df_candidates["raw_acquisition"] = acquisitions

        # remove already-known or too-close candidates
        df_candidates = filter_known_candidates(
            df_known=pd.DataFrame(known_data),
            df_candidates=df_candidates,
            columns=composition_labels,
            tol=1e-6,  # try 0.01 later for stronger novelty
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
        # evaluate KKR on those top coordinates
        tasks = []
        for _, row in df_top_candidates.iterrows():
            comp_dict = dict(zip(composition_labels, row[composition_labels].values))
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
        new_data = read_experiments_from_directory(computationdir)

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
