import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from src.process_kkr import process_kkr
from src.utils import generate_dirname, append_errorlog


composition_labels = ["Ti", "Nb", "Zr", "Hf", "Ta", "Sc", "Mo", "W", "Y", "La"]
minimal_compositions = {'Ti': 0, 'Nb': 0, 'Zr': 0, 'Hf': 0, 'Ta': 0, 'Sc': 0, 'Mo': 0, 'W': 0, 'Y': 0, 'La': 0}
maximal_compositions = {'Ti': 1, 'Nb': 1, 'Zr': 1, 'Hf': 1, 'Ta': 1, 'Sc': 1, 'Mo': 1, 'W': 1, 'Y': 1, 'La': 1}
assert len(composition_labels) <= len(minimal_compositions) # check actual labels
assert len(composition_labels) <= len(maximal_compositions)


def read_experiments_from_directory(path):
    results = []
    dirlist = os.listdir(path)
    for dir in dirlist:
        res = process_kkr(path=path, dirname=dir)
        if res:
            results.append(res)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--initdir", type=str, required=True)
    parser.add_argument("--errorlog", type=str, required=True)
    parser.add_argument("--number_of_samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    #os.makedirs(args["workdir"], exist_ok=True)
    init_data = read_experiments_from_directory(args["initdir"])
    pass