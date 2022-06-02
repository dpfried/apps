import argparse
import json
import numpy as np
import os
import pickle

def load_file(fname):
    if fname.endswith(".pkl"):
        with open(fname, "rb") as f:
            return pickle.load(f)
    elif fname.endswith(".json"):
        with open(fname, "r") as f:
            return json.load(f)
    else:
        raise Exception(f"unknown file extension for {fname}")

def write_file(obj, fname):
    if fname.endswith(".pkl"):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
    elif fname.endswith(".json"):
        with open(fname, "w") as f:
            json.dump(obj, f)
    else:
        raise Exception(f"unknown file extension for {fname}")

def combine(args):
    result_files = os.listdir(args.root)
    tmp_codes = {}
   
    # load the results and combine them
    for r_file in result_files:
        path = os.path.join(args.root, r_file)
        if path.endswith(args.suffix) and args.save not in path:
            print(path)
            results = load_file(path)
            for res in results:
                assert res not in tmp_codes
                tmp_codes[res] = results[res]
    print(f"number of problems found: {len(tmp_codes)}")
    write_file(tmp_codes, os.path.join(args.root, args.save))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="print debugging statements",
                        action="store_true")
    parser.add_argument("--root", default="./results", type=str, help="which folder to merge the results")
    parser.add_argument("--suffix", default="results.pkl")
    parser.add_argument("-s","--save", default="all_results.pkl", type=str, help="Large final save file name. Note other files use the default value.")
    args = parser.parse_args()

    combine(args)

if __name__ == "__main__":
    main()
