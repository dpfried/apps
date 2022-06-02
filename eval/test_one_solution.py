"""
Run solutions from one problem.
"""

import io
import json
import logging
import math
import numpy as np
import os
import pprint
import sys
import testing_util as test_util
import time
import pickle

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from typing import List
from collections import Counter

def print_results(results, args):

    per_problem_accuracies = []
    all_correct = []

    total_problems = 0
    total_testcases = 0
    total_testcases_run = 0

    error_counter = Counter()

    for index in results:
        results_per_output: List[test_util.TestResults] = results[index]
        assert len(results_per_output) == 1
        total_problems += 1
        for this_results in results_per_output:
            if this_results.error is not None:
                error_counter[this_results.error] += 1
            total_testcases += this_results.num_tests
            total_testcases_run += len(this_results.test_results)

            # this_results.test_results contains True, False, and -1. Only True are successes
            test_results = [1 if r == True else 0 for r in this_results.test_results]

            while len(test_results) <= this_results.num_tests:
                # if tests are missing, they failed
                test_results.append(0)

            test_results = np.array(test_results)
            per_problem_accuracies.append(np.mean(test_results))
            all_correct.append(np.all(test_results))

    print("errors:")
    print(error_counter.most_common())

    print(f"number of problems = {total_problems}")
    print(f"number of test cases = {total_testcases}")
    print(f"number of test cases run = {total_testcases_run}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_problem_accuracies)}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")

def print_results_old(results, args):
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
       res.extend(results[index])
       per_prob_res.append(np.mean(results[index]))
       all_correct.append(np.all(results[index]))
    tmp_results = res
    compile_errors = len(tmp_results[tmp_results==-2])
    runtime_errors = len(tmp_results[tmp_results==-1])
    failures = len(tmp_results[tmp_results==False])
    successes = len(tmp_results[tmp_results==True])
    total_testcases = len(res)
    if args.debug:
        print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }")
        print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
        print(f"number of test cases run = {total_testcases}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")


def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))
    print(len(problems))

    if args.shard is not None:
        assert 0 <= args.shard < args.num_shards
        shard_size = int(math.ceil(len(problems) / args.num_shards))
        start = shard_size * args.shard
        end = shard_size * (args.shard + 1)
        if args.shard == args.num_shards - 1:
            assert end == len(problems)
        args.start = start
        args.end = end
        print(f"shard {args.shard}: [{args.start}, {args.end})")


    gpt_codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    results_loc = os.path.join(args.save, f"all_results.pkl") 
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.pkl") 

    print(codes_loc, results_loc)

    with open(codes_loc, "r") as f: 
        gpt_codes = json.load(f)

    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    if args.stop_early:
        problems = problems[:args.stop_early]

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        try:
            output_str = gpt_codes[str(index+args.start)]
            if args.debug:
                print(f"\n\nproblem path = {problem}")
                print(output_str)
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue
        prob_path = os.path.join(args.root, problem)

        # with open(os.path.join(prob_path, "solutions.json"), "r") as f:
        #     sols = json.load(f)
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if isinstance(output_str, str):
            outputs = [output_str]
        else:
            assert isinstance(output_str, list)
            outputs = output_str

        res = []
        for o_idx, o in enumerate(outputs):
            if args.debug:
                print(f"\nTesting solution {o_idx}")
                print(f"{o}")
            # curr_res = [-2]
            # try:
            if True:
                curr_res = test_util.run_test(prob_path=prob_path, test=o, debug=args.debug)
                # fixed = []
                # for e in curr_res:
                #     if isinstance(e, np.ndarray):
                #        e = e.item(0)
                #     if isinstance(e, np.bool_):
                #         e = bool(e)
                #     fixed.append(e)
                # curr_res = fixed
                # if not np.all(curr_res):
                #     print(f"Results were not all True: {curr_res}")
            # except Exception as e:
            #     print(f"test framework exception = {repr(e)}{e}\n")
            #     break
            # finally:
                # assert isinstance(curr_res, list)
                res.append(curr_res)

        # if args.debug:
            # print(f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
            #print(f"results = {res}")
 
        results[index+args.start+args.index] = res
        
        with open(results_loc, "wb") as f:
            try:
                pickle.dump(results, f)
                # f.write(json.dumps(results))
            except Exception as e:
                print(f"didn't save problem due to {e}")
                import pdb; pdb.set_trace()

    return results


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.pkl") 
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.pkl") 
        with open(results_loc, "rb") as f: 
            results = pickle.load(f)
    else:
        results = eval_and_save_problems(args)

    print_results(results, args)
    # print_results_old(results, args)


if __name__ == "__main__":
    import argparse

    import sys
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--num_shards", default=10, type=int)
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stop-early", default=None, type=int)
 
    args = parser.parse_args()

    main(args)
