"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from reindent import run as run_reindent

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm

MAX_LENGTH = 1024

EXTENSION_LENGTH = 512


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def truncate_at_stop_words(tokenizer, stop_words, sequence_ids, logprobs=None, show_warnings=True):
    # search for stopwords, to truncate after them
    full_seq_decoded = tokenizer.decode(sequence_ids, skip_special_tokens=False)
    min_index = None
    for stop_word in stop_words:
        index = full_seq_decoded.find(stop_word)
        if index < 0:
            continue
        if min_index is None or index < min_index:
            min_index = index
    
    if min_index is not None:
        # if you we find one of the stopwords, then we delete everything from the stopword on
        seq_decoded = full_seq_decoded[:min_index]
        # figure out how many tokens to take from log probs by reencoding the truncated string
        # TODO: this may not exactly be right since this I don't think BPE is a prefix code
        seq = tokenizer.encode(seq_decoded, add_special_tokens=True)
        if logprobs is not None:
            logprobs = logprobs[:len(seq)]
    else:
        if show_warnings:
            print('no stopword found!') # not having any stopword found is probably a very bad sign
        seq = sequence_ids
        seq_decoded = full_seq_decoded
    return seq, seq_decoded, logprobs

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None, notebook_formatting=False):
    if notebook_formatting:
        _input = "<| file ext=.ipynb:python |>\n<text>\n"
    else:
        _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    if notebook_formatting:
        _input += "\n</text>\n<cell>\n"
    else:
        _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking. 

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    # Only do the problems that are specified.
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

    print("Loading model...")
    if args.arch.startswith("facebook/incoder"):
        global MAX_LENGTH
        MAX_LENGTH = 2048
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.arch)
        if args.arch == "facebook/incoder-6B":
            kwargs = dict(
                        revision="float16", 
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
        else:
            kwargs = {}
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load or args.arch, **kwargs)

        stop_words = ["<|", "<|/", "<code>", "</code>", "<cell>", "</cell>", "<text>", "</text>"]
        notebook_formatting = True
    else:
        # Tokenizer
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)

        # Set up model
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load)

        stop_words = ["ANSWER:"]
        notebook_formatting = False

    model.cuda()
    print(f"Loaded {args.arch}: {args.load or ''}.")

    # main eval loop
    for index, problem in enumerate(tqdm(problems, ncols=80)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
                starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            continue

        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path, notebook_formatting=notebook_formatting)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        # Feed this into the model.
        start = time.time()
        try:
            with torch.no_grad():
                # input_ids = torch.LongTensor(tokenizer.encode(prompt_text, verbose=False)).unsqueeze(0).cuda()
                input_ids = tokenizer.encode(prompt_text, return_tensors='pt').cuda()
                output_ids = model.generate(
                    input_ids,
                    #num_beams=args.num_beams,
                    #early_stopping=True,
                    do_sample=True, 
                    top_p=0.95,
                    temperature=0.2,
                    max_length=min(MAX_LENGTH, input_ids.size(1) + EXTENSION_LENGTH),
                )
                if args.debug:
                    print("input size:")
                    print(input_ids.size())
                    print("output size:")
                    print(output_ids.size())
                output_ids = output_ids[0][input_ids.size(1):]
                if stop_words:
                    output_ids, output_str, _ = truncate_at_stop_words(tokenizer, stop_words, output_ids)
                else:
                    output_str = tokenizer.decode(output_ids)
        except Exception as e:
            if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                # See https://github.com/huggingface/transformers/issues/5118
                if args.debug:
                    print(f"Problem text was > {MAX_LENGTH} tokens, so cannot do generation")
                    print(e)
            else:
                print("Unexpected exception in generating solution")
                print(e)
            # Default to empty string on errors
            output_str = ""
        end = time.time()

        if args.peeking == 1.0:
            output_str = sample_sol
        elif len(output_str):
            output_str = output_str.replace("<|endoftext|>", "")
            # split_str = "<cell>\n" if notebook_formatting else "ANSWER:\n"
            # output_str = output_str.split(split_str)[1].replace("<|endoftext|>", "")

        # Save the generated sol
        gpt_codes[index+args.start] = output_str

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + ["facebook/incoder-6B", "facebook/incoder-1B"])
    parser.add_argument("-t","--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
 
    args = parser.parse_args()

    main(args)
