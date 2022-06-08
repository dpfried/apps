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

import sys
sys.path.append("../train")
from dataset_apps.APPSBaseDataset import APPSBaseDataset, FORMATTING_TYPES

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm

MAX_LENGTH = 1024

EXTENSION_LENGTH = 512

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

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.train_loc, "r") as f:
        train_problems = json.load(f)
    train_problems = sorted(train_problems)
    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    
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

    # ncg_str = f"ncg-{args.num_candidates_generated}"

    if not args.end:
        # codes_loc = os.path.join(args.save, f"all_codes_{ncg_str}.json")
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        # codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes_{ncg_str}.json")
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

    formatting_type = args.formatting_type

    print("Loading model...")
    if args.arch.startswith("facebook/incoder"):
        global MAX_LENGTH
        MAX_LENGTH = 2048
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.arch)
        if args.arch == "facebook/incoder-6B":
            # kwargs = dict(
            #             revision="float16", 
            #             torch_dtype=torch.float16,
            #             low_cpu_mem_usage=True,
            #         )
            kwargs = {}
        else:
            kwargs = {}
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load or args.arch, **kwargs)

        model = model.half()

        stop_words = ["<|", "<|/", "<code>", "</code>", "<cell>", "</cell>", "<text>", "</text>"]
        if formatting_type is None:
            formatting_type = "notebook"
        reindent_code = True
    else:
        # Tokenizer
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)

        # Set up model
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load)

        stop_words = ["ANSWER:"]
        if formatting_type is None:
            formatting_type = "qa"
        if formatting_type != "qa":
            raise NotImplementedError("need to handle stopwords for non-qa types for non-incoder models")
        
        reindent_code = False

    model.cuda()
    print(f"Loaded {args.arch}: {args.load or ''}.")

    def generate_prompt_from_path(dataroot, problem_name):
        samples = APPSBaseDataset.load_samples(dataroot, problem_name, require_solutions=False, reindent_code=reindent_code)
        if samples is None:
            return None, None, None
        short_sol = min(samples, key=lambda sample:len(sample.sol_str)).sol_str
        sample_sol = random.choice(samples).sol_str
        prompt_text = APPSBaseDataset.prompt_from_sample(samples[0], formatting_type=formatting_type)

        # peek at part of the solution
        if args.peeking > 0.0:
            rand_sol = random.choice(samples).sol_str
            rand_sol = tokenizer.encode(rand_sol, verbose=False)
            tokens_taken = int(args.peek_frac * len(rand_sol))
            rand_sol = rand_sol[:tokens_taken]
            prompt_text += tokenizer.decode(rand_sol)

        return prompt_text, sample_sol, short_sol

    if args.k_shot_prompts > 0:
        print(f"len(train_problems): {len(train_problems)}")
        k_shot_prompts = []
        for path in train_problems:
            prob_path = os.path.join(args.root, path)
            prompt_text, sample_sol, short_sol = generate_prompt_from_path(args.root, path)
            if prompt_text is None:
                continue
            this_prompt = prompt_text + short_sol
            if formatting_type == "notebook":
                this_prompt += "\n</cell>"
            elif formatting_type == "stackoverflow":
                raise NotImplementedError("k-shot stackoverflow prompting")
            k_shot_prompts.append(this_prompt)
            if len(k_shot_prompts) >= args.k_shot_prompts:
                break
        print(f"len(k_shot_prompts): {len(k_shot_prompts)}")
        if args.debug:
            print("K_SHOT_PROMPTS:")
            print("\n".join(k_shot_prompts))
    else:
        k_shot_prompts = []

    # main eval loop
    for index, problem in enumerate(tqdm(problems, ncols=80)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")
        prompt_text, sample_sol, short_sol = generate_prompt_from_path(args.root, problem)
        if prompt_text is None:
            continue

        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        if k_shot_prompts:
            prompt_text = "\n".join(k_shot_prompts) + "\n" + prompt_text

        # Feed this into the model.
        start = time.time()
        try:
        # if True:
            with torch.no_grad():
                # input_ids = torch.LongTensor(tokenizer.encode(prompt_text, verbose=False)).unsqueeze(0).cuda()
                input_ids = tokenizer.encode(prompt_text, return_tensors='pt').cuda()
                if input_ids.size(-1) >= MAX_LENGTH:
                    raise ValueError(f"input {problem} is too long; skipping")
                # TODO: subbatching if num_candidates_generated is too big for the GPU
                batch_output_ids = model.generate(
                    input_ids,
                    #num_beams=args.num_beams,
                    #early_stopping=True,
                    do_sample=True, 
                    top_p=0.95,
                    temperature=args.temperature,
                    max_length=min(MAX_LENGTH, input_ids.size(1) + EXTENSION_LENGTH),
                    num_return_sequences=args.num_candidates_generated,
                    # output_scores=True,
                    # return_dict_in_generate=True,
                )
                # batch_output_ids = ret.sequences
                if args.debug:
                    print("input size:")
                    print(input_ids.size())
                    print("output size:")
                    print(batch_output_ids.size())
                batch_output_ids = batch_output_ids[:,input_ids.size(1):]

                # # ncg x num_tokens_generated x vocab
                # scores = torch.stack(ret.scores, 1)
                # assert scores.size(0) == args.num_candidates_generated
                # assert batch_output_ids.size(1) == scores.size(1)
                # scores_selected = scores.gather(-1, batch_output_ids.unsqueeze(-1)).squeeze(-1)
                # assert scores_selected.size(0) == batch_output_ids.size(0) == args.num_candidates_generated
                # assert scores_selected.size(1) == batch_output_ids.size(1)
                # assert batch_output_ids.size(1) + input_ids.size(1) == ret.sequences.size(1)

                output_strs = []
                for output_ids in batch_output_ids:
                    if stop_words:
                        _, output_str, _ = truncate_at_stop_words(tokenizer, stop_words, output_ids)
                    else:
                        output_str = tokenizer.decode(output_ids)
                    output_strs.append(output_str)
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
            output_strs = ["" for _ in range(args.num_candidates_generated)]
        end = time.time()

        for i in range(len(output_strs)):
            output_str = output_strs[i]
            if args.peeking == 1.0:
                output_str = sample_sol
            elif len(output_str):
                output_str = output_str.replace("<|endoftext|>", "")
                # split_str = "<cell>\n" if notebook_formatting else "ANSWER:\n"
                # output_str = output_str.split(split_str)[1].replace("<|endoftext|>", "")
            output_strs[i] = output_str

        # Save the generated sol
        gpt_codes[index+args.start] = output_strs

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f)


if __name__ == "__main__":
    import argparse

    import sys
    print(' '.join(sys.argv))

    MODEL_ARCHS = transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + [
        "facebook/incoder-6B",
        "facebook/incoder-1B",
        ]

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2", choices=MODEL_ARCHS)
    parser.add_argument("-t","--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("--train_loc", default="~/apps/data_split/train.json", type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("--num-candidates-generated", default=1, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("--shard", default=None, type=int)
    parser.add_argument("--num_shards", default=10, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--formatting_type", choices=FORMATTING_TYPES)

    parser.add_argument("--k_shot_prompts", type=int, default=0)
 
    args = parser.parse_args()

    main(args)
