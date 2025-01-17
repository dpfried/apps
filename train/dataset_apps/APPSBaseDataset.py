"""
Dataset to be used for APPS Training
"""

import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from collections import namedtuple
from typing import List

from dataset_lm.reindent import run as run_reindent
from tqdm import tqdm 

import json

Sample = namedtuple("Sample", ["question_str", "starter_code", "sol_str", "answer_type", "metadata"])

FORMATTING_TYPES = ["qa", "notebook", "stackoverflow"]

def do_reindent_code(codestr):
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
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, mode, max_tokens, sample_mode, tokenizer, formatting_type=None, reindent_code=True):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs # Loaded from train/test split json files
        if formatting_type is None:
            formatting_type = 'notebook' if 'facebook' in self.mode else 'qa'
        self.formatting_type = formatting_type
        self.reindent_code = reindent_code

        self.mode = mode
        self.sample_mode = sample_mode # Either "uniform_sol" or "uniform_prob"
        self.max_tokens = max_tokens

        self.samples = []           # Should be set in initialize()
        self.initialize()

        self.tokenizer = tokenizer

    @staticmethod
    def load_samples(dataroot, problem_name, require_solutions=True, reindent_code=True) -> List[Sample]:
        test_case_path = os.path.join(dataroot, problem_name, "input_output.json")
        question_fname = os.path.join(dataroot, problem_name, "question.txt")
        sols_fname = os.path.join(dataroot, problem_name, "solutions.json")
        starter_code = os.path.join(dataroot, problem_name, "starter_code.py")
        metadata = os.path.join(dataroot, problem_name, "metadata.json")

        # with open(test_case_path, "r") as f:
        #     test_data = json.load(f)
        # print(question_fname)

        if os.path.exists(starter_code):
            answer_type = "\nUse Call-Based format\n"
            # assert bool(test_data.get("fn_name"))
        else:
            answer_type = "\nUse Standard Input format\n"
            # assert not bool(test_data.get("fn_name"))

        if (not os.path.isfile(question_fname)):
            return None

        if (os.path.isfile(starter_code)):
            with open(starter_code, 'r') as f:
                starter_code = f.read()
            if reindent_code:
                starter_code = do_reindent_code(starter_code)
        else:
            starter_code = ""

        if (os.path.isfile(metadata)):
            with open(metadata, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = None

        # Read the question description
        with open(question_fname, 'r') as f:
            question_str = f.read()

        if not os.path.isfile(sols_fname):
            if require_solutions:
                return None
            sols_str_list = ['']
        else:
            with open(sols_fname, 'r') as f:
                sols_str_list = [do_reindent_code(sol_str) if reindent_code else sol_str for sol_str in json.load(f)]

        samples = []
        # Read all the solutions
        for sol_str in sols_str_list:
            sample = Sample(question_str, starter_code, sol_str, answer_type, metadata=metadata)
            samples.append(sample)
        return samples

    @staticmethod
    def prompt_from_sample(sample: Sample, formatting_type = "qa"):
        assert formatting_type in FORMATTING_TYPES
        if formatting_type == "notebook":
            prompt = "<text>\n" + sample.question_str 
            if sample.starter_code:
                prompt += "\n</text>\n<code>\n"
                prompt += sample.starter_code
                prompt += "\n</code>\n<text>\n"
            prompt += "\n" + sample.answer_type
            prompt += "\n</text>\n"
            prompt += "\n<cell>\n"
        elif formatting_type == "stackoverflow":
            prompt = "<| q dscore=3 tags=python >\n" + sample.question_str 
            prompt += "\n" + sample.answer_type
            if sample.starter_code:
                # sol_str_tok = sample.sol_str.split()
                # starter_code_tok = sample.starter_code.split()
                # if not sol_str_tok[:len(starter_code_tok)] == starter_code_tok:
                #     print("starter_code")
                #     print(sample.starter_code)
                #     print("sol_str")
                #     print(sample.sol_str)
                #     assert False
                prompt += "\n<code>\n"
                prompt += sample.starter_code
                prompt += "\n</code>"
            prompt += "\n<|/ q |>\n<| a dscore=3 tags=python |>\n"
            prompt += "<code>\n"
            # print(prompt)
            # print("solution:")
            # print(sample.sol_str)
        elif formatting_type == "qa":
            prompt =  "\nQUESTION:\n" + sample.question_str + "\n" + sample.starter_code + "\n" + sample.answer_type + "\nANSWER:\n"
        else:
            raise NotImplementedError(f"invalid formatting_type {formatting_type}")
        return prompt

    def initialize(self):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        skipped_problems = []

        all_samples_dict = {} # Mapping from question_fname to list of samples

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):
            samples = self.load_samples(self.dataroot, problem_name, reindent_code=self.reindent_code)
            if samples is not None:
                all_samples.extend(samples)
                question_str = samples[0][0]
                assert question_str not in all_samples_dict, (problem_name, question_str)
                all_samples_dict[question_str] = samples
            else:
                skipped_problems.append(problem_name)
        
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        self.samples = all_samples
        self.samples_dict = all_samples_dict


    def __len__(self):
        return len(self.samples)


    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 

        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix, curr_q_meta = self.samples[idx]
        elif self.sample_mode == 'uniform_prob':
            curr_q = random.choice(list(self.samples_dict.keys()))
            curr_q, curr_s, curr_a, curr_q_prefix, curr_q_meta = random.choice(self.samples_dict[curr_q])
        elif self.sample_mode == 'example_only':
            return [self.samples[idx]]
        else:
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            if self.mode in {'codebert'}:
                curr_q = curr_q.replace('\t', '\0')
                curr_s = curr_s.replace('\t', '\0')
                curr_a = curr_a.replace('\t', '\0')

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append(Sample(curr_q, curr_s, curr_a, curr_q_prefix, metadata=curr_q_meta))

            if self.sample_mode == 'uniform_sol':
                curr_q, curr_s, curr_a, curr_q_prefix, curr_q_meta = random.choice(self.samples)
            elif self.sample_mode == 'uniform_prob':
                curr_q = random.choice(list(self.samples_dict.keys()))
                curr_q, curr_s, curr_a, curr_q_prefix, curr_q_meta = random.choice(self.samples_dict[curr_q])
            else:
                raise NotImplementedError()

        return curr_samples

    def __getitem__(self, idx):
        
        raw_samples = self.pack_samples(idx)

        if 'gpt' in self.mode or self.mode in {'codebert'} or 'facebook' in self.mode:
            retval = sample_gpt_task(
                raw_samples,
                max_tokens=self.max_tokens, 
                tokenizer=self.tokenizer, 
                formatting_type=self.formatting_type
            )
        else:
            raise NotImplementedError()
    
        gc.collect()
        return retval

def sample_gpt_task(raw_samples: List[Sample], max_tokens, tokenizer, formatting_type:str = "qa"):
    """
    Create the true sample used for the GPT task
    """

    input_ids = []
    label_ids = []
    
    for sample_ix, sample in enumerate(raw_samples):
        prompt = APPSBaseDataset.prompt_from_sample(sample, formatting_type)

        # Loss is not calculated on this
        logging.debug("--prompt--")
        logging.debug(prompt)
        logging.debug("--sol_str--")
        logging.debug(sample.sol_str)

        # incoder tokenizer (and fairseq-imported models generally) add a BOS
        # token, which we *do* want at the beginning of the document, but not
        # elsewhere
        prompt_token_ids = tokenizer.encode(prompt, verbose=False, add_special_tokens=sample_ix == 0)
        answer_token_ids   = tokenizer.encode(sample.sol_str, verbose=False, add_special_tokens=False)

        input_ids.extend(prompt_token_ids)
        input_ids.extend(answer_token_ids)
        
        label_ids.extend([-100] * len(prompt_token_ids))
        label_ids.extend(answer_token_ids)
        if formatting_type in ["notebook", "stackoverflow"]:
            if formatting_type == "notebook":
                # TODO fix this so it's a closing tag
                closing_ids = tokenizer.encode("\n<code>\n", verbose=False, add_special_tokens=False)
            else:
                closing_ids = tokenizer.encode("\n<|/ a |>\n", verbose=False, add_special_tokens=False)
            assert None not in closing_ids, closing_ids
            input_ids.extend(closing_ids)
            label_ids.extend([-100] * len(closing_ids))

        input_ids.append(tokenizer.bos_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id)
        label_ids.append(-100)
    
    # Sanity check
    assert len(input_ids) == len(label_ids)

    if len(input_ids) < max_tokens:
        pass
        # print("short doc:", len(input_ids))
        # import pdb; pdb.set_trace()

    # Cut off the excess
    input_ids = input_ids[:max_tokens]
    label_ids = label_ids[:max_tokens]

    assert None not in input_ids, input_ids

    return {
        "input_ids" : torch.LongTensor(input_ids),
        "labels" :  torch.LongTensor(label_ids)
    }


if __name__ == '__main__':
    import json

    # Do sanity checking
    with open("~/apps/data_split/train.json") as f:
        fnames = json.load(f)
    
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    dataset = APPSBaseDataset(
        dataroot='~/apps/', 
        problem_dirs=fnames,
        mode='gpt2', 
        max_tokens=1024
    )

    e = dataset[0]
    print(e)
    print("------- input_ids ------------------------------------------------------------------------------------")
    print(tokenizer.decode(e['input_ids']))
    print("------- labels ------------------------------------------------------------------------------------")
    labels = e['labels']
    labels[labels == -100] = tokenizer.eos_token_id
    labels_str = tokenizer.decode(labels)
    print(labels_str)

    import pdb; pdb.set_trace()