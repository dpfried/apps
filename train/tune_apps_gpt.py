"""
Tune LM on Code
"""

import io
import logging
import math
import os
import pprint
import sys
import time
import json

import numpy as np

import transformers
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, EvalPrediction

from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss

from dataset_lm.base_lm_dataset import BaseLMDataset
from dataset_apps.APPSBaseDataset import APPSBaseDataset
from CustomTensorboardCallback import CustomTensorBoardCallback

BIG_NEG = torch.tensor(-1.0e9)

# torch.set_num_threads(2)

# https://github.com/pytorch/pytorch/issues/11201
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class NoCheckpointingFilter(logging.Filter):
    def filter(self, record):
        return 'is incompatible with gradient checkpointing' not in record.getMessage()

def compute_metrics(eval_preds: EvalPrediction):
    log_probs = eval_preds.predictions
    labels = eval_preds.label_ids
    labels = labels[:, 1:]
    inputs = eval_preds.inputs

    # for some reason, HF pads log_probs using -100 to fill space
    # https://github.com/huggingface/transformers/blob/7999ec125fc31428ed6879bf01bb013483daf704/src/transformers/trainer.py#L2762
    log_probs[log_probs == -100] = 0
    num_tokens = (labels >= 0).sum()

    # we've already zero-ed out the non-masked tokens, so we can just sum them
    # log_probs is a float16 numpy array. convert to float32 to avoid overflow
    total_log_prob = np.float32(log_probs).sum()

    mean_log_prob = total_log_prob / num_tokens

    perplexity = np.exp(-mean_log_prob)

    d = {
        "total_log_prob": total_log_prob,
        "mean_log_prob": mean_log_prob,
        "num_tokens": num_tokens,
        "perplexity": perplexity,
    }
    return d

def run_training(args):

    ## Checkpoint Loading ######################################################## 
    if args.load:
        if '2700' in args.load:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.load)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Loaded model from {args.load}")
    else:
        if "EleutherAI" in args.arch:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.arch)
        elif "facebook" in args.arch:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.arch)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)

    if ('EleutherAI' in args.arch or '2700' in args.arch):
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif 'facebook' in args.arch:
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/incoder-1B")
    elif 'gpt' in args.arch: # Should handle GPT-2 and GPT-Neo
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)
    elif args.arch in {'codebert'}:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    else:
        raise NotImplementedError()

    train_data, valid_data = get_dataset(args, tokenizer)

    if args.apps_sample_mode == 'example_only':
        collator = None
        # print("padding out examples")
        # tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        # collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        collator = None

    if args.resume:
        raise NotImplementedError
        model = transformers.GPT2LMHeadModel.from_pretrained(args.resume)
        print(f"Loaded model from {args.resume}")
        start_epoch = 0
        start_iteration = int(args.resume.split("-")[-1])
        print("start_iteration = ", start_iteration)
    else:
        start_iteration = 0

    # suppress warnings about use_cache=True being incompatible with gradient checkpointing
    logger = logging.getLogger(model.__module__)
    logger.addFilter(NoCheckpointingFilter())

    ## Dataloading ######################################################## 
    train_data.start_iteration = start_iteration

    ## Start Loop ########################################################
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=valid_data is not None,
        do_predict=False,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        # per_device_eval_batch_size=args.batch_size_per_replica // 2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc_steps,

        # eval_accumulation_steps=1,
        prediction_loss_only=False,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.lr_warmup_steps,
        # max_grad_norm=1.0,

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        eval_steps=args.eval_freq,

        save_steps=args.save_freq or args.eval_freq,
        evaluation_strategy="steps" if valid_data is not None else "no",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,

    )

    def preprocess_logits_for_metrics(logits, labels):
        labels = labels[:, 1:]
        logits = logits[:, :-1]
        log_probs = logits.log_softmax(-1)
        # the dataset uses -100 for tokens that should not have a loss applied
        # clamp the labels so that we can gather with them
        labels_clamped = labels.clamp(min=0)
        lps = log_probs.gather(-1, labels_clamped.unsqueeze(-1)).squeeze(-1)
        # change the -100 tokens to have 0 log probs, so that they will be ignored when we sum log probs
        lps = torch.where(labels >= 0, lps, torch.tensor(0.0).to(log_probs))
        return lps

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # print(f"is_model_parallel: {trainer.is_model_parallel}")

    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())

    trainer.train()
    
    if args.local_rank == 0:
        model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def get_dataset(args, tokenizer): 
    
    fnames = os.listdir(args.apps_train_files)

    if args.frac_valid_data is not None:
        import random
        fnames = list(sorted(fnames))
        random.seed(1)
        random.shuffle(fnames)
        n_valid = int(len(fnames) * args.frac_valid_data)
        valid_fnames = fnames[-n_valid:]
        train_fnames = fnames[:-n_valid]
        assert not (set(valid_fnames) & set(train_fnames))
        print(f"{len(train_fnames)} train instances and {len(valid_fnames)} validation instances")
    else:
        train_fnames = fnames
        valid_fnames = []

    max_tokens = 2048 if ('EleutherAI' in args.arch or 'facebook' in args.arch or '2700' in args.load) else 1024
 
    train_data = APPSBaseDataset(
        dataroot=args.apps_dataroot, 
        problem_dirs=train_fnames,
        mode=args.arch, 
        max_tokens=max_tokens,
        sample_mode=args.apps_sample_mode,
        tokenizer=tokenizer,
    )
    if valid_fnames:
        valid_data = APPSBaseDataset(
            dataroot=args.apps_dataroot, 
            problem_dirs=valid_fnames,
            mode=args.arch, 
            max_tokens=max_tokens,
            # we should only include each example a single time in validation,
            # rather than packing to the max length as we do in training for
            # other sample modes
            sample_mode='example_only',
            # sample_mode=args.apps_sample_mode,
            tokenizer=tokenizer,
        )
    else:
        valid_data = None

    return train_data, valid_data

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    run_training(args)


if __name__ == "__main__":
    import argparse
    print(' '.join(sys.argv))

    MODEL_ARCHS = transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + [
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "facebook/incoder-6B",
        "facebook/incoder-1B",
        ]

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=MODEL_ARCHS)
    parser.add_argument('--dummy-model', action='store_true')
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)

    # Dataloading
    parser.add_argument('--apps-dataroot', default='../apps/', type=str)
    parser.add_argument('--apps-train-files', default='../apps/data_split/train.json', type=str)
    parser.add_argument('--apps-sample-mode', choices=['uniform_sol', 'uniform_prob', 'example_only'], default='uniform_sol')
    
    # Training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--lr-warmup-steps', default=0, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--gradient-checkpointing', default=False, action='store_true')

    parser.add_argument('--frac-valid-data', type=float, default=0.05)

    # Logging and stuff
    parser.add_argument('--save-dir', default="checkpoints/TEMP", type=str)
    parser.add_argument('--log-freq', default=5, type=int)
    parser.add_argument('--eval-freq', default=100, type=int)
    parser.add_argument('--save-freq', type=int, help='will be set to eval-freq if not passed')
    parser.add_argument('--save-total-limit', type=int)

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%m-%d-%Y__%H:%M:%S"))

    main(args)
