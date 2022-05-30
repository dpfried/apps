import os
import sys
import json
import subprocess
import pandas
import argparse
import glob

def parse_file(log_file):
    train_stats = []
    val_stats = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            line = line.replace('\'', '"')
            line = line.replace('"eval_loss": nan,', '')
            if line.startswith('{') and line.endswith('}'):
                this_stats = json.loads(line)
                if any(key.startswith('eval_') for key in this_stats):
                    stats = val_stats
                else:
                    stats = train_stats
                stats.append(this_stats)
            if 'Saving model checkpoint:' in line:
                model_dir = os.path.dirname(os.path.dirname(line.split('checkpoint:')[1].strip()))
                assert os.path.exists(model_dir)
                val_stats[-1]['model_path'] = model_dir
    return pandas.DataFrame(train_stats), pandas.DataFrame(val_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_globs", nargs="+")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--remove_others", action="store_true")
    args = parser.parse_args()
    error_files = []
    for log_glob in args.log_globs:
        for log_file in glob.glob(log_glob):
            experiment_root_dir = os.path.dirname(log_file)
            if os.path.exists(os.path.join(experiment_root_dir, "checkpoint-best")):
                print(f"seem to have already processed {experiment_root_dir}; skipping")
                continue
            print(experiment_root_dir)
            try:
                _, val_df = parse_file(log_file)
                best_model_path = None
                for key in ["eval_perplexity", "eval_loss"]:
                    if key in val_df.columns:
                        best_model_path = val_df.sort_values(key).iloc[0].model_path
                        break
                assert best_model_path is not None, "couldn't find best model using validation metric"
            except Exception as e:
                error_files.append((log_file, e))
                continue
            out_model_path = os.path.join(experiment_root_dir, os.path.basename(best_model_path))
            # copy only the model, not the training state
            commands = [
                f"mkdir {out_model_path}",
                f"cp -r {best_model_path}/*.json {out_model_path}",
                f"cp -r {best_model_path}/*.bin {out_model_path}",
                f"cp -r {best_model_path}/*.py {out_model_path}",
                f"cd {experiment_root_dir}; ln -s {os.path.basename(best_model_path)} checkpoint-best",
            ]
            if args.remove_others:
                commands.append(f"rm -r {experiment_root_dir}/*-*-*__*:*:*")
            for command in commands:
                print(command)
                if not args.dry_run:
                    subprocess.run(command, shell=True, check=True)
    if error_files:
        print("error files:")
        for log_file, e in error_files:
            print(log_file, ":")
            print(e)