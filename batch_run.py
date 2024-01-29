"""
This file contains the code to run a batch of jobs on a cluster. It is meant to be used 
with a runfile that specifies the command for each job. The runfile should contain one
command per line. Lines starting with '#' will be ignored. Empty lines will be ignored.
"""

import os
import datetime

LOGGER="--log.logger-type wandb"
# SEEDS=[860243, 337454, 418653, 484615, 259303, 405532, 167707, 743908, 278058, 527888]
SEEDS=[484615]
ALL_ARGS = ""

def get_jobfile(cmd: str, jobname: str = None) -> str:
    """
    Generate a jobfile for submitting a job to a cluster.

    Args:
        cmd (str): The command to be executed in the job.
        jobname (str, optional): The name of the job. If not provided, the current timestamp will be used.

    Returns:
        str: The generated jobfile as a string.
    """
    if jobname is None:
        jobname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cwd = os.getcwd()
    jobfile = f"""#!/bin/bash

    ## Resource Request
    #SBATCH --job-name={jobname}
    #SBATCH --output=jobfiles_out/{jobname}.out
    #SBATCH --time=0-01:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=20G
    #SBATCH --gpus=1

    ## Job Steps
    cd {cwd}
    source activate sde
    srun {cmd}
    """

    return jobfile.replace("    ", "")

def get_command_list(filename: str) -> list[str]:
    """
    Read the contents of a file and return a list of non-empty lines that do not start with '#'.

    Args:
        filename (str): The path to the file.

    Returns:
        list[str]: A list of non-empty lines from the file.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]
    lines = [line for line in lines if not line.startswith("#")]
    return lines

def commands_to_seeds(commands: list[str], seeds: list[int]) -> list[str]:
    """
    Generates a list of command-line strings by combining each command with each seed.

    Args:
        commands (list[str]): A list of command-line strings.
        seeds (list[int]): A list of seed values.

    Returns:
        list[str]: A list of command-line strings with seed values appended.
    """
    command_seed_list = [cmd + f" --training.seed {seed}" for cmd in commands for seed in seeds]
    return command_seed_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", type=str)
    parser.add_argument("--retrain", type=str)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--add_seeds", action="store_true")
    parser.add_argument("--add_logger", action="store_true")
    parser.add_argument("--add_name", action="store_true")
    args = parser.parse_args()

    commands = get_command_list(args.run_file)
    if args.add_seeds:
        commands = commands_to_seeds(commands, SEEDS)
    if not args.add_logger:
        LOGGER = ""

    commands = [cmd + " " + LOGGER + " " + ALL_ARGS for cmd in commands]

    if not args.retrain:
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        dt = args.retrain

    for i, cmd in enumerate(commands):
        rid = str(i+30).zfill(4)
        jobname = dt + '-' + rid

        if args.add_name:
            cmd += " --log.name " + jobname
        
        if args.retrain:
            cmd += " --training.retrain"

        jobfile = get_jobfile(cmd, jobname)

        if args.dry:
            print(cmd)
            continue

        with open(f"jobfiles/{jobname}.job", "w") as f:
            f.write(jobfile)

        os.system(f"sbatch jobfiles/{jobname}.job")
