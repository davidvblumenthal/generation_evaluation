#!/bin/bash
#SBATCH --job-name generation # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/evaluation/generation/python-scripts"
pushd $WORKING_DIR
#echo "Running script with sampling!!"

#python gpt_neo_125m.py \
#        --model_path /pfs/work7/workspace/scratch/ukmwn-training_hug/1.3B_standard_wiki \
#        --tokenizer EleutherAI/gpt-neo-1.3B \
#        --model_name standard_wiki_1.3B  \
#        --output_path ./gen_outputs \
#        --sampling

#echo "Starting to run script without sampling!!"

python generation_pipeline.py \
        --model_path EleutherAI/pythia-1.4b-deduped \
        --tokenizer EleutherAI/pythia-1.4b-deduped \
        --model_name Pythia-1.4_deduped \
        --output_path ../gen_outputs


# Addtional arguments
# --trained_with_padding \
# --return_full_text
# --return_tensors \