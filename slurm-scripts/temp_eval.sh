#!/bin/bash
#SBATCH --job-name generation-1.3B_coref # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=5:30:00
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

echo "Running script with sampling!!"

python gpt_neo_125m.py \
        --model_path /pfs/work7/workspace/scratch/ukmwn-training_hug/1.3B_coref_wiki/checkpoint-8691 \
        --tokenizer EleutherAI/gpt-neo-1.3B \
        --model_name coref_1.3B \
        --output_path ./gen_outputs \
        --sampling

echo "Starting to run script without sampling!!"

python gpt_neo_125m.py \
        --model_path /pfs/work7/workspace/scratch/ukmwn-training_hug/1.3B_coref_wiki/checkpoint-8691 \
        --tokenizer EleutherAI/gpt-neo-1.3B \
        --model_name coref_1.3B \
        --output_path ./gen_outputs \

# python gpt_neo_125m.py \ 
# --model_path  /pfs/work7/workspace/scratch/ukmwn-training_hug/125m_coref_wiki/checkpoint-45000\
# --model_name standard_1.3B \
# --tokenizer EleutherAI/gpt-neo-1.3B \  EleutherAI/gpt-neo-125M
# --output_path ./debugging