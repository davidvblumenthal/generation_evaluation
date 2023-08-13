#!/bin/bash
#SBATCH --job-name generation-2.7B_standard_wiki # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=30gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc

############ TASK ARRAY STUFF ############
LIST_MODELS=(
        "EleutherAI/gpt-neo-1.3B"
        "davidvblumenthal/GPT-Verite-125M-padding"
        )

MODEL=${LIST_MODELS[$SLURM_ARRAY_TASK_ID]}

echo "MODEL_NAME "$MODEL
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID

############ JOB STUFF ############
module load devel/cuda/11.7

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/evaluation/gpt-neo/python-scripts"
pushd $WORKING_DIR

# CUT THE WORKSPACE FROM THE MODEL STRING
OUTPUT_NAME=$(echo "$MODEL" | cut -d'/' -f2)

echo "$OUTPUT_NAME"

python perplexity.py \
        --model_path ${MODEL} \
        --tokenizer ${MODEL} \
        --dataset perplexity_factualityprompts \
        --output_path perplexity_${OUTPUT_NAME}.json

