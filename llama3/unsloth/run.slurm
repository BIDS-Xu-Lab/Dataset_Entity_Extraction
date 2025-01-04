#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fine_tune_unsloth_filtered
#SBATCH --output=/home/gy237/project/llama3/unsloth/slrum_output/fine_tune_unsloth_filtered.out
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --gpus=a100:1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=ALL

echo '-------------------------------------------------'
echo "Job Name: ${SLURM_JOB_NAME}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s) on partition ${SLURM_JOB_PARTITION}"
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo "WorkDir is ${SLURM_SUBMIT_DIR}"
echo '-------------------------------------------------'
echo -e '\n\n'

source /home/gy237/anaconda3/bin/activate unsloth3.9
echo PATH: $PATH
echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH
echo CUDA_HOME: $CUDA_HOME
echo '-------------------------------------------------'
echo -e '\n\n'

output_model=./output
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi

python fine-tuning.py