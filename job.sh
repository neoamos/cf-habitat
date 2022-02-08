#!/bin/bash
#SBATCH -J test-hab
#SBATCH --mail-type=ALL
# Please make sure pathes are correct and directories exist beforehand::
#SBATCH -e /home/an38gezy/cf-habitat/job_out/job5.err.%j
#SBATCH -o /home/an38gezy/cf-habitat/job_out/job5.out.%j
#                           instead of <Job_Name>, you can use %x (your Job_Name given above with '-J')
#
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 24                 # 24 CPU cores per process 
#                               can be referenced as $SLURM_CPUS_PER_TASK​ in the "payload" part
#SBATCH --mem-per-cpu=3600    # Main memory in MByte per CPU core
#SBATCH -t 00:15:00           # in hours:minutes, or '#SBATCH -t 10' - just minutes

# GPU specification
#SBATCH --gres=gpu:v100:1     # 2 GPUs of type NVidia "Volta 100"

# -------------------------------
# your real job commands, eg.
module purge
module load gcc cuda
nvidia-smi 1>&2


cd /home/an38gezy/cf-habitat

export MAGNUM_GPU_VALIDATION=on
python train.py
