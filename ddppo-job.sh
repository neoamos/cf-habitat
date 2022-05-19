#!/bin/bash
#SBATCH -J test-hab
#SBATCH --mail-type=ALL
# Please make sure pathes are correct and directories exist beforehand::
#SBATCH -e /home/an38gezy/thesis/cf-habitat/data/experiments/job_out/job.err.%j
#SBATCH -o /home/an38gezy/thesis/cf-habitat/data/experiments/job_out/job.out.%j
#                           instead of <Job_Name>, you can use %x (your Job_Name given above with '-J')
#
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 128                 # 24 CPU cores per process 
#                               can be referenced as $SLURM_CPUS_PER_TASK​ in the "payload" part
#SBATCH --mem-per-cpu=7200    # Main memory in MByte per CPU core
#SBATCH -t 17:00:00           # in hours:minutes, or '#SBATCH -t 10' - just minutes

# GPU specification
#SBATCH -C dgx
#SBATCH --gres=gpu:a100:8     # 2 GPUs of type NVidia "Volta 100"

# -------------------------------
# your real job commands, eg.
module purge
module load gcc cuda
nvidia-smi 1>&2


cd /home/an38gezy/thesis/habitat-lab
pytest test/test_ddppo_reduce.py

cd /home/an38gezy/thesis/cf-habitat

# export MAGNUM_GPU_VALIDATION=on
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 8 \
    main.py \
    --exp-config configs/experiments/ddppo_pointnav_gibson4plus_mobilenet_lstm1.yaml \
    --run-type train

# python main.py \
#     --exp-config configs/experiments/ddppo_pointnav_gibson0plus_resnet50.yaml \
#     --run-type eval

# while true
# do
#   sleep 30
#   nvidia-smi
#   free -mh
# done