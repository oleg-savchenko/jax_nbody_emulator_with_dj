#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=0:20:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err
## Activate right env

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
# module load OpenMPI/4.1.4-GCC-11.3.0
# module load FFTW.MPI/3.3.10-gompi-2022a
# module load GSL/2.7-GCC-11.3.0
# module load HDF5/1.12.2-gompi-2022a
# module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# module load UCX/1.12.1-GCCcore-11.3.0
export OMP_NUM_THREADS=18
export PYTHONNOUSERSITE=0

cd /home/osavchenko/jax_nbody_emulator

# python scripts/run_emulator.py \
#   --seed 42 \
#   --res 512 \
#   --boxsize 1000 \
#   --ndiv 2,2,2 \
#   --emu-precision f16 \
#   --num-sims 1 \
#   --output-dir outputs/discodj_emulator_seed42_res512_box1000

# python scripts/run_emulator.py \
#   --seed 42 \
#   --res 128 \
#   --boxsize 250 \
#   --ndiv 1 \
#   --emu-precision f32 \
#   --num-sims 3 \
#   --output-dir outputs/discodj_emulator_seed42_res128_box250

python scripts/run_emulator.py \
  --seed 42 \
  --n-part 256 \
  --res 128 \
  --boxsize 500 \
  --ndiv 1,1,1 \
  --num-sims 1 \
  --emu-precision f16 \
  --output-dir outputs/discodj_emulator_res256_box500 \
  --no-compute-vel \
  --mas-worder 2

# python scripts/quijote_comparison.py
  
