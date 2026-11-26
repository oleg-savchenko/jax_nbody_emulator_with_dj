#!/bin/bash
#SBATCH --job-name=true_ic_control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate

cd /home/osavchenko/jax_nbody_emulator

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export PYTHONNOUSERSITE=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

# HOW TO RUN (examples):
# 1) Default control run:
#    sbatch paper_scripts/jobs/true_ic_upsampling_control_job.sh
# 2) Force recompute forward outputs:
#    sbatch --export=ALL,REUSE_EXISTING_FORWARD=0 paper_scripts/jobs/true_ic_upsampling_control_job.sh
# 3) Match PPC settings explicitly:
#    sbatch --export=ALL,N_PART=512,RES=128,NDIV=4,2,2,EMU_PRECISION=f16,DELTA_UPSAMPLE_METHOD=mode_inject paper_scripts/jobs/true_ic_upsampling_control_job.sh
# 4) Test without MAS deconvolution:
#    sbatch --export=ALL,DECONVOLVE_MAS=0 paper_scripts/jobs/true_ic_upsampling_control_job.sh

TARGET_PATH=${TARGET_PATH:-/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt}
IC_KEY=${IC_KEY:-delta_z127}
FINAL_KEY=${FINAL_KEY:-delta_z0}
Z_IC=${Z_IC:-127.0}
Z_TARGET=${Z_TARGET:-0.0}
IC_RESCALE_MODE=${IC_RESCALE_MODE:-growth}

N_PART=${N_PART:-512}
RES=${RES:-128}
BOXSIZE=${BOXSIZE:-1000}
NDIV=${NDIV:-4,2,2}
EMU_PRECISION=${EMU_PRECISION:-f16}
NUM_SIMS=${NUM_SIMS:-1}
MAS_WORDER=${MAS_WORDER:-4}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
TARGET_DECONVOLVED=${TARGET_DECONVOLVED:-1}
DELTA_UPSAMPLE_METHOD=${DELTA_UPSAMPLE_METHOD:-mode_inject}
DOWNSAMPLE_METHOD=${DOWNSAMPLE_METHOD:-fourier_truncation}
DOWNSAMPLE_THREADS=${DOWNSAMPLE_THREADS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}
PK_THREADS=${PK_THREADS:-1}

OUTPUT_DIR=${OUTPUT_DIR:-/home/osavchenko/jax_nbody_emulator/outputs/true_ic_upsampling_control}
REUSE_EXISTING_FORWARD=${REUSE_EXISTING_FORWARD:-1}

EXTRA_ARGS=()
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${TARGET_DECONVOLVED}" -eq 1 ]; then EXTRA_ARGS+=(--target-deconvolved); else EXTRA_ARGS+=(--no-target-deconvolved); fi
if [ "${REUSE_EXISTING_FORWARD}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing-forward); else EXTRA_ARGS+=(--no-reuse-existing-forward); fi

srun python paper_scripts/true_ic_upsampling_control.py \
  --target-path "${TARGET_PATH}" \
  --ic-key "${IC_KEY}" \
  --final-key "${FINAL_KEY}" \
  --z-ic "${Z_IC}" \
  --z-target "${Z_TARGET}" \
  --ic-rescale-mode "${IC_RESCALE_MODE}" \
  --n-part "${N_PART}" \
  --res "${RES}" \
  --boxsize "${BOXSIZE}" \
  --ndiv "${NDIV}" \
  --emu-precision "${EMU_PRECISION}" \
  --num-sims "${NUM_SIMS}" \
  --mas-worder "${MAS_WORDER}" \
  --delta-upsample-method "${DELTA_UPSAMPLE_METHOD}" \
  --downsample-method "${DOWNSAMPLE_METHOD}" \
  --downsample-threads "${DOWNSAMPLE_THREADS}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --pk-threads "${PK_THREADS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
