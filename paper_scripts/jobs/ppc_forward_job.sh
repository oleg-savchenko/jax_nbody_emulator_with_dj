#!/bin/bash
#SBATCH --job-name=ppc_forward
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
# 1) Default run:
#    sbatch paper_scripts/jobs/ppc_forward_job.sh
# 2) Run first 100 samples, force recompute even if outputs exist:
#    sbatch --export=ALL,NUM_SAMPLES=100,REUSE_EXISTING=0 paper_scripts/jobs/ppc_forward_job.sh
# 3) Continue from sample index 100:
#    sbatch --export=ALL,NUM_SAMPLES=100,SAMPLE_START=100,REUSE_EXISTING=0 paper_scripts/jobs/ppc_forward_job.sh
# 4) Override key emulator settings:
#    sbatch --export=ALL,N_PART=512,RES=128,NDIV=4,2,2,EMU_PRECISION=f16 paper_scripts/jobs/ppc_forward_job.sh

SAMPLES_DIR=${SAMPLES_DIR:-/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD}
SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
NUM_SAMPLES=${NUM_SAMPLES:-10}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
SAMPLE_FOLDER_TEMPLATE=${SAMPLE_FOLDER_TEMPLATE:-sample_{sample_id:04d}}

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
DELTA_UPSAMPLE_METHOD=${DELTA_UPSAMPLE_METHOD:-mode_inject}
MODE_INJECT_SEED_POLICY=${MODE_INJECT_SEED_POLICY:-sample_id}
MODE_INJECT_FIXED_SEED=${MODE_INJECT_FIXED_SEED:-0}
DOWNSAMPLE_METHOD=${DOWNSAMPLE_METHOD:-fourier_truncation}
DOWNSAMPLE_THREADS=${DOWNSAMPLE_THREADS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}
COMPUTE_VEL=${COMPUTE_VEL:-0}
REUSE_EXISTING=${REUSE_EXISTING:-1}

AGGREGATE_DIR=${AGGREGATE_DIR:-${SAMPLES_DIR}/ppc_results}
AGGREGATE_PREFIX=${AGGREGATE_PREFIX:-posterior_predictive}

EXTRA_ARGS=()
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${COMPUTE_VEL}" -eq 1 ]; then EXTRA_ARGS+=(--compute-vel); else EXTRA_ARGS+=(--no-compute-vel); fi
if [ "${REUSE_EXISTING}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi

srun python paper_scripts/ppc_forward.py \
  --samples-dir "${SAMPLES_DIR}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --sample-folder-template "${SAMPLE_FOLDER_TEMPLATE}" \
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
  --mode-inject-seed-policy "${MODE_INJECT_SEED_POLICY}" \
  --mode-inject-fixed-seed "${MODE_INJECT_FIXED_SEED}" \
  --downsample-method "${DOWNSAMPLE_METHOD}" \
  --downsample-threads "${DOWNSAMPLE_THREADS}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --aggregate-prefix "${AGGREGATE_PREFIX}" \
  "${EXTRA_ARGS[@]}"
