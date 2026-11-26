#!/bin/bash
#SBATCH --job-name=a2a_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=03:00:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate

cd /home/osavchenko/jax_nbody_emulator

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export PYTHONNOUSERSITE=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_${USER}_${SLURM_JOB_ID:-manual}}
mkdir -p "${MPLCONFIGDIR}"

# HOW TO RUN (examples):
# 1) Default run (k_cut = 1.0 * k_Nyq, 10 posterior samples, run inference):
#    sbatch paper_scripts/jobs/apples_to_apples_pipeline_job.sh
#
# 2) Force fresh recompute (no reuse):
#    sbatch --export=ALL,REUSE_EXISTING=0 paper_scripts/jobs/apples_to_apples_pipeline_job.sh
#
# 3) Use stricter cutoff and paired injection:
#    sbatch --export=ALL,K_CUT_FRACTION=0.9,SAMPLE_INJECT_SEED_POLICY=paired paper_scripts/jobs/apples_to_apples_pipeline_job.sh
#
# 4) Skip inference and use existing samples:
#    sbatch --export=ALL,RUN_INFERENCE=0,EXISTING_SAMPLES_DIR=/path/to/samples,NUM_SAMPLES=10 paper_scripts/jobs/apples_to_apples_pipeline_job.sh
#
# 5) Override run folder name:
#    sbatch --export=ALL,RUN_NAME=apples_to_apples_cut1p0 paper_scripts/jobs/apples_to_apples_pipeline_job.sh

SAMPLES_ROOT=${SAMPLES_ROOT:-/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD}
RUN_NAME=${RUN_NAME:-}
REUSE_EXISTING=${REUSE_EXISTING:-1}

TARGET_PATH=${TARGET_PATH:-/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt}
IC_KEY=${IC_KEY:-delta_z127}
TARGET_KEY=${TARGET_KEY:-delta_z0}
Z_IC=${Z_IC:-127.0}
Z_TARGET=${Z_TARGET:-0.0}

RES=${RES:-128}
N_PART=${N_PART:-512}
BOXSIZE=${BOXSIZE:-1000}
NDIV=${NDIV:-4,2,2}
EMU_PRECISION=${EMU_PRECISION:-f16}
NUM_SIMS=${NUM_SIMS:-1}
MAS_WORDER=${MAS_WORDER:-4}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}

K_CUT_FRACTION=${K_CUT_FRACTION:-1.0}
K_CUT_HMPC=${K_CUT_HMPC:-}
INJECT_PK_SOURCE=${INJECT_PK_SOURCE:-class}
REFERENCE_INJECT_SEED=${REFERENCE_INJECT_SEED:-0}
SAMPLE_INJECT_SEED_POLICY=${SAMPLE_INJECT_SEED_POLICY:-paired}
SAMPLE_INJECT_FIXED_SEED=${SAMPLE_INJECT_FIXED_SEED:-0}

RUN_INFERENCE=${RUN_INFERENCE:-1}
MODEL_DIR=${MODEL_DIR:-}
INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-/home/osavchenko/gaussian_npe/paper_plots_scripts/save_samples.py}
INFERENCE_REPO_DIR=${INFERENCE_REPO_DIR:-/home/osavchenko/gaussian_npe}
INFERENCE_PYTHON=${INFERENCE_PYTHON:-python}
NUM_SAMPLES=${NUM_SAMPLES:-10}
NOISE_SEED=${NOISE_SEED:-42}
EXISTING_SAMPLES_DIR=${EXISTING_SAMPLES_DIR:-}
SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}

PK_THREADS=${PK_THREADS:-1}
SAVE_INJECTED_NPART=${SAVE_INJECTED_NPART:-0}
DRY_RUN=${DRY_RUN:-0}

EXTRA_ARGS=()

if [ "${REUSE_EXISTING}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${RUN_INFERENCE}" -eq 1 ]; then EXTRA_ARGS+=(--run-inference); else EXTRA_ARGS+=(--no-run-inference); fi
if [ "${SAVE_INJECTED_NPART}" -eq 1 ]; then EXTRA_ARGS+=(--save-injected-npart); else EXTRA_ARGS+=(--no-save-injected-npart); fi
if [ "${DRY_RUN}" -eq 1 ]; then EXTRA_ARGS+=(--dry-run); else EXTRA_ARGS+=(--no-dry-run); fi

if [ -n "${RUN_NAME}" ]; then EXTRA_ARGS+=(--run-name "${RUN_NAME}"); fi
if [ -n "${K_CUT_HMPC}" ]; then EXTRA_ARGS+=(--k-cut-hmpc "${K_CUT_HMPC}"); fi
if [ -n "${MODEL_DIR}" ]; then EXTRA_ARGS+=(--model-dir "${MODEL_DIR}"); fi
if [ -n "${EXISTING_SAMPLES_DIR}" ]; then EXTRA_ARGS+=(--existing-samples-dir "${EXISTING_SAMPLES_DIR}"); fi

srun python paper_scripts/apples_to_apples_emulator_pipeline.py \
  --samples-root "${SAMPLES_ROOT}" \
  --target-path "${TARGET_PATH}" \
  --ic-key "${IC_KEY}" \
  --target-key "${TARGET_KEY}" \
  --z-ic "${Z_IC}" \
  --z-target "${Z_TARGET}" \
  --res "${RES}" \
  --n-part "${N_PART}" \
  --boxsize "${BOXSIZE}" \
  --ndiv "${NDIV}" \
  --emu-precision "${EMU_PRECISION}" \
  --num-sims "${NUM_SIMS}" \
  --mas-worder "${MAS_WORDER}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --k-cut-fraction "${K_CUT_FRACTION}" \
  --inject-pk-source "${INJECT_PK_SOURCE}" \
  --reference-inject-seed "${REFERENCE_INJECT_SEED}" \
  --sample-inject-seed-policy "${SAMPLE_INJECT_SEED_POLICY}" \
  --sample-inject-fixed-seed "${SAMPLE_INJECT_FIXED_SEED}" \
  --inference-script "${INFERENCE_SCRIPT}" \
  --inference-repo-dir "${INFERENCE_REPO_DIR}" \
  --inference-python "${INFERENCE_PYTHON}" \
  --num-samples "${NUM_SAMPLES}" \
  --noise-seed "${NOISE_SEED}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --pk-threads "${PK_THREADS}" \
  "${EXTRA_ARGS[@]}"

