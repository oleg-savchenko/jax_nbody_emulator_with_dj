#!/bin/bash
# Auto-submitting launcher for scripts/generate_dataset.py.
# It chooses resources from STAGE:
# - STAGE=emulator or STAGE=all -> GPU job
# - STAGE=halos or STAGE=hod    -> CPU job
#
# HOW TO RUN (recommended; auto resource selection):
# 1) Default emulator stage (GPU):
#    bash my_generate_dataset_job.sh
#
# 2) Halos-only stage (CPU):
#    STAGE=halos DATASET_NAME=box250_npart128_res64_nsims100 bash my_generate_dataset_job.sh
#
# 3) HOD-only stage (CPU):
#    STAGE=hod DATASET_NAME=box250_npart128_res64_nsims100 bash my_generate_dataset_job.sh
#    # Use your usual module/system Python environment (no venv activation):
#    STAGE=hod HOD_USE_SYSTEM_PYTHON=1 DATASET_NAME=box250_npart128_res64_nsims100 bash my_generate_dataset_job.sh
#    # Or use a custom HOD venv:
#    STAGE=hod HOD_VENV_PATH=/path/to/venv/bin/activate DATASET_NAME=... bash my_generate_dataset_job.sh
#
# 4) Emulator stage with custom settings:
#    STAGE=emulator N_SIMS=100 SEED_START=0 N_PART=128 RES=64 BOXSIZE=250 bash my_generate_dataset_job.sh
#
# 5) Keep temporary displacement files:
#    CLEANUP_TEMP_DISPLACEMENT=0 bash my_generate_dataset_job.sh
#
# You can still use sbatch directly; in that case Slurm resources are whatever you request.

set -euo pipefail

REPO_DIR=/home/osavchenko/jax_nbody_emulator
STAGE=${STAGE:-emulator}
AUTO_SUBMIT=${AUTO_SUBMIT:-1}
SBATCH_OUTPUT=${SBATCH_OUTPUT:-${REPO_DIR}/jobs_outputs/%x-%j-%N_slurm.out}
SBATCH_ERROR=${SBATCH_ERROR:-${REPO_DIR}/jobs_outputs/R-%x.%j.err}

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${AUTO_SUBMIT}" -eq 1 ]; then
  case "${STAGE}" in
    halos|hod)
      JOB_NAME=${JOB_NAME:-generate_dataset_${STAGE}}
      CPU_PARTITION=${CPU_PARTITION:-rome}
      CPU_NODES=${CPU_NODES:-1}
      CPU_NTASKS=${CPU_NTASKS:-16}
      CPU_CPUS_PER_TASK=${CPU_CPUS_PER_TASK:-1}
      CPU_TIME=${CPU_TIME:-05:00:00}
      echo "[generate_dataset job] Submitting CPU job for STAGE=${STAGE} (${CPU_PARTITION})."
      exec sbatch \
        --job-name="${JOB_NAME}" \
        --nodes="${CPU_NODES}" \
        --ntasks="${CPU_NTASKS}" \
        --cpus-per-task="${CPU_CPUS_PER_TASK}" \
        --partition="${CPU_PARTITION}" \
        --time="${CPU_TIME}" \
        --output="${SBATCH_OUTPUT}" \
        --error="${SBATCH_ERROR}" \
        --export=ALL,AUTO_SUBMIT=0 \
        "$0"
      ;;
    emulator|all)
      JOB_NAME=${JOB_NAME:-generate_dataset_${STAGE}}
      GPU_PARTITION=${GPU_PARTITION:-gpu_a100}
      GPU_NODES=${GPU_NODES:-1}
      GPU_NTASKS=${GPU_NTASKS:-1}
      GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-18}
      GPU_COUNT=${GPU_COUNT:-1}
      GPU_TIME=${GPU_TIME:-05:00:00}
      echo "[generate_dataset job] Submitting GPU job for STAGE=${STAGE} (${GPU_PARTITION})."
      exec sbatch \
        --job-name="${JOB_NAME}" \
        --nodes="${GPU_NODES}" \
        --ntasks="${GPU_NTASKS}" \
        --cpus-per-task="${GPU_CPUS_PER_TASK}" \
        --gpus="${GPU_COUNT}" \
        --partition="${GPU_PARTITION}" \
        --time="${GPU_TIME}" \
        --output="${SBATCH_OUTPUT}" \
        --error="${SBATCH_ERROR}" \
        --export=ALL,AUTO_SUBMIT=0 \
        "$0"
      ;;
    *)
      echo "[generate_dataset job] Unknown STAGE='${STAGE}'. Use emulator, halos, hod, or all." >&2
      exit 2
      ;;
  esac
fi

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
if [ "${STAGE}" = "halos" ] || [ "${STAGE}" = "hod" ] || [ "${STAGE}" = "all" ]; then
  module load OpenMPI/5.0.3-GCC-13.3.0
fi
cd "${REPO_DIR}"

STAGE=${STAGE:-emulator}
VENV_PATH=${VENV_PATH:-}
HOD_VENV_PATH=${HOD_VENV_PATH:-}
HOD_USE_SYSTEM_PYTHON=${HOD_USE_SYSTEM_PYTHON:-0}
HOD_PYTHONNOUSERSITE=${HOD_PYTHONNOUSERSITE:-0}

if [ "${STAGE}" = "halos" ]; then
  if [ -n "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
  else
    source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  fi
  export PYTHONNOUSERSITE=1
elif [ "${STAGE}" = "hod" ]; then
  if [ "${HOD_USE_SYSTEM_PYTHON}" -eq 1 ]; then
    echo "[generate_dataset job] STAGE=hod using module/system Python environment (no venv activation)."
  elif [ -n "${HOD_VENV_PATH}" ]; then
    source "${HOD_VENV_PATH}"
  elif [ -n "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
  else
    source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  fi
  export PYTHONNOUSERSITE="${HOD_PYTHONNOUSERSITE}"
elif [ "${STAGE}" = "all" ]; then
  if [ -n "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
  else
    source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
    echo "[generate_dataset job] STAGE=all selected. .venv is used by default; set VENV_PATH if you have a combined env with nbodykit + emulator + pyhod deps."
  fi
  export PYTHONNOUSERSITE=0
  export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
else
  source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
  export PYTHONNOUSERSITE=0
  export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
fi

if [ "${STAGE}" = "halos" ] || [ "${STAGE}" = "hod" ]; then
  RUN_NTASKS=${SLURM_NTASKS:-1}
  RUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
else
  RUN_NTASKS=1
  RUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-18}
fi
export OMP_NUM_THREADS=${RUN_CPUS_PER_TASK}

DATASET_ROOT=${DATASET_ROOT:-/gpfs/scratch1/shared/osavchenko/emulator_datasets}
DATASET_NAME=${DATASET_NAME:-}
N_SIMS=${N_SIMS:-100}
SEED_START=${SEED_START:-0}
REUSE_EXISTING=${REUSE_EXISTING:-1}
CLEANUP_TEMP_DISPLACEMENT=${CLEANUP_TEMP_DISPLACEMENT:-1}

N_PART=${N_PART:-128}
RES=${RES:-64}
BOXSIZE=${BOXSIZE:-250}
NDIV=${NDIV:-1}
EMU_PRECISION=${EMU_PRECISION:-f16}
NUM_SIMS_INTERNAL=${NUM_SIMS_INTERNAL:-1}
MAS_WORDER=${MAS_WORDER:-2}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}
DELTA_UPSAMPLE_METHOD=${DELTA_UPSAMPLE_METHOD:-mode_inject}
DOWNSAMPLE_METHOD=${DOWNSAMPLE_METHOD:-fourier_truncation}
DOWNSAMPLE_GAUSSIAN_SIGMA_MPC_H=${DOWNSAMPLE_GAUSSIAN_SIGMA_MPC_H:-}
DOWNSAMPLE_THREADS=${DOWNSAMPLE_THREADS:-1}

OMEGA_M=${OMEGA_M:-0.3175}
LINKING_LENGTH=${LINKING_LENGTH:-0.2}
ABSOLUTE_LINKING=${ABSOLUTE_LINKING:-0}
NMIN=${NMIN:-20}
CATALOG_FILE=${CATALOG_FILE:-fof_catalog.npz}

HOD_CATALOG_FILE=${HOD_CATALOG_FILE:-hod_catalog.npz}
HOD_LOGMMIN=${HOD_LOGMMIN:-13.0}
HOD_SIGMALOGM=${HOD_SIGMALOGM:-0.2}
HOD_LOGM0=${HOD_LOGM0:-13.1}
HOD_ALPHA=${HOD_ALPHA:-0.75}
HOD_LOGM1=${HOD_LOGM1:-14.3}
HOD_METHOD=${HOD_METHOD:-NFW}
HOD_RSD=${HOD_RSD:-0}
HOD_SEED_POLICY=${HOD_SEED_POLICY:-sim_index}
HOD_SEED=${HOD_SEED:-42}
HOD_OMEGA_M=${HOD_OMEGA_M:-0.3089}
HOD_OMEGA_L=${HOD_OMEGA_L:-0.6911}
HOD_HUBBLE=${HOD_HUBBLE:-0.6774}
HOD_REDSHIFT=${HOD_REDSHIFT:-0.0}
HOD_M_STAR=${HOD_M_STAR:-1.670992e13}
HOD_DELTAC=${HOD_DELTAC:-200}
HOD_ZERR=${HOD_ZERR:-0.0}
HOD_EXTRA_ARGS=${HOD_EXTRA_ARGS:-}

EXTRA_ARGS=()
if [ -n "${DATASET_NAME}" ]; then EXTRA_ARGS+=(--dataset-name "${DATASET_NAME}"); fi
if [ "${REUSE_EXISTING}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi
if [ "${CLEANUP_TEMP_DISPLACEMENT}" -eq 1 ]; then EXTRA_ARGS+=(--cleanup-temp-displacement); else EXTRA_ARGS+=(--no-cleanup-temp-displacement); fi
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${ABSOLUTE_LINKING}" -eq 1 ]; then EXTRA_ARGS+=(--absolute-linking); else EXTRA_ARGS+=(--no-absolute-linking); fi
if [ -n "${DOWNSAMPLE_GAUSSIAN_SIGMA_MPC_H}" ]; then EXTRA_ARGS+=(--downsample-gaussian-sigma-mpc-h "${DOWNSAMPLE_GAUSSIAN_SIGMA_MPC_H}"); fi
if [ -n "${HOD_EXTRA_ARGS}" ]; then EXTRA_ARGS+=(--hod-extra-args "${HOD_EXTRA_ARGS}"); fi

srun --ntasks="${RUN_NTASKS}" --cpus-per-task="${RUN_CPUS_PER_TASK}" python scripts/generate_dataset.py \
  --stage "${STAGE}" \
  --dataset-root "${DATASET_ROOT}" \
  --n-sims "${N_SIMS}" \
  --seed-start "${SEED_START}" \
  --n-part "${N_PART}" \
  --res "${RES}" \
  --boxsize "${BOXSIZE}" \
  --ndiv "${NDIV}" \
  --emu-precision "${EMU_PRECISION}" \
  --num-sims-internal "${NUM_SIMS_INTERNAL}" \
  --mas-worder "${MAS_WORDER}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --delta-upsample-method "${DELTA_UPSAMPLE_METHOD}" \
  --downsample-method "${DOWNSAMPLE_METHOD}" \
  --downsample-threads "${DOWNSAMPLE_THREADS}" \
  --omega-m "${OMEGA_M}" \
  --linking-length "${LINKING_LENGTH}" \
  --nmin "${NMIN}" \
  --catalog-file "${CATALOG_FILE}" \
  --hod-catalog-file "${HOD_CATALOG_FILE}" \
  --hod-logMmin "${HOD_LOGMMIN}" \
  --hod-sigmalogM "${HOD_SIGMALOGM}" \
  --hod-logM0 "${HOD_LOGM0}" \
  --hod-alpha "${HOD_ALPHA}" \
  --hod-logM1 "${HOD_LOGM1}" \
  --hod-method "${HOD_METHOD}" \
  --hod-rsd "${HOD_RSD}" \
  --hod-seed-policy "${HOD_SEED_POLICY}" \
  --hod-seed "${HOD_SEED}" \
  --hod-omega-m "${HOD_OMEGA_M}" \
  --hod-omega-l "${HOD_OMEGA_L}" \
  --hod-hubble "${HOD_HUBBLE}" \
  --hod-redshift "${HOD_REDSHIFT}" \
  --hod-m-star "${HOD_M_STAR}" \
  --hod-deltac "${HOD_DELTAC}" \
  --hod-zerr "${HOD_ZERR}" \
  "${EXTRA_ARGS[@]}"
