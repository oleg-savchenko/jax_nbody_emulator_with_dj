#!/bin/bash
# Auto-submitting launcher for scripts/ppc_resimulate_emulator.py.
#
# Stage routing:
# - STAGE=forward -> GPU job (single task)
# - STAGE=halos   -> CPU/MPI job (multi-task)
# - STAGE=hod     -> CPU job (multi-task; root writes HOD outputs)
# - STAGE=all     -> GPU job (single task; requires combined env with emulator + nbodykit + pyhod)
#
# Typical usage:
#   STAGE=forward PPC_SIM_DIR=/path/to/.../ppc/<tag>/<sim_id> bash my_ppc_resimulate_job.sh
#   STAGE=halos   PPC_SIM_DIR=/path/to/.../ppc/<tag>/<sim_id> bash my_ppc_resimulate_job.sh
#   STAGE=hod     PPC_SIM_DIR=/path/to/.../ppc/<tag>/<sim_id> bash my_ppc_resimulate_job.sh
#   STAGE=all     PPC_SIM_DIR=/path/to/.../ppc/<tag>/<sim_id> VENV_PATH=/path/to/combined_env/bin/activate bash my_ppc_resimulate_job.sh

set -euo pipefail

REPO_DIR=/home/osavchenko/jax_nbody_emulator
STAGE=${STAGE:-forward}
AUTO_SUBMIT=${AUTO_SUBMIT:-1}
SBATCH_OUTPUT=${SBATCH_OUTPUT:-${REPO_DIR}/jobs_outputs/%x-%j-%N_slurm.out}
SBATCH_ERROR=${SBATCH_ERROR:-${REPO_DIR}/jobs_outputs/R-%x.%j.err}

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${AUTO_SUBMIT}" -eq 1 ]; then
  case "${STAGE}" in
    halos|hod)
      if [ "${STAGE}" = "hod" ]; then
        JOB_NAME=${JOB_NAME:-ppc_resim_hod}
      else
        JOB_NAME=${JOB_NAME:-ppc_resim_halos}
      fi
      CPU_PARTITION=${CPU_PARTITION:-rome}
      CPU_NODES=${CPU_NODES:-1}
      CPU_NTASKS=${CPU_NTASKS:-16}
      CPU_CPUS_PER_TASK=${CPU_CPUS_PER_TASK:-1}
      CPU_TIME=${CPU_TIME:-05:00:00}
      echo "[ppc_resim job] Submitting CPU/MPI job for STAGE=${STAGE} (${CPU_PARTITION})."
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
    forward|all)
      JOB_NAME=${JOB_NAME:-ppc_resim_${STAGE}}
      GPU_PARTITION=${GPU_PARTITION:-gpu_a100}
      GPU_NODES=${GPU_NODES:-1}
      GPU_NTASKS=${GPU_NTASKS:-1}
      GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-18}
      GPU_COUNT=${GPU_COUNT:-1}
      GPU_TIME=${GPU_TIME:-03:00:00}
      echo "[ppc_resim job] Submitting GPU job for STAGE=${STAGE} (${GPU_PARTITION})."
      if [ "${STAGE}" = "all" ]; then
        echo "[ppc_resim job] STAGE=all requires an environment containing emulator + mpi4py + nbodykit + pyhod dependencies."
      fi
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
      echo "[ppc_resim job] Unknown STAGE='${STAGE}'. Use forward, halos, hod, or all." >&2
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

VENV_PATH=${VENV_PATH:-}
HOD_VENV_PATH=${HOD_VENV_PATH:-}
HOD_USE_SYSTEM_PYTHON=${HOD_USE_SYSTEM_PYTHON:-0}
HOD_PYTHONNOUSERSITE=${HOD_PYTHONNOUSERSITE:-0}
if [ "${STAGE}" = "halos" ]; then
  source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  export PYTHONNOUSERSITE=1
elif [ "${STAGE}" = "hod" ]; then
  if [ "${HOD_USE_SYSTEM_PYTHON}" -eq 1 ]; then
    echo "[ppc_resim job] STAGE=hod using module/system Python environment (no venv activation)."
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
    echo "[ppc_resim job] STAGE=all selected. .venv is active by default."
    echo "[ppc_resim job] Set VENV_PATH to a combined env with nbodykit + emulator + pyhod deps if needed."
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
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
mkdir -p "${MPLCONFIGDIR}"

PPC_SIM_DIR=${PPC_SIM_DIR:-}
if [ -z "${PPC_SIM_DIR}" ]; then
  echo "[ppc_resim job] PPC_SIM_DIR is required (example: .../ppc/<tag>/<sim_id>)." >&2
  exit 2
fi

DATASET_ROOT=${DATASET_ROOT:-}
SIM_ID=${SIM_ID:-}
TRUE_SIM_DIR=${TRUE_SIM_DIR:-}

SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
NUM_SAMPLES=${NUM_SAMPLES:-}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}

N_PART=${N_PART:-}
RES=${RES:-64}
BOXSIZE=${BOXSIZE:-}
NDIV=${NDIV:-1}
EMU_PRECISION=${EMU_PRECISION:-f16}
MAS_WORDER=${MAS_WORDER:-2}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}
MODE_INJECT_SEED_POLICY=${MODE_INJECT_SEED_POLICY:-sample_id}
MODE_INJECT_FIXED_SEED=${MODE_INJECT_FIXED_SEED:-0}
REUSE_EXISTING=${REUSE_EXISTING:-1}

# Leave FoF fields unset by default so script can inherit true-catalog metadata.
LINKING_LENGTH=${LINKING_LENGTH:-}
ABSOLUTE_LINKING=${ABSOLUTE_LINKING:-}
NMIN=${NMIN:-}
OMEGA_M=${OMEGA_M:-}
CLEANUP_TEMP_DISPLACEMENT=${CLEANUP_TEMP_DISPLACEMENT:-1}

HMF_NBINS=${HMF_NBINS:-25}
HMF_MMIN=${HMF_MMIN:-}
HMF_MMAX=${HMF_MMAX:-}

HOD_LOGMMIN=${HOD_LOGMMIN:-13.0}
HOD_SIGMALOGM=${HOD_SIGMALOGM:-0.2}
HOD_LOGM0=${HOD_LOGM0:-13.1}
HOD_ALPHA=${HOD_ALPHA:-0.75}
HOD_LOGM1=${HOD_LOGM1:-14.3}
HOD_METHOD=${HOD_METHOD:-NFW}
HOD_RSD=${HOD_RSD:-0}
HOD_SEED_POLICY=${HOD_SEED_POLICY:-sample_id}
HOD_SEED=${HOD_SEED:-42}
# HOD cosmology = Quijote fiducial, matching the emulator and FoF (was Planck15).
HOD_OMEGA_M=${HOD_OMEGA_M:-0.3175}
HOD_OMEGA_L=${HOD_OMEGA_L:-0.6825}
HOD_HUBBLE=${HOD_HUBBLE:-0.6711}
HOD_REDSHIFT=${HOD_REDSHIFT:-0.0}
HOD_M_STAR=${HOD_M_STAR:-1.670992e13}
HOD_DELTAC=${HOD_DELTAC:-200}
HOD_ZERR=${HOD_ZERR:-0.0}
HOD_EXTRA_ARGS=${HOD_EXTRA_ARGS:-}

MATERIALIZE_TRUE_MASS_FUNCTION_IF_MISSING=${MATERIALIZE_TRUE_MASS_FUNCTION_IF_MISSING:-1}
FORCE_OVERWRITE_TRUE_MASS_FUNCTION=${FORCE_OVERWRITE_TRUE_MASS_FUNCTION:-0}

RUN_PPC=${RUN_PPC:-1}
G2F_RUN_DIR=${G2F_RUN_DIR:-}
G2F_PPC_SCRIPT=${G2F_PPC_SCRIPT:-/home/osavchenko/GNN_FLI/g2f_yannick/g2f_emu/ppc.py}
PPC_N_WORKERS=${PPC_N_WORKERS:-}

MANIFEST_NAME=${MANIFEST_NAME:-ppc_resimulation_manifest.json}

EXTRA_ARGS=()

if [ -n "${DATASET_ROOT}" ]; then EXTRA_ARGS+=(--dataset-root "${DATASET_ROOT}"); fi
if [ -n "${SIM_ID}" ]; then EXTRA_ARGS+=(--sim-id "${SIM_ID}"); fi
if [ -n "${TRUE_SIM_DIR}" ]; then EXTRA_ARGS+=(--true-sim-dir "${TRUE_SIM_DIR}"); fi

if [ -n "${NUM_SAMPLES}" ]; then EXTRA_ARGS+=(--num-samples "${NUM_SAMPLES}"); fi
if [ -n "${N_PART}" ]; then EXTRA_ARGS+=(--n-part "${N_PART}"); fi
if [ -n "${BOXSIZE}" ]; then EXTRA_ARGS+=(--boxsize "${BOXSIZE}"); fi

if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${REUSE_EXISTING}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi

if [ -n "${LINKING_LENGTH}" ]; then EXTRA_ARGS+=(--linking-length "${LINKING_LENGTH}"); fi
if [ -n "${ABSOLUTE_LINKING}" ]; then
  if [ "${ABSOLUTE_LINKING}" -eq 1 ]; then EXTRA_ARGS+=(--absolute-linking); else EXTRA_ARGS+=(--no-absolute-linking); fi
fi
if [ -n "${NMIN}" ]; then EXTRA_ARGS+=(--nmin "${NMIN}"); fi
if [ -n "${OMEGA_M}" ]; then EXTRA_ARGS+=(--omega-m "${OMEGA_M}"); fi
if [ "${CLEANUP_TEMP_DISPLACEMENT}" -eq 1 ]; then EXTRA_ARGS+=(--cleanup-temp-displacement); else EXTRA_ARGS+=(--no-cleanup-temp-displacement); fi

if [ -n "${HMF_MMIN}" ]; then EXTRA_ARGS+=(--hmf-mmin "${HMF_MMIN}"); fi
if [ -n "${HMF_MMAX}" ]; then EXTRA_ARGS+=(--hmf-mmax "${HMF_MMAX}"); fi
if [ -n "${HOD_EXTRA_ARGS}" ]; then EXTRA_ARGS+=(--hod-extra-args "${HOD_EXTRA_ARGS}"); fi

if [ "${MATERIALIZE_TRUE_MASS_FUNCTION_IF_MISSING}" -eq 1 ]; then
  EXTRA_ARGS+=(--materialize-true-mass-function-if-missing)
else
  EXTRA_ARGS+=(--no-materialize-true-mass-function-if-missing)
fi
if [ "${FORCE_OVERWRITE_TRUE_MASS_FUNCTION}" -eq 1 ]; then
  EXTRA_ARGS+=(--force-overwrite-true-mass-function)
else
  EXTRA_ARGS+=(--no-force-overwrite-true-mass-function)
fi

if [ "${RUN_PPC}" -eq 1 ]; then EXTRA_ARGS+=(--run-ppc); else EXTRA_ARGS+=(--no-run-ppc); fi
if [ -n "${G2F_RUN_DIR}" ]; then EXTRA_ARGS+=(--g2f-run-dir "${G2F_RUN_DIR}"); fi
if [ -n "${G2F_PPC_SCRIPT}" ]; then EXTRA_ARGS+=(--g2f-ppc-script "${G2F_PPC_SCRIPT}"); fi
if [ -n "${PPC_N_WORKERS}" ]; then EXTRA_ARGS+=(--ppc-n-workers "${PPC_N_WORKERS}"); fi

srun --ntasks="${RUN_NTASKS}" --cpus-per-task="${RUN_CPUS_PER_TASK}" \
  python scripts/ppc_resimulate_emulator.py \
  --stage "${STAGE}" \
  --ppc-sim-dir "${PPC_SIM_DIR}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --res "${RES}" \
  --ndiv "${NDIV}" \
  --emu-precision "${EMU_PRECISION}" \
  --mas-worder "${MAS_WORDER}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --mode-inject-seed-policy "${MODE_INJECT_SEED_POLICY}" \
  --mode-inject-fixed-seed "${MODE_INJECT_FIXED_SEED}" \
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
  --hmf-nbins "${HMF_NBINS}" \
  --manifest-name "${MANIFEST_NAME}" \
  "${EXTRA_ARGS[@]}"
